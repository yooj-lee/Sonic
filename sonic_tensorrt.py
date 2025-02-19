import argparse
import os
import torch
import torch.utils.checkpoint
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from transformers import WhisperModel, AutoFeatureExtractor, CLIPVisionModelWithProjection

from trt_src.pipelines.pipeline_sonic import SonicPipeline
from trt_src.utils.utilities import (
    PIPELINE_TYPE,
    add_arguments,
    download_image,
)
from trt_src.utils.arg_utils import get_args

from src.utils.util import save_videos_grid, seed_everything
from src.dataset.test_preprocess import process_bbox, image_audio_to_tensor
from src.models.audio_adapter.audio_proj import AudioProjModel
from src.models.audio_adapter.audio_to_bucket import Audio2bucketModel
from src.utils.RIFE.RIFE_HDv3 import RIFEModel
from src.dataset.face_align.align import AlignImage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SonicTRT:
    config_file = os.path.join(BASE_DIR, 'config/inference/sonic.yaml')
    config = OmegaConf.load(config_file)

    def __init__(self, 
                 device_id=0,
                 enable_interpolate_frame=True,
                 ):
        
        config = self.config
        config.use_interframe = enable_interpolate_frame

        device = 'cuda:{}'.format(device_id) if device_id > -1 else 'cpu'

        # config.pretrained_model_name_or_path = os.path.join(BASE_DIR, config.pretrained_model_name_or_path)
        
        audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024, context_tokens=32).to(device)
        audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024, intermediate_dim=1024, output_dim=1, context_tokens=2).to(device)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path, 
            subfolder="image_encoder",
            variant="fp16") # TODO: 나중에 제거. 일단은 디버깅 용도.
        audio2token_checkpoint_path = os.path.join(BASE_DIR, config.audio2token_checkpoint_path)
        audio2bucket_checkpoint_path = os.path.join(BASE_DIR, config.audio2bucket_checkpoint_path)
        
        audio2token.load_state_dict(
            torch.load(audio2token_checkpoint_path, map_location="cpu"),
            strict=True,
        )

        audio2bucket.load_state_dict(
            torch.load(audio2bucket_checkpoint_path, map_location="cpu"),
            strict=True,
        )
        

        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        elif config.weight_dtype == "fp32":
            weight_dtype = torch.float32
        elif config.weight_dtype == "bf16":
            weight_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Do not support weight dtype: {config.weight_dtype} during training"
            )

        whisper = WhisperModel.from_pretrained(os.path.join(BASE_DIR, 'checkpoints/whisper-tiny/')).to(device).eval()
        
        whisper.requires_grad_(False)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(os.path.join(BASE_DIR, 'checkpoints/whisper-tiny/'))

        det_path = os.path.join(BASE_DIR, os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt'))
        self.face_det = AlignImage(device, det_path=det_path)
        if config.use_interframe:
            rife = RIFEModel(device=device)
            rife.load_model(os.path.join(BASE_DIR, 'checkpoints', 'RIFE/'))
            self.rife = rife

        args, kwargs_init_pipeline, kwargs_load_engine, kwargs_run_demo = get_args()

        self.kwargs_run_demo = kwargs_run_demo

        pipe = SonicPipeline(
         pipeline_type = PIPELINE_TYPE.IMG2VID,
         **kwargs_init_pipeline
        )

        # load engines
        pipe.loadEngines(
            args.engine_dir,
            args.framework_model_dir,
            args.onnx_dir,
            **kwargs_load_engine
        )

        # load resources
        pipe.loadResources(args.height, args.width, args.batch_size, args.seed)

        self.pipe = pipe
        self.whisper = whisper
        self.audio2token = audio2token
        self.audio2bucket = audio2bucket
        self.image_encoder = image_encoder

        print('init done')


    # 가장 큰 얼굴만 리턴
    def preprocess(self,
              image_path, expand_ratio=1.0):
        face_image = cv2.imread(image_path)
        h, w = face_image.shape[:2]
        _, _, bboxes = self.face_det(face_image, maxface=True)
        face_num = len(bboxes)
        bbox = []
        if face_num > 0:
            x1, y1, ww, hh = bboxes[0]
            x2, y2 = x1 + ww, y1 + hh
            bbox = x1, y1, x2, y2
            bbox_s = process_bbox(bbox, expand_radio=expand_ratio, height=h, width=w)

        return {
            'face_num': face_num,
            'crop_bbox': bbox_s,
        }
    
    # 이것도 process call하기 전에 얼굴 잘라서 pipeline에 call하긴 하는데 그냥 자른 얼굴 이미지 저장하는 코드로 보임
    def crop_image(self,
                   input_image_path,
                   output_image_path,
                   crop_bbox):
        face_image = cv2.imread(input_image_path)
        crop_image = face_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
        cv2.imwrite(output_image_path, crop_image)

    @torch.no_grad()
    def process(self,
                image_path,
                audio_path,
                output_path,
                min_resolution=512,
                inference_steps=25,
                dynamic_scale=1.0,
                keep_resolution=False,
                seed=None,
                **kwargs_run_demo):
        
        config = self.config
        pipe = self.pipe
        device = pipe.device
        whisper = self.whisper.to(device)
        audio2token = self.audio2token.to(device)
        audio2bucket = self.audio2bucket.to(device)
        image_encoder = self.image_encoder.to(device)


        # specific parameters
        if seed:
            config.seed = seed

        config.num_inference_steps = inference_steps
        config.motion_bucket_scale = dynamic_scale

        seed_everything(config.seed)

        video_path = output_path.replace('.mp4', '_noaudio.mp4')
        audio_video_path = output_path

        imSrc_ = Image.open(image_path).convert('RGB')
        raw_w, raw_h = imSrc_.size

        test_data = image_audio_to_tensor(self.face_det, self.feature_extractor, image_path, audio_path, limit=config.frame_num, image_size=min_resolution, area=config.area)
        if test_data is None:
            return -1
        height, width = test_data['ref_img'].shape[-2:]
        if keep_resolution:
            resolution = f'{raw_w//2*2}x{raw_h//2*2}'
        else:
            resolution = f'{width}x{height}'

        video = pipe.run(
            image_encoder=image_encoder,
            wav_enc=whisper,
            audio_pe=audio2token,
            audio2bucket=audio2bucket,
            width=width,
            height=height,
            batch=test_data,
            batch_count=1,
            num_warmup_runs=1,
            use_cuda_graph=False,
            **kwargs_run_demo
            )

        if config.use_interframe:
            rife = self.rife
            out = video.to(device)
            results = []
            video_len = out.shape[2]
            for idx in tqdm(range(video_len-1), ncols=0):
                I1 = out[:, :, idx]
                I2 = out[:, :, idx+1]
                middle = rife.inference(I1, I2).clamp(0, 1).detach()
                results.append(out[:, :, idx])
                results.append(middle)
            results.append(out[:, :, video_len-1])
            video = torch.stack(results, 2).cpu()
        
        save_videos_grid(video, video_path, n_rows=video.shape[0], fps=config.fps * 2 if config.use_interframe else config.fps)
        os.system(f"ffmpeg -i '{video_path}'  -i '{audio_path}' -s {resolution} -vcodec libx264 -acodec aac -crf 18 -shortest '{audio_video_path}' -y; rm '{video_path}'")
        pipe.teardown()

        return 0
        