#
# Copyright 2024 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuda import cudart
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from ..trt_models import (
    CLIPImageProcessorModel,
    CLIPVisionWithProjModel,
    UNetTemporalModel,
    VAEDecTemporalModel,
)
import os
import pathlib
import tensorrt as trt
import time
import torch
import random
from tqdm.auto import tqdm
from typing import Optional
from ..utils.utilities import (
    GiB,
    PIPELINE_TYPE,
    TRT_LOGGER,
    Engine,
    _append_dims,
    _resize_with_antialiasing,
    tensor2vid,
    load_calibration_images,
)
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from ..utils.utils_modelopt import (
    filter_func,
    quantize_lvl,
    check_lora,
    set_fmha,
    generate_fp8_scales,
    SD_FP8_FP16_DEFAULT_CONFIG,
)

from .stable_diffusion_pipeline import StableDiffusionPipeline
from omegaconf import OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SonicPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Video Diffusion pipelines using NVidia TensorRT.
    """
    def __init__(
        self,
        version='svd-xt-1.1',
        pipeline_type=PIPELINE_TYPE.IMG2VID,
        # min_guidance_scale: float = 1.0,
        # max_guidance_scale: float = 3.0,
        decode_chunk_size: Optional[int] = None,
        config: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of [svd-xt-1.1]
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
                `max_guidance_scale = 1` corresponds to doing no classifier free guidance.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
        """
        assert type(config) == str

        super().__init__(
            version=version,
            pipeline_type=pipeline_type,
            **kwargs
        )
        # self.min_guidance_scale = min_guidance_scale
        # self.max_guidance_scale = max_guidance_scale
        # self.do_classifier_free_guidance = max_guidance_scale > 1
        self.do_classifier_free_guidance = True
        # FIXME vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_factor = 8
        # FIXME num_frames = self.config.num_frames
        select_num_frames = {
            'svd-xt-1.1': 25,
        }
        self.num_frames = select_num_frames.get(version, 14)
        # TODO decode_chunk_size from args
        self.decode_chunk_size = 8 if not decode_chunk_size else decode_chunk_size
        # TODO: scaling_factor = vae.config.scaling_factor
        self.scaling_factor = 0.18215

        # TODO user configurable cuda_device_id
        cuda_device_id = 0
        vram_size = cudart.cudaGetDeviceProperties(cuda_device_id)[1].totalGlobalMem
        self.low_vram = vram_size < GiB(40)
        if self.low_vram:
            print(f"[W] WARNING low VRAM ({vram_size/GiB(1):.2f} GB) mode selected. Certain optimizations may be skipped.")
        if self.use_cuda_graph and self.low_vram:
            print(f"[W] WARNING CUDA graph disabled in low VRAM mode.")
            self.use_cuda_graph = False

        self.config = {}
        if self.pipeline_type.is_img2vid():
            self.config['clip_vis_torch_fallback'] = True
            self.config['clip_imgfe_torch_fallback'] = True
            self.config['vae_temp_torch_fallback'] = True

        config = OmegaConf.load(config)
        self.config = OmegaConf.merge(config, self.config)

        # initialized in loadEngines()
        self.max_shared_device_memory_size = 0

    def loadResources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        self.seed = seed
        self.generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # Create CUDA events and stream
        for stage in ['clip', 'denoise', 'vae', 'vae_encoder']:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        # Allocate shared device memory for TensorRT engines
        if not self.low_vram and not self.torch_inference:
            for model_name in self.models.keys():
                if not self.torch_fallback[model_name]:
                    self.max_shared_device_memory_size = max(self.max_shared_device_memory_size, self.engine[model_name].engine.device_memory_size)
            self.shared_device_memory = cudart.cudaMalloc(self.max_shared_device_memory_size)[1]
            # Activate TensorRT engines
            for model_name in self.models.keys():
                if not self.torch_fallback[model_name]:
                    self.engine[model_name].activate(device_memory=self.shared_device_memory)
                    alloc_shape = self.models[model_name].get_shape_dict(batch_size, image_height, image_width)
                    self.engine[model_name].allocate_buffers(shape_dict=alloc_shape, device=self.device)

    def loadEngines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=False,
        static_shape=False,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        fp8=False,
        quantization_level=0.0,
        calibration_size=32,
        calib_batch_size=2
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to store the TensorRT engines.
            framework_model_dir (str):
                Directory to store the framework model ckpt.
            onnx_dir (str):
                Directory to store the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_refit (bool):
                Build engines with refit option enabled.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to speed up TensorRT build.
            fp8 (bool):
                Whether to quantize to fp8 format or not.
            quantization_level (float):
                Controls which layers to quantize.
            calibration_size (int):
                The number of steps to use for calibrating the model for quantization.
            calib_batch_size (int):
                The batch size to use for calibration. Defaults to 2.
        """
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        # Load pipeline models
        models_args = {'version': self.version, 'pipeline': self.pipeline_type, 'device': self.device,
            'hf_token': self.hf_token, 'verbose': self.verbose, 'framework_model_dir': framework_model_dir,
            'max_batch_size': self.max_batch_size}
        if 'clip-vis' in self.stages:
            self.models['clip-vis'] = CLIPVisionWithProjModel(**models_args, subfolder='image_encoder')
        if 'clip-imgfe' in self.stages:
            self.models['clip-imgfe'] = CLIPImageProcessorModel(**models_args, subfolder='feature_extractor')
        if 'unet-temp' in self.stages:
            self.models['unet-temp'] = UNetTemporalModel(**models_args, fp16=True, fp8=fp8, num_frames=self.num_frames, do_classifier_free_guidance=self.do_classifier_free_guidance)
        if 'vae-temp' in self.stages:
            self.models['vae-temp'] = VAEDecTemporalModel(**models_args, decode_chunk_size=self.decode_chunk_size)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # Configure pipeline models to load
        model_names = self.models.keys()
        self.torch_fallback = dict(zip(model_names, [self.torch_inference or self.config.get(model_name.replace('-','_')+'_torch_fallback', False) for model_name in model_names]))
        onnx_path = dict(zip(model_names, [self.getOnnxPath(model_name, onnx_dir, opt=False) for model_name in model_names]))
        onnx_opt_path = dict(zip(model_names, [self.getOnnxPath(model_name, onnx_dir) for model_name in model_names]))
        engine_path = dict(zip(model_names, [self.getEnginePath(model_name, engine_dir) for model_name in model_names]))
        do_engine_refit = dict(zip(model_names, [enable_refit and model_name.startswith('unet') for model_name in model_names]))

        # Quantization.
        model_suffix = dict(zip(model_names, ['' for model_name in model_names]))
        use_fp8 = dict.fromkeys(model_names, False)
        if fp8:
            model_name = "unet-temp"
            use_fp8[model_name] = True
            model_suffix[model_name] += f"-fp8.l{quantization_level}.bs2.s{self.denoising_steps}.c{calibration_size}"
        onnx_path = { model_name : self.getOnnxPath(model_name, onnx_dir, opt=False, suffix=model_suffix[model_name]) for model_name in model_names }
        onnx_opt_path = { model_name : self.getOnnxPath(model_name, onnx_dir, suffix=model_suffix[model_name]) for model_name in model_names }
        engine_path = { model_name : self.getEnginePath(model_name, engine_dir, do_engine_refit[model_name], suffix=model_suffix[model_name]) for model_name in model_names }
        weights_map_path = { model_name : (self.getWeightsMapPath(model_name, onnx_dir) if do_engine_refit[model_name] else None) for model_name in model_names }


        # Export models to ONNX
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            do_export_onnx = not os.path.exists(engine_path[model_name]) and not os.path.exists(onnx_opt_path[model_name])
            do_export_weights_map = weights_map_path[model_name] and not os.path.exists(weights_map_path[model_name])
            if do_export_onnx or do_export_weights_map:
                if use_fp8[model_name]:
                    pipeline = obj.get_pipeline()
                    model = pipeline.unet

                    state_dict_path = self.getStateDictPath(model_name, onnx_dir, suffix=model_suffix[model_name])
                    if not os.path.exists(state_dict_path):
                        # Load calibration images
                        print(f"[I] Calibrated weights not found, generating {state_dict_path}")
                        calibration_image_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'calibration-images')
                        calibration_image_list = load_calibration_images(calibration_image_folder)
                        print("Number of images loaded:", len(calibration_image_list))

                        # TODO check size > calibration_size
                        def do_calibrate(pipeline, calibration_images, **kwargs):
                            for i_th, image in enumerate(calibration_images):
                                if i_th >= kwargs["calib_size"]:
                                    return
                                pipeline(
                                    image=image,
                                    num_inference_steps=kwargs["n_steps"],
                                ).frames[0]

                        def forward_loop(model):
                            pipeline.unet = model
                            do_calibrate(
                                pipeline=pipeline,
                                calibration_images=calibration_image_list,
                                calib_size=calibration_size // calib_batch_size,
                                n_steps=self.denoising_steps,
                            )

                        print(f"[I] Performing calibration for {calibration_size} steps.")
                        if use_fp8[model_name]:
                            quant_config = SD_FP8_FP16_DEFAULT_CONFIG
                        check_lora(model)
                        mtq.quantize(model, quant_config, forward_loop)
                        mto.save(model, state_dict_path)
                    else:
                        mto.restore(model, state_dict_path)

                    print(f"[I] Generating quantized ONNX model: {onnx_opt_path[model_name]}")
                    if not os.path.exists(onnx_path[model_name]):
                        """
                            Error: Torch bug, ONNX export failed due to unknown kernel shape in QuantConv3d.
                            TRT_FP8QuantizeLinear and TRT_FP8DequantizeLinear operations in UNetSpatioTemporalConditionModel for svd
                            cause issues. Inputs on different devices (CUDA vs CPU) may contribute to the problem.
                        """
                        quantize_lvl(self.version, model, quantization_level, enable_conv_3d=False)
                        mtq.disable_quantizer(model, filter_func)
                        if use_fp8[model_name]:
                            generate_fp8_scales(model)
                    else:
                        model = None

                    obj.export_onnx(onnx_path[model_name], onnx_opt_path[model_name], onnx_opset, opt_image_height, opt_image_width, custom_model=model, static_shape=static_shape)
                else:
                    obj.export_onnx(onnx_path[model_name], onnx_opt_path[model_name], onnx_opset, opt_image_height, opt_image_width)

        # Clean model cache
        torch.cuda.empty_cache()

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            engine = Engine(engine_path[model_name])
            if not os.path.exists(engine_path[model_name]):
                update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
                engine.build(onnx_opt_path[model_name],
                    fp16=True,
                    input_profile=obj.get_input_profile(
                        opt_batch_size, opt_image_height, opt_image_width,
                        static_batch=static_batch, static_shape=static_shape,
                    ),
                    enable_refit=do_engine_refit[model_name],
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    update_output_names=update_output_names,
                    native_instancenorm=False,
)
            self.engine[model_name] = engine # self.engine -> dictionary type

        # Load TensorRT engines
        for model_name in self.models.keys():
            if not self.torch_fallback[model_name]:
                self.engine[model_name].load()

    def activateEngines(self, model_name, alloc_shape=None):
        if not self.torch_fallback[model_name]:
            device_memory_update = self.low_vram and not self.shared_device_memory
            if device_memory_update:
                assert not self.use_cuda_graph
                # Reclaim GPU memory from torch cache
                torch.cuda.empty_cache()
                self.shared_device_memory = cudart.cudaMalloc(self.max_shared_device_memory_size)[1]
            # Create TensorRT execution context
            if not self.engine[model_name].context:
                assert not self.use_cuda_graph
                self.engine[model_name].activate(device_memory=self.shared_device_memory)
            if device_memory_update:
                self.engine[model_name].reactivate(device_memory=self.shared_device_memory)
            if alloc_shape and not self.engine[model_name].tensors:
                assert not self.use_cuda_graph
                self.engine[model_name].allocate_buffers(shape_dict=alloc_shape, device=self.device)
        else:
            # Load torch model
            if not model_name in self.torch_models:
                self.torch_models[model_name] = self.models[model_name].get_model(torch_inference=self.torch_inference)

    def deactivateEngines(self, model_name, release_model=True):
        if not release_model:
            return
        if not self.torch_fallback[model_name]:
            assert not self.use_cuda_graph
            self.engine[model_name].deallocate_buffers()
            self.engine[model_name].deactivate()
            # Shared device memory deallocated only in low VRAM mode
            if self.low_vram and self.shared_device_memory:
                cudart.cudaFree(self.shared_device_memory)
                self.shared_device_memory = None
        else:
            del self.torch_models[model_name]

    def print_summary(self, denoising_steps, walltime_ms, batch_size, num_frames):
        print('|-----------------|--------------|')
        print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('VAE-Enc', cudart.cudaEventElapsedTime(self.events['vae_encoder'][0], self.events['vae_encoder'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('CLIP', cudart.cudaEventElapsedTime(self.events['clip'][0], self.events['clip'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('UNet'+('+CNet' if self.pipeline_type.is_controlnet() else '')+' x '+str(denoising_steps), cudart.cudaEventElapsedTime(self.events['denoise'][0], self.events['denoise'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('VAE-Dec', cudart.cudaEventElapsedTime(self.events['vae'][0], self.events['vae'][1])[1]))
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', walltime_ms))
        print('|-----------------|--------------|')
        print('Throughput: {:.2f} videos/min ({} frames)'.format(batch_size*60000./walltime_ms, num_frames))

    def save_video(self, frames, pipeline, seed):
        video_name_prefix = '-'.join([pipeline, 'fp16', str(seed), str(random.randint(1000,9999))])
        video_name_suffix = 'torch' if self.torch_inference else 'trt'
        video_path = video_name_prefix+'-'+video_name_suffix+'.gif'
        print(f"Saving video to: {video_path}")
        frames[0].save(os.path.join(self.output_dir, video_path), save_all=True,optimize=False, append_images=frames[1:], loop=0)
    
    def _clip_encode_image(self, image, audio_prompts, uncond_audio_prompts, num_frames, device, num_videos_per_prompt, do_classifier_free_guidance, frames_per_batch):
        dtype = next(self.torch_models['clip-vis'].parameters()).dtype

        image = image.to(device=self.device, dtype=dtype)
        image_embeddings = self.torch_models['clip-vis'](image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        # image_embeddings 여기서 num_frames로 확장해줌
        image_embeddings = image_embeddings.unsqueeze(1).repeat((1, num_frames, 1, 1))
        
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # audio_prompts 추가해서 쓰기
            audio_prompts = torch.stack(audio_prompts, dim=0).to(device=device, dtype=dtype)
            audio_prompts = audio_prompts.unsqueeze(0)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings, image_embeddings])


            uncond_audio_prompts = torch.stack(uncond_audio_prompts, dim=0).to(device=device, dtype=dtype)
            uncond_audio_prompts = uncond_audio_prompts.unsqueeze(0)


            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes -> 애초에 이렇게 적어놨구나?
            audio_prompts = torch.cat([uncond_audio_prompts, uncond_audio_prompts, audio_prompts])

        return image_embeddings, audio_prompts
    
    # [NOTE] 여기 있는 코드가 원래 sonic 코드에서는 Sonic 클래스 내부에서 활용되는 걸로 보이는데...
    def _encode_image(self, image, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.torch_models['clip-vis'].parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.torch_models['clip-imgfe'](
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=self.device, dtype=dtype)
        image_embeddings = self.torch_models['clip-vis'](image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.torch_models['vae-temp'].encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids]) # 이 부분을 왜 두 번?

        return add_time_ids

    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        latents=None,
        ref_image_latents=None,
        timestep=None
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            noise = randn_tensor(shape, generator=self.generator, device=device, dtype=dtype)
        else:
            noise = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if timestep is not None:
            init_latents = ref_image_latents.unsqueeze(1)
            latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = noise * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents, num_frames, decode_chunk_size):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.scaling_factor * latents

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            # TODO only pass num_frames_in if it's expected
            if self.torch_fallback['vae-temp']:
                frame = self.torch_models['vae-temp'].decode(latents[i : i + decode_chunk_size], num_frames=num_frames_in).sample
            else:
                params = {
                    'latent': latents[i : i + decode_chunk_size],
                    # FIXME segfault
                    #'num_frames_in': torch.Tensor([num_frames_in]).to(device=latents.device, dtype=torch.int64),
                }
                frame = self.runEngine('vae-temp', params)['frames']
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames
    
    def get_timesteps(self, num_inference_steps, strength):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def infer(
        self,
        ref_image,
        clip_image,
        face_mask,
        audio_prompts,
        uncond_audio_prompts,
        motion_buckets,
        num_frames = None,
        num_inference_steps = 25,
        min_guidance_scale1=1.0, # 1.0,
        max_guidance_scale1=3.0,
        min_guidance_scale2=1.0, # 1.0,
        max_guidance_scale2=3.0,
        image_height: int = 512,
        image_width: int = 512,
        fps: int = 7,
        motion_bucket_scale = 1.0,
        # motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        num_videos_per_prompt: Optional[int] = 1,
        latents=None,
        warmup: bool = False,
        # save_video: bool = True,
        overlap = 7,
        shift_offset = 3,
        frames_per_batch = 14,
        i2i_noise_strength = 1.0,
    ):
        """
        Run the video diffusion pipeline.

        Args:
            input_image (image):
                Input image used to initialize the latents or to be inpainted.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the image, the higher it is the less the video will look like the init image. Increase it for more motion.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            warmup (bool):
                Indicate if this is a warmup run.
            save_video (bool):
                Save the video image.
        """

        if self.generator and self.seed:
            self.generator.manual_seed(self.seed)

        # TODO
        batch_size = 1
        # Fast warmup
        denoising_steps = 1 if warmup else self.denoising_steps

        torch.cuda.synchronize()
        e2e_tic = time.perf_counter()

        class LoadModelContext:
            def __init__(ctx, model_name, alloc_shape=None, release_model=False):
                ctx.model_name = model_name
                ctx.release_model = release_model
                ctx.alloc_shape = alloc_shape
            def __enter__(ctx):
                self.activateEngines(ctx.model_name, alloc_shape=ctx.alloc_shape)
            def __exit__(ctx, exc_type, exc_val, exc_tb):
                self.deactivateEngines(ctx.model_name, release_model=ctx.release_model)

        # Release model opportunistically in TensorRT pipeline only in low VRAM mode
        release_model = self.low_vram and not self.torch_inference
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # with LoadModelContext('clip-imgfe', release_model=release_model), LoadModelContext('clip-vis', release_model=release_model):
            #     self.profile_start('clip', color='green')
            #     image_embeddings = self._encode_image(input_image, num_videos_per_prompt, self.do_classifier_free_guidance)
            #     self.profile_stop('clip')

            with LoadModelContext('clip-vis', release_model=release_model):
                self.profile_start('clip', color='green')
                image_embeddings, audio_prompts = self._clip_encode_image(
                    clip_image, # 애초에 이 clip_image 자체가 feature extractor 결과물이지 않나? 그냥 context 두 개 열어서 해도 될 거 같다
                    audio_prompts,
                    uncond_audio_prompts,
                    num_frames,
                    self.device,
                    num_videos_per_prompt,
                    self.do_classifier_free_guidance,
                    frames_per_batch
                )
                self.profile_stop('clip')
            # NOTE Stable Diffusion Video was conditioned on fps - 1
            fps = fps - 1

            motion_buckets = torch.stack(motion_buckets, dim = 0).to(device=self.device)
            motion_buckets = motion_buckets.unsqueeze(0)

            # self.profile_start('preprocess', color='pink')
            # input_image = self.image_processor.preprocess(input_image, height=image_height, width=image_width).to(self.device)
            # noise = randn_tensor(input_image.shape, generator=self.generator, device=input_image.device, dtype=input_image.dtype)
            # input_image = input_image + noise_aug_strength * noise
            # self.profile_stop('preprocess')

            # TODO
            # assert self.torch_models['vae-temp'].dtype == torch.float32

            with LoadModelContext('vae-temp'):
                self.profile_start('vae_encoder', color='red')

                # for reference image
                ref_image_tensor = ref_image.to(
                    dtype = self.torch_models['vae-temp'].dtype,
                ) # 애초에 ref_image 자체가 torch.Tensor 형태로 들어오긴 함
                ref_image_latents = self.torch_models['vae-temp'].encode(ref_image_tensor).latent_dist.mean
                ref_image_latents = ref_image_latents * self.scaling_factor

                noise = randn_tensor(
                    ref_image.shape,
                    generator = self.generator,
                    device = ref_image.device,
                    dtype = ref_image.dtype,
                )

                ref_image_tensor = ref_image_tensor + noise_aug_strength * noise

                image_latents = self._encode_vae_image(ref_image_tensor, self.device, num_videos_per_prompt, self.do_classifier_free_guidance)
                image_latents = image_latents.to(image_embeddings.dtype)
                ref_image_latents = ref_image_latents.to(image_embeddings.dtype)

                self.profile_stop('vae_encoder')

            # Repeat the image latents for each frame so we can concatenate them with the noise
            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            image_latents = image_latents.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
            motion_buckets = motion_buckets * motion_bucket_scale

            # # Get Added Time IDs
            # added_time_ids = self._get_add_time_ids(
            #     fps,
            #     motion_bucket_id,
            #     noise_aug_strength,
            #     image_embeddings.dtype,
            #     batch_size,
            #     num_videos_per_prompt,
            #     self.do_classifier_free_guidance,
            # )
            # added_time_ids = added_time_ids.to(self.device)

            # Prepare timesteps
            self.scheduler.set_timesteps(denoising_steps, device=self.device)
            # timesteps = self.scheduler.timesteps
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, i2i_noise_strength)
            latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
            
            # Prepare latent variables
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_frames,
                8, # TODO: num_channels_latents = unet.config.in_channels
                image_height,
                image_width,
                image_embeddings.dtype,
                ref_image.device,
                latents,
                ref_image_latents,
                timestep=latent_timestep,
                # None, # pre-generated latents
            ) # ref_image_latents' shape: [1, 4, 64, 64]

            # Prepare a list of pose condition images
            face_mask = face_mask.to(
                device=self.device, dtype=latents.dtype
            )[:, :1]

            # Prepare guidance scale
            guidance_scale = torch.linspace(min_guidance_scale1, max_guidance_scale1, num_inference_steps)
            guidance_scale1 = guidance_scale.to(self.device, latents.dtype)
            
            guidance_scale = torch.linspace(
                min_guidance_scale2,
                max_guidance_scale2,
                num_inference_steps
            )
            guidance_scale2 = guidance_scale.to(self.device, latents.dtype)

            self.guidance_scale1 = guidance_scale1
            self.guidance_scale2 = guidance_scale2

            # guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
            # guidance_scale = _append_dims(guidance_scale, latents.ndim)

            # Denoising loop
            # latents_all -> [1, 25, 4, 64, 64]
            latents_all = latents

            num_warmup_steps = len(timesteps) - denoising_steps * self.scheduler.order
            unet_shape_dict = self.models['unet-temp'].get_shape_dict(batch_size, image_height, image_width)
            shift = 0
            with LoadModelContext('unet-temp', alloc_shape=unet_shape_dict, release_model=release_model), tqdm(total=denoising_steps) as progress_bar:
                self.profile_start('denoise', color='blue')
                for i, t in enumerate(timesteps):
                    
                    # init
                    pred_latents = torch.zeros_like(
                        latents_all,
                        dtype=latents.dtype,
                    ).to(device=latents_all.device) # pred_latents' shape: [1, 25, 4, 64, 64]
                    counter = torch.zeros(
                        (latents_all.shape[0], num_frames, 1, 1, 1),
                        dtype=latents.dtype,
                    ).to(device=latents_all.device)

                    # expand the latents if we are doing classifier free guidance
                    for _, index_start in enumerate(range(0, num_frames, frames_per_batch - overlap)):
                        self.scheduler._step_index = None
                        index_start -= shift
                        def indice_slice(tensor, idx_list):
                            tensor_list = []
                            for idx in idx_list:
                                idx = idx % tensor.shape[1]
                                tensor_list.append(tensor[:, idx])
                            return torch.stack(tensor_list, 1)
                        idx_list = list(range(index_start, index_start+frames_per_batch))
                        latents = indice_slice(latents_all, idx_list)
                        image_latents_input = indice_slice(image_latents, idx_list)
                        batch_image_embeddings = indice_slice(image_embeddings, idx_list)
                        batch_audio_prompts = indice_slice(audio_prompts, idx_list)

                        # face_mask: [1,1,512,512] -> [b,c,h,w] (c는 왜 하나만 가져오는 거지? 아니면 c가 아니라 그냥 개수인 걸 수도 있을 듯)
                        latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        latent_model_input = torch.cat([latent_model_input, image_latents_input], dim=2)

                        motion_bucket = indice_slice(motion_buckets, idx_list) # [1,25,1,2,1]로 쪼개질 듯?
                        motion_bucket = torch.mean(motion_bucket, dim=1).squeeze() # [1,1,1,2,1] -> [2] (after squeeze)
                        motion_bucket_id = motion_bucket[0]
                        motion_bucket_id_exp = motion_bucket[1]

                        added_time_ids = self._get_add_time_ids(
                            fps,
                            motion_bucket_id,
                            motion_bucket_id_exp, # instead of noise_aug_strength
                            image_embeddings.dtype,
                            batch_size,
                            num_videos_per_prompt,
                            self.do_classifier_free_guidance,
                        )
                        added_time_ids = added_time_ids.to(self.device, dtype=latents.dtype)
                        
                        # predict the noise residual
                        if self.torch_fallback['unet-temp']:
                            noise_pred = self.torch_models['unet-temp'](
                                latent_model_input,
                                t,
                                encoder_hidden_states=(batch_image_embeddings.flatten(0,1), [batch_audio_prompts.flatten(0,1)]),
                                # cross_attention_kwargs=cross_attention_kwargs,
                                ip_adapter_masks=face_mask,
                                added_time_ids=added_time_ids,
                                return_dict=False,
                            )[0]
                        else:
                            # batch_audio_prompts: [3, 25, 32, 1024]
                            # batch_image_embeddings: [3, 25, 1, 1024]
                            params = {
                                "sample": latent_model_input,
                                "timestep": t,
                                "encoder_hidden_states": torch.cat([batch_image_embeddings, batch_audio_prompts], dim=-2),
                                "added_time_ids": added_time_ids,
                                'ip_adapter_masks': face_mask,
                            }
                            noise_pred = self.runEngine('unet-temp', params)['latent']

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_drop_audio, noise_pred_cond = noise_pred.chunk(3)
                            noise_pred = noise_pred_uncond + self.guidance_scale1[i] * (noise_pred_drop_audio - noise_pred_uncond) + self.guidance_scale2[i] * (noise_pred_cond - noise_pred_drop_audio)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t.to(latents.dtype), latents).prev_sample

                        # if batch == 0:
                        for iii in range(frames_per_batch):
                            p = (index_start + iii) % pred_latents.shape[1]
                            pred_latents[:, p] += latents[:, iii]
                            counter[:, p] += 1
                    shift += shift_offset

                    pred_latents = pred_latents / counter
                    latents_all = pred_latents

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                
                self.profile_stop('denoise')    
            latents = latents_all
            

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER), LoadModelContext('vae-temp'):
            self.profile_start('vae', color='red')
            self.torch_models['vae-temp'].to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, self.decode_chunk_size)
            # frames = tensor2vid(frames, self.image_processor, output_type='pil')
            self.profile_stop('vae')

        torch.cuda.synchronize()

        if warmup:
            return

        e2e_toc = time.perf_counter()
        walltime_ms = (e2e_toc - e2e_tic) * 1000.
        self.print_summary(denoising_steps, walltime_ms, batch_size, len(frames[0]))
        # if save_video:
        #     self.save_video(frames[0], self.pipeline_type.name.lower(), self.seed)

        return frames #, walltime_ms

    # def run(self, input_image, height, width, batch_size, batch_count, num_warmup_runs, use_cuda_graph, **kwargs):
    #     num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
    #     if num_warmup_runs > 0:
    #         print("[I] Warming up ..")
    #         for _ in range(num_warmup_runs):
    #             self.infer(input_image, height, width, warmup=True)

    #     for _ in range(batch_count): # batch_count는 디폴트로 1로 지정.
    #         print("[I] Running StableDiffusion pipeline")
    #         if self.nvtx_profile:
    #             cudart.cudaProfilerStart()
    #         self.infer(input_image, height, width, warmup=False)
    #         if self.nvtx_profile:
    #             cudart.cudaProfilerStop()

    def run(self, image_encoder, wav_enc, audio_pe, audio2bucket, width, height, batch, batch_count, num_warmup_runs, use_cuda_graph, **kwargs):
        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        assert batch_count == 1
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(self.device).float()
                
        ref_img = batch['ref_img']
        clip_img = batch['clip_images']
        face_mask = batch['face_mask']
        # image_embeds = self.torch_models['clip-vis'](
        #     clip_img
        #         ).image_embeds # activate을 call (infer 메소드 안에서 가능함) 해야 torch_models에 모델이 assign이 되는데 이래서 안되는 듯... 이걸 infer 안에서 할 수 있게 하는 게 나을 것 같다
        image_embeds = image_encoder(
            clip_img
        ).image_embeds # 이런 부분도 좋지 못한 부분이긴 함
        audio_feature = batch['audio_feature']
        audio_len = batch['audio_len']
        step = int(self.config.step)

        window = 3000
        audio_prompts = []
        last_audio_prompts = []
        for i in range(0, audio_feature.shape[-1], window):
            audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window], output_hidden_states=True).hidden_states
            last_audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window]).last_hidden_state
            last_audio_prompt = last_audio_prompt.unsqueeze(-2)
            audio_prompt = torch.stack(audio_prompt, dim=2)
            audio_prompts.append(audio_prompt)
            last_audio_prompts.append(last_audio_prompt)

        audio_prompts = torch.cat(audio_prompts, dim=1)
        audio_prompts = audio_prompts[:,:audio_len*2]
        audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:,:4]), audio_prompts, torch.zeros_like(audio_prompts[:,:6])], 1)

        last_audio_prompts = torch.cat(last_audio_prompts, dim=1)
        last_audio_prompts = last_audio_prompts[:,:audio_len*2]
        last_audio_prompts = torch.cat([torch.zeros_like(last_audio_prompts[:,:24]), last_audio_prompts, torch.zeros_like(last_audio_prompts[:,:26])], 1)


        ref_tensor_list = []
        audio_tensor_list = []
        uncond_audio_tensor_list = []
        motion_buckets = []
        for i in tqdm(range(audio_len//step)):


            audio_clip = audio_prompts[:,i*2*step:i*2*step+10].unsqueeze(0)
            audio_clip_for_bucket = last_audio_prompts[:,i*2*step:i*2*step+50].unsqueeze(0)
            motion_bucket = audio2bucket(audio_clip_for_bucket, image_embeds)
            motion_bucket = motion_bucket * 16 + 16
            motion_buckets.append(motion_bucket[0])

            cond_audio_clip = audio_pe(audio_clip).squeeze(0)
            uncond_audio_clip = audio_pe(torch.zeros_like(audio_clip)).squeeze(0)

            ref_tensor_list.append(ref_img[0])
            audio_tensor_list.append(cond_audio_clip[0])
            uncond_audio_tensor_list.append(uncond_audio_clip[0])

        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                video = self.infer(
                ref_img,
                clip_img,
                face_mask,
                audio_tensor_list,
                uncond_audio_tensor_list,
                motion_buckets,
                image_height=height,
                image_width=width,
                num_frames=len(audio_tensor_list),
                motion_bucket_scale=self.config.motion_bucket_scale,
                fps=self.config.fps,
                noise_aug_strength=self.config.noise_aug_strength,
                min_guidance_scale1=self.config.min_appearance_guidance_scale, # 1.0,
                max_guidance_scale1=self.config.max_appearance_guidance_scale,
                min_guidance_scale2=self.config.audio_guidance_scale, # 1.0,
                max_guidance_scale2=self.config.audio_guidance_scale,
                overlap=self.config.overlap, # 0
                shift_offset=self.config.shift_offset, # 7
                frames_per_batch=self.config.n_sample_frames,
                num_inference_steps=self.config.num_inference_steps,
                i2i_noise_strength=self.config.i2i_noise_strength,
                warmup=True
            )


        print("[I] Running StableDiffusion pipeline")
        if self.nvtx_profile:
            cudart.cudaProfilerStart()
        video = self.infer(
        ref_img,
        clip_img,
        face_mask,
        audio_tensor_list,
        uncond_audio_tensor_list,
        motion_buckets,
        image_height=height,
        image_width=width,
        num_frames=len(audio_tensor_list),
        motion_bucket_scale=self.config.motion_bucket_scale,
        fps=self.config.fps,
        noise_aug_strength=self.config.noise_aug_strength,
        min_guidance_scale1=self.config.min_appearance_guidance_scale, # 1.0,
        max_guidance_scale1=self.config.max_appearance_guidance_scale,
        min_guidance_scale2=self.config.audio_guidance_scale, # 1.0,
        max_guidance_scale2=self.config.audio_guidance_scale,
        overlap=self.config.overlap,
        shift_offset=self.config.shift_offset,
        frames_per_batch=self.config.n_sample_frames,
        num_inference_steps=self.config.num_inference_steps,
        i2i_noise_strength=self.config.i2i_noise_strength,
        warmup=False
        )
        if self.nvtx_profile:
            cudart.cudaProfilerStop()
            

        # Concat it with pose tensor
        # pose_tensor = torch.stack(pose_tensor_list,1).unsqueeze(0)
        video = (video*0.5 + 0.5).clamp(0, 1)
        video = torch.cat([video.to(self.device)], dim=0).cpu()

        return video
