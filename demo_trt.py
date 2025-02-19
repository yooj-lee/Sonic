import os
from trt_src.utils.arg_utils import parseArgs
from sonic_tensorrt import SonicTRT


if __name__ == "__main__":
    args = parseArgs()

    pipe = SonicTRT()
    face_info = pipe.preprocess(args.image_path, expand_ratio=0.5)
    print(face_info)
    if face_info['face_num'] >= 0:
        if args.crop:
            crop_image_path = args.image_path + '.crop.png'
            pipe.crop_image(args.image_path, crop_image_path, face_info['crop_bbox'])
            args.image_path = crop_image_path
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        pipe.process(args.image_path, args.audio_path, args.output_path, min_resolution=512, inference_steps=25, dynamic_scale=args.dynamic_scale)
