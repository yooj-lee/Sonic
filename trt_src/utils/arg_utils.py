import argparse
from .utilities import add_arguments

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Img2Vid Demo", conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--version', type=str, default="svd-xt-1.1", choices=["svd-xt-1.1"], help="Version of Stable Video Diffusion")
    parser.add_argument('--input-image', type=str, default="", help="Path to the input image")
    parser.add_argument('--height', type=int, default=576, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Width of image to generate (must be multiple of 8)")
    parser.add_argument('--min-guidance-scale', type=float, default=1.0, help="The minimum guidance scale. Used for the classifier free guidance with first frame")
    parser.add_argument('--max-guidance-scale', type=float, default=3.0, help="The maximum guidance scale. Used for the classifier free guidance with last frame")
    parser.add_argument('--denoising-steps', type=int, default=25, help="Number of denoising steps")
    parser.add_argument('--num-warmup-runs', type=int, default=1, help="Number of warmup runs before benchmarking performance")
    parser.add_argument('--config-path', type=str, default="/home/yujin-lee/Sonic/config/inference/sonic.yaml")
    parser.add_argument('--image_path', type=str, default="examples/image/anime1.png")
    parser.add_argument('--audio_path', type=str, default="examples/wav/sing_female_10s.wav")
    parser.add_argument('--output_path', type=str, default="examples/results/anime-sing_female.mp4")
    parser.add_argument('--dynamic_scale', type=float, default=1.0)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()

def process_pipeline_args(args):

    if not args.input_image:
        args.input_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true"
    # if isinstance(args.input_image, str):
    #     input_image = download_image(args.input_image).resize((args.width, args.height))
    # elif isinstance(args.input_image, Image.Image):
    #     input_image = Image.open(args.input_image)
    # else:
    #     raise ValueError(f"Input image(s) must be of type `PIL.Image.Image` or `str` (URL) but is {type(args.input_image)}")

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but are: {args.image_height} and {args.width}.")

    # TODO enable BS>1
    max_batch_size = 1
    args.build_static_batch = True

    if args.batch_size > max_batch_size:
        raise ValueError(f"Batch size {args.batch_size} is larger than allowed {max_batch_size}.")

    if not args.build_static_batch or args.build_dynamic_shape:
        raise ValueError(f"Dynamic shapes not supported. Do not specify `--build-dynamic-shape`")

    if args.fp8:
        import torch
        device_info = torch.cuda.get_device_properties(0)
        version = device_info.major * 10 + device_info.minor
        if version < 90: # FP8 is only supppoted on Hopper.
            raise ValueError(f"Cannot apply FP8 quantization for GPU with compute capability {version / 10.0}. FP8 is only supppoted on Hopper.")
        args.optimization_level = 4
        print(f"[I] The default optimization level has been set to {args.optimization_level} for FP8.")

    if args.quantization_level == 0.0 and args.fp8:
        args.quantization_level = 3.0
        print("[I] The default quantization level has been set to 3.0 for FP8.")

    kwargs_init_pipeline = {
        'version': args.version,
        'max_batch_size': max_batch_size,
        'denoising_steps': args.denoising_steps,
        'scheduler': args.scheduler,
        # 'min_guidance_scale': args.min_guidance_scale,
        # 'max_guidance_scale': args.max_guidance_scale,
        'output_dir': args.output_dir,
        'hf_token': args.hf_token,
        'verbose': args.verbose,
        'nvtx_profile': args.nvtx_profile,
        'use_cuda_graph': args.use_cuda_graph,
        'framework_model_dir': args.framework_model_dir,
        'torch_inference': args.torch_inference,
        'config': args.config_path,
    }

    kwargs_load_engine = {
        'onnx_opset': args.onnx_opset,
        'opt_batch_size': args.batch_size,
        'opt_image_height': args.height,
        'opt_image_width': args.width,
        'static_batch': args.build_static_batch,
        'static_shape': not args.build_dynamic_shape,
        'enable_all_tactics': args.build_all_tactics,
        'enable_refit': args.build_enable_refit,
        'timing_cache': args.timing_cache,
        'fp8': args.fp8,
        'quantization_level': args.quantization_level,

    }

    kwargs_run_demo = {
        'height': args.height,
        'width': args.width,
        'batch_count': args.batch_count,
        'num_warmup_runs': args.num_warmup_runs, 
        'use_cuda_graph': args.use_cuda_graph
    }


    return kwargs_init_pipeline, kwargs_load_engine, kwargs_run_demo


def get_args():
    args = parseArgs()
    kwargs_init_pipeline, kwargs_load_engine, kwargs_run_demo = process_pipeline_args(args)

    return args, kwargs_init_pipeline, kwargs_load_engine, kwargs_run_demo
