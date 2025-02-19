#
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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any

import diffusers
from diffusers import DiffusionPipeline
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.pipelines.wuerstchen import PaellaVQModel
import json
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
from glob import glob
import onnx_graphsurgeon as gs
from safetensors import safe_open
import os
import sys
from polygraphy.backend.onnx.loader import fold_constants
import re
import tempfile
import torch
from transformers import (
    AutoConfig,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5TokenizerFast
)
from huggingface_hub import hf_hub_download
from .utils.utilities import merge_loras, import_from_diffusers
from .utils.utils_modelopt import (
    convert_zp_fp8,
    cast_resize_io,
    convert_fp16_io,
    cast_fp8_mha_io,
)
from onnxmltools.utils.float16_converter import convert_float_to_float16

from .models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from .models.base.attention_processor import *

# from diffusers.models import UNetSpatioTemporalConditionModel # 디버깅 용도로 일단 따로 import

# List of models to import from diffusers.models
models_to_import = [
    'AutoencoderKL',
    'AutoencoderKLTemporalDecoder',
    'ControlNetModel',
    'UNet2DConditionModel',
    # 'UNetSpatioTemporalConditionModel',
    'StableCascadeUNet',
    'FluxTransformer2DModel',
]
for model in models_to_import:
    globals()[model] = import_from_diffusers(model, 'diffusers.models')

def onnx_graph_needs_external_data(onnx_graph):
    if sys.platform == "win32":
        # ByteSize is broken (wraps around) on Windows
        return True
    else:
        return onnx_graph.ByteSize() > 2147483648


class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        return gs.export_onnx(self.graph) if return_onnx else self.graph

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph_needs_external_data(onnx_graph):
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, 'model.onnx')
            onnx_inferred_path = os.path.join(temp_dir, 'inferred.onnx')
            onnx.save_model(onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, hidden_layer_offset, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers+hidden_layer_offset):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers+hidden_layer_offset):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph

    def fuse_mha_qkv_int8_sq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()

        # mha  : fuse QKV QDQ nodes
        # mhca : fuse KV QDQ nodes
        q_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_q/input_quantizer/DequantizeLinear_output_0"
        )
        k_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_k/input_quantizer/DequantizeLinear_output_0"
        )
        v_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_v/input_quantizer/DequantizeLinear_output_0"
        )

        qs = list(sorted(map(
            lambda x: x.group(0),  # type: ignore
            filter(lambda x: x is not None, [re.match(q_pat, key) for key in keys]),
            )))
        ks = list(sorted(map(
            lambda x: x.group(0),  # type: ignore
            filter(lambda x: x is not None, [re.match(k_pat, key) for key in keys]),
            )))
        vs = list(sorted(map(
            lambda x: x.group(0),  # type: ignore
            filter(lambda x: x is not None, [re.match(v_pat, key) for key in keys]),
            )))

        removed = 0
        assert len(qs) == len(ks) == len(vs), "Failed to collect tensors"
        for q, k, v in zip(qs, ks, vs):
            is_mha = all(["attn1" in tensor for tensor in [q, k, v]])
            is_mhca = all(["attn2" in tensor for tensor in [q, k, v]])
            assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

            if is_mha:
                tensors[k].outputs[0].inputs[0] = tensors[q]
                tensors[v].outputs[0].inputs[0] = tensors[q]
                del tensors[k]
                del tensors[v]
                removed += 2
            else:  # is_mhca
                tensors[k].outputs[0].inputs[0] = tensors[v]
                del tensors[k]
                removed += 1
        print(f"Removed {removed} QDQ nodes")
        return removed # expected 72 for L2.5

    def modify_fp8_graph(self, is_fp16_io=True):
        onnx_graph = gs.export_onnx(self.graph)
        # Convert INT8 Zero to FP8.
        onnx_graph = convert_zp_fp8(onnx_graph)
        # Convert weights and activations to FP16 and insert Cast nodes in FP8 MHA.
        onnx_graph = convert_float_to_float16(onnx_graph, keep_io_types=True, disable_shape_infer=True)
        self.graph = gs.import_onnx(onnx_graph)
        # Add cast nodes to Resize I/O.
        cast_resize_io(self.graph)
        # Convert model inputs and outputs to fp16 I/O.
        if is_fp16_io:
            convert_fp16_io(self.graph)
        # Add cast nodes to MHA's BMM1 and BMM2's I/O.
        cast_fp8_mha_io(self.graph)
    
    def flux_convert_rope_weight_type(self):
        for node in self.graph.nodes:
            if node.op == "Einsum":
                node.inputs[1].dtype == "float32"
                print(f"Fixed RoPE (Rotary Position Embedding) weight type: {node.name}")
        return gs.export_onnx(self.graph)


def get_path(version, pipeline, controlnets=None):
    if controlnets is not None:
        return ["lllyasviel/sd-controlnet-" + modality for modality in controlnets]

    if version in ("1.4", "1.5") and pipeline.is_inpaint():
        return "benjamin-paine/stable-diffusion-v1-5-inpainting"
    elif version == "1.4":
        return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        return "KiwiXR/stable-diffusion-v1-5"
    elif version == 'dreamshaper-7':
        return 'Lykon/dreamshaper-7'
    elif version in ("2.0-base", "2.0") and pipeline.is_inpaint():
        return "stabilityai/stable-diffusion-2-inpainting"
    elif version == "2.0-base":
        return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        return "stabilityai/stable-diffusion-2"
    elif version == "2.1-base":
        return "stabilityai/stable-diffusion-2-1-base"
    elif version == "2.1":
        return "stabilityai/stable-diffusion-2-1"
    elif version == 'xl-1.0' and pipeline.is_sd_xl_base():
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif version == 'xl-1.0' and pipeline.is_sd_xl_refiner():
        return "stabilityai/stable-diffusion-xl-refiner-1.0"
    # TODO SDXL turbo with refiner
    elif version == 'xl-turbo' and pipeline.is_sd_xl_base():
        return "stabilityai/sdxl-turbo"
    elif version == 'sd3':
        return "stabilityai/stable-diffusion-3-medium"
    elif version == 'svd-xt-1.1' and pipeline.is_img2vid():
        return "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    elif version == 'cascade':
        if pipeline.is_cascade_decoder():
            return "stabilityai/stable-cascade"
        else:
            return "stabilityai/stable-cascade-prior"
    elif version == 'flux.1-dev':
        return "black-forest-labs/FLUX.1-dev"
    elif version == 'flux.1-schnell':
        return "black-forest-labs/FLUX.1-schnell"
    elif version == "flux.1-dev-canny":
        return "black-forest-labs/FLUX.1-Canny-dev"
    elif version == "flux.1-dev-depth":
        return "black-forest-labs/FLUX.1-Depth-dev"
    else:
        raise ValueError(f"Unsupported version {version} + pipeline {pipeline.name}")

def get_clip_embedding_dim(version, pipeline):
    if version in ("1.4", "1.5", "dreamshaper-7", "flux.1-dev", "flux.1-schnell", "flux.1-dev-canny", "flux.1-dev-depth"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_base():
        return 768
    elif version in ("sd3"):
        return 4096
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

def get_clipwithproj_embedding_dim(version, pipeline):
    if version in ("xl-1.0", "xl-turbo", "cascade"):
        return 1280
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

def get_unet_embedding_dim(version, pipeline):
    if version in ("1.4", "1.5", "dreamshaper-7"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_base():
        return 2048
    elif version in ("cascade"):
        return 1280
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_refiner():
        return 1280
    elif pipeline.is_img2vid():
        return 1024
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

# FIXME serialization not supported for torch.compile
def get_checkpoint_dir(framework_model_dir, version, pipeline, subfolder):
    return os.path.join(framework_model_dir, version, pipeline, subfolder)

torch_inference_modes = ['default', 'reduce-overhead', 'max-autotune']
# FIXME update callsites after serialization support for torch.compile is added
def optimize_checkpoint(model, torch_inference):
    if not torch_inference or torch_inference == 'eager':
        return model
    assert torch_inference in torch_inference_modes
    return torch.compile(model, mode=torch_inference, dynamic=False, fullgraph=False) # 모델 미리 compile해서 사용하는 구조 -> Torch 2.0으로 버전 up하면서 가장 main이 되는 api

class LoraLoader(StableDiffusionLoraLoaderMixin):
    def __init__(self,
        paths,
        weights,
        scale
    ):
        self.paths = paths
        self.weights = weights
        self.scale = scale

def is_model_cached(model_dir, model_opts, hf_safetensor, model_name="diffusion_pytorch_model"):
    variant = "." + model_opts.get("variant") if "variant" in model_opts else ""
    suffix = ".safetensors" if hf_safetensor else ".bin"
    # WAR with * for larger models that are split into multiple smaller ckpt files
    model_file = model_name + variant + "*" + suffix
    return bool(glob(os.path.join(model_dir, model_file)))

class BaseModel():
    def __init__(self,
        version='1.5',
        pipeline=None,
        device='cuda',
        hf_token='',
        verbose=True,
        framework_model_dir='pytorch_model',
        fp16=False,
        tf32=False,
        bf16=False,
        int8=False,
        fp8=False,
        max_batch_size=16,
        text_maxlen=77,
        embedding_dim=768,
        compression_factor=8
    ):

        self.name = self.__class__.__name__
        self.pipeline = pipeline.name
        self.version = version
        self.path = get_path(version, pipeline)
        self.device = device
        self.hf_token = hf_token
        self.hf_safetensor = not (pipeline.is_inpaint() and version in ("1.4", "1.5"))
        self.verbose = verbose
        self.framework_model_dir = framework_model_dir

        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16
        self.int8 = int8
        self.fp8 = fp8

        self.compression_factor = compression_factor
        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256   # min image resolution: 256x256
        self.max_image_shape = 1344  # max image resolution: 1344x1344
        self.min_latent_shape = self.min_image_shape // self.compression_factor
        self.max_latent_shape = self.max_image_shape // self.compression_factor

        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        self.extra_output_names = []

        self.do_constant_folding = True

    def get_pipeline(self):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        model_opts = {'torch_dtype': torch.bfloat16} if self.bf16 else model_opts
        return DiffusionPipeline.from_pretrained(
            self.path,
            use_safetensors=self.hf_safetensor,
            token=self.hf_token,
            **model_opts,
        ).to(self.device)

    def get_model(self, torch_inference=''):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    # Helper utility for ONNX export
    def export_onnx(
        self,
        onnx_path,
        onnx_opt_path,
        onnx_opset,
        opt_image_height,
        opt_image_width,
        custom_model=None,
        enable_lora_merge=False,
        static_shape=False,
        lora_loader=None
    ):
        onnx_opt_graph = None
        # Export optimized ONNX model (if missing)
        if not os.path.exists(onnx_opt_path):
            if not os.path.exists(onnx_path):
                print(f"[I] Exporting ONNX model: {onnx_path}")
                def export_onnx(model):
                    if enable_lora_merge:
                        assert lora_loader is not None
                        model = merge_loras(model, lora_loader)
                    inputs = self.get_sample_input(1, opt_image_height, opt_image_width, static_shape)
                    torch.onnx.export(model,
                        inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=onnx_opset,
                        do_constant_folding=self.do_constant_folding,
                        input_names=self.get_input_names(),
                        output_names=self.get_output_names(),
                        dynamic_axes=self.get_dynamic_axes(),
                    )
                if custom_model:
                    with torch.inference_mode():
                        export_onnx(custom_model)
                else: # if there's no custom model
                    # WAR: Enable autocast for BF16 Stable Cascade pipeline
                    do_autocast = True if self.version == "cascade" and self.bf16 else False
                    with torch.inference_mode(), torch.autocast("cuda", enabled=do_autocast):
                        export_onnx(self.get_model()) # 어쨌든 get_model은 있는 거니까 UNetSpatioTemporalModel은 보장이 되어 있는 듯?
            else:
                print(f"[I] Found cached ONNX model: {onnx_path}")

            print(f"[I] Optimizing ONNX model: {onnx_opt_path}")
            onnx_opt_graph = self.optimize(onnx.load(onnx_path))
            if onnx_graph_needs_external_data(onnx_opt_graph):
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_opt_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False)
            else:
                onnx.save(onnx_opt_graph, onnx_opt_path)
        else:
            print(f"[I] Found cached optimized ONNX model: {onnx_opt_path} ")

    # Helper utility for weights map
    def export_weights_map(self, onnx_opt_path, weights_map_path):
        if not os.path.exists(weights_map_path):
            onnx_opt_dir = os.path.dirname(onnx_opt_path)
            onnx_opt_model = onnx.load(onnx_opt_path)
            state_dict = self.get_model().state_dict()
            # Create initializer data hashes
            initializer_hash_mapping = {}
            for initializer in onnx_opt_model.graph.initializer:
                initializer_data = numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
                initializer_hash = hash(initializer_data.data.tobytes())
                initializer_hash_mapping[initializer.name] = (initializer_hash, initializer_data.shape)

            weights_name_mapping = {}
            weights_shape_mapping = {}
            # set to keep track of initializers already added to the name_mapping dict
            initializers_mapped = set()
            for wt_name, wt in state_dict.items():
                # get weight hash
                wt = wt.cpu().detach().numpy().astype(np.float16)
                wt_hash = hash(wt.data.tobytes())
                wt_t_hash = hash(np.transpose(wt).data.tobytes())

                for initializer_name, (initializer_hash, initializer_shape) in initializer_hash_mapping.items():
                    # Due to constant folding, some weights are transposed during export
                    # To account for the transpose op, we compare the initializer hash to the
                    # hash for the weight and its transpose
                    if wt_hash == initializer_hash or wt_t_hash == initializer_hash:
                        # The assert below ensures there is a 1:1 mapping between
                        # PyTorch and ONNX weight names. It can be removed in cases where 1:many
                        # mapping is found and name_mapping[wt_name] = list()
                        assert initializer_name not in initializers_mapped
                        weights_name_mapping[wt_name] = initializer_name
                        initializers_mapped.add(initializer_name)
                        is_transpose = False if wt_hash == initializer_hash else True
                        weights_shape_mapping[wt_name] = (initializer_shape, is_transpose)

                # Sanity check: Were any weights not matched
                if wt_name not in weights_name_mapping:
                    print(f'[I] PyTorch weight {wt_name} not matched with any ONNX initializer')
            print(f'[I] {len(weights_name_mapping.keys())} PyTorch weights were matched with ONNX initializers')
            assert weights_name_mapping.keys() == weights_shape_mapping.keys()
            with open(weights_map_path, 'w') as fp:
                json.dump([weights_name_mapping, weights_shape_mapping], fp)
        else:
            print(f"[I] Found cached weights map: {weights_map_path} ")

    def optimize(self, onnx_graph, return_onnx=True, **kwargs):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        if kwargs.get('modify_fp8_graph', False):
            is_fp16_io = kwargs.get('is_fp16_io', True)
            opt.modify_fp8_graph(is_fp16_io=is_fp16_io)
            opt.info(self.name + ': modify fp8 graph')
        if self.version.startswith("flux.1") and self.fp8:
            opt.flux_convert_rope_weight_type()
            opt.info(self.name + ': convert rope weight type for fp8 flux')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        if kwargs.get('fuse_mha_qkv_int8', False):
            opt.fuse_mha_qkv_int8_sq()
            opt.info(self.name + ': fuse QKV nodes')
        onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        latent_height = image_height // self.compression_factor
        latent_width = image_width // self.compression_factor
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // self.compression_factor
        latent_width = image_width // self.compression_factor
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)


class CLIPModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        embedding_dim,
        fp16=False,
        tf32=False,
        bf16=False,
        output_hidden_states=False,
        keep_pooled_output=False,
        subfolder="text_encoder",
    ):
        super(CLIPModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, tf32=tf32, bf16=bf16, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.subfolder = subfolder
        self.hidden_layer_offset = 0 if pipeline.is_cascade() else -1
        self.keep_pooled_output = keep_pooled_output

        # Output the final hidden state
        if output_hidden_states:
            self.extra_output_names = ['hidden_states']

    def get_model(self, torch_inference=''):
        model_opts = {'torch_dtype': torch.float16} if self.fp16 else {'torch_dtype': torch.bfloat16} if self.bf16 else {}
        clip_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not is_model_cached(clip_model_dir, model_opts, self.hf_safetensor, model_name='model'):
            model = CLIPTextModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(clip_model_dir, **model_opts)
        else:
            print(f"[I] Load CLIPTextModel model from: {clip_model_dir}")
            model = CLIPTextModel.from_pretrained(clip_model_dir, **model_opts).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
        output_names = ['text_embeddings']
        if self.keep_pooled_output:
            output_names += ['pooled_embeddings']
        return output_names

    def get_dynamic_axes(self):
        dynamic_axes =  {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'},
        }
        if self.keep_pooled_output:
            dynamic_axes['pooled_embeddings'] = {0: 'B'}
        return dynamic_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim),
        }
        if self.keep_pooled_output:
            output['pooled_embeddings'] = (batch_size, self.embedding_dim)
        if 'hidden_states' in self.extra_output_names:
            output['hidden_states'] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        keep_outputs = [0, 1] if self.keep_pooled_output else [0]
        opt.select_outputs(keep_outputs)
        opt.cleanup()
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs(keep_outputs, names=self.get_output_names()) # rename network outputs
        opt.info(self.name + ': rename network output(s)')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        if 'hidden_states' in self.extra_output_names:
            opt_onnx_graph = opt.clip_add_hidden_states(self.hidden_layer_offset, return_onnx=True)
            opt.info(self.name + ': added hidden_states')
        opt.info(self.name + ': finished')
        return opt_onnx_graph


class CLIPWithProjModel(CLIPModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        bf16=False,
        max_batch_size=16,
        output_hidden_states=False,
        subfolder="text_encoder_2",
    ):

        super(CLIPWithProjModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, bf16=bf16, max_batch_size=max_batch_size, embedding_dim=get_clipwithproj_embedding_dim(version, pipeline), output_hidden_states=output_hidden_states)
        self.subfolder = subfolder

    def get_model(self, torch_inference=''):
        model_opts = {'variant': 'bf16', 'torch_dtype': torch.bfloat16} if self.bf16 else {}
        clip_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not is_model_cached(clip_model_dir, model_opts, self.hf_safetensor, model_name='model'):
            model = CLIPTextModelWithProjection.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(clip_model_dir, **model_opts)
        else:
            print(f"[I] Load CLIPTextModelWithProjection model from: {clip_model_dir}")
            model = CLIPTextModelWithProjection.from_pretrained(clip_model_dir, **model_opts).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['input_ids', 'attention_mask']

    def get_output_names(self):
       return ['text_embeddings']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'attention_mask': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)],
            'attention_mask': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'attention_mask': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        return (
            torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device),
            torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)
        )

class CLIPVisionWithProjModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=1,
        subfolder="image_encoder",
    ):

        super(CLIPVisionWithProjModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size)
        self.subfolder = subfolder

    def get_model(self, torch_inference=''):
        clip_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder) # 이렇게 clip_model_dir를 지정하는 형태가 아니라 그냥 아예 config에서 받는 형태로 변경하는 게 나을 듯함
        if not os.path.exists(clip_model_dir):
            model = CLIPVisionModelWithProjection.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIPVisionModelWithProjection model from: {clip_model_dir}")
            model = CLIPVisionModelWithProjection.from_pretrained(clip_model_dir).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model


class CLIPImageProcessorModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=1,
        subfolder="feature_extractor",
    ):

        super(CLIPImageProcessorModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size)
        self.subfolder = subfolder

    def get_model(self, torch_inference=''):
        clip_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        # NOTE to(device) not supported
        if not os.path.exists(clip_model_dir):
            model = CLIPImageProcessor.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIPImageProcessor model from: {clip_model_dir}")
            model = CLIPImageProcessor.from_pretrained(clip_model_dir)
        model = optimize_checkpoint(model, torch_inference)
        return model


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets) -> None:
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets

    def forward(self, sample, timestep, encoder_hidden_states, images, controlnet_scales):
        for i, (image, conditioning_scale, controlnet) in enumerate(zip(images, controlnet_scales, self.controlnets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                return_dict=False,
            )

            down_samples = [
                    down_sample * conditioning_scale
                    for down_sample in down_samples
                ]
            mid_sample *= conditioning_scale

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )
        return noise_pred


class UNetModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16 = False,
        int8 = False,
        fp8 = False,
        max_batch_size = 16,
        text_maxlen = 77,
        controlnets = None,
        do_classifier_free_guidance = False,
    ):

        super(UNetModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, int8=int8, fp8=fp8, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=get_unet_embedding_dim(version, pipeline))
        self.subfolder = 'unet'
        self.controlnets = get_path(version, pipeline, controlnets) if controlnets else None
        self.unet_dim = (9 if pipeline.is_inpaint() else 4)
        self.xB = 2 if do_classifier_free_guidance else 1 # batch multiplier

    def get_model(self, torch_inference=''):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.controlnets:
            unet_model = UNet2DConditionModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            cnet_model_opts = {'torch_dtype': torch.float16} if self.fp16 else {}
            controlnets = torch.nn.ModuleList([ControlNetModel.from_pretrained(path, **cnet_model_opts).to(self.device) for path in self.controlnets])
            # FIXME - cache UNet2DConditionControlNetModel
            model = UNet2DConditionControlNetModel(unet_model, controlnets)
        else:
            unet_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
            if not is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
                model = UNet2DConditionModel.from_pretrained(self.path,
                    subfolder=self.subfolder,
                    use_safetensors=self.hf_safetensor,
                    token=self.hf_token,
                    **model_opts).to(self.device)
                model.save_pretrained(unet_model_dir, **model_opts)
            else:
                print(f"[I] Load UNet2DConditionModel  model from: {unet_model_dir}")
                model = UNet2DConditionModel.from_pretrained(unet_model_dir, **model_opts).to(self.device)
            if torch_inference:
                model.to(memory_format=torch.channels_last)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        if self.controlnets is None:
            return ['sample', 'timestep', 'encoder_hidden_states']
        else:
            return ['sample', 'timestep', 'encoder_hidden_states', 'images', 'controlnet_scales']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        xB = '2B' if self.xB == 2 else 'B'
        if self.controlnets is None:
            return {
                'sample': {0: xB, 2: 'H', 3: 'W'},
                'encoder_hidden_states': {0: xB},
                'latent': {0: xB, 2: 'H', 3: 'W'}
            }
        else:
            return {
                'sample': {0: xB, 2: 'H', 3: 'W'},
                'encoder_hidden_states': {0: xB},
                'images': {1: xB, 3: '8H', 4: '8W'},
                'latent': {0: xB, 2: 'H', 3: 'W'}
            }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        if self.controlnets is None:
            return {
                'sample': [(self.xB*min_batch, self.unet_dim, min_latent_height, min_latent_width), (self.xB*batch_size, self.unet_dim, latent_height, latent_width), (self.xB*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
                'encoder_hidden_states': [(self.xB*min_batch, self.text_maxlen, self.embedding_dim), (self.xB*batch_size, self.text_maxlen, self.embedding_dim), (self.xB*max_batch, self.text_maxlen, self.embedding_dim)]
            }
        else:
            return {
                'sample': [(self.xB*min_batch, self.unet_dim, min_latent_height, min_latent_width),
                           (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
                           (self.xB*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
                'encoder_hidden_states': [(self.xB*min_batch, self.text_maxlen, self.embedding_dim),
                                          (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
                                          (self.xB*max_batch, self.text_maxlen, self.embedding_dim)],
                'images': [(len(self.controlnets), self.xB*min_batch, 3, min_image_height, min_image_width),
                          (len(self.controlnets), self.xB*batch_size, 3, image_height, image_width),
                          (len(self.controlnets), self.xB*max_batch, 3, max_image_height, max_image_width)]
            }


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        if self.controlnets is None:
            return {
                'sample': (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
                'latent': (self.xB*batch_size, 4, latent_height, latent_width)
            }
        else:
            return {
                'sample': (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
                'images': (len(self.controlnets), self.xB*batch_size, 3, image_height, image_width),
                'latent': (self.xB*batch_size, 4, latent_height, latent_width)
                }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        if self.controlnets is None:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device),
                torch.tensor([1.], dtype=dtype, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
            )
        else:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device),
                torch.tensor(999, dtype=dtype, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
                torch.randn(len(self.controlnets), batch_size, 3, image_height, image_width, dtype=dtype, device=self.device),
                torch.randn(len(self.controlnets), dtype=dtype, device=self.device)
            )

    def optimize(self, onnx_graph):
        if self.fp8:
            return super().optimize(onnx_graph, modify_fp8_graph=True)
        if self.int8:
            return super().optimize(onnx_graph, fuse_mha_qkv_int8=True)
        return super().optimize(onnx_graph)


class UNetXLModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16 = False,
        int8 = False,
        fp8 = False,
        max_batch_size = 16,
        text_maxlen = 77,
        do_classifier_free_guidance = False,
    ):
        super(UNetXLModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, int8=int8, fp8=fp8, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=get_unet_embedding_dim(version, pipeline))
        self.subfolder = 'unet'
        self.unet_dim = (9 if pipeline.is_inpaint() else 4)
        self.time_dim = (5 if pipeline.is_sd_xl_refiner() else 6)
        self.xB = 2 if do_classifier_free_guidance else 1 # batch multiplier

    def get_model(self, torch_inference=''):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        unet_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
            model = UNet2DConditionModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            # Use default attention processor for ONNX export
            if not torch_inference:
                model.set_default_attn_processor()
            model.save_pretrained(unet_model_dir, **model_opts)
        else:
            print(f"[I] Load UNet2DConditionModel model from: {unet_model_dir}")
            model = UNet2DConditionModel.from_pretrained(unet_model_dir, **model_opts).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states', 'text_embeds', 'time_ids']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        xB = '2B' if self.xB == 2 else 'B'
        return {
            'sample': {0: xB, 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: xB},
            'latent': {0: xB, 2: 'H', 3: 'W'},
            'text_embeds': {0: xB},
            'time_ids': {0: xB}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(self.xB*min_batch, self.unet_dim, min_latent_height, min_latent_width), (self.xB*batch_size, self.unet_dim, latent_height, latent_width), (self.xB*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(self.xB*min_batch, self.text_maxlen, self.embedding_dim), (self.xB*batch_size, self.text_maxlen, self.embedding_dim), (self.xB*max_batch, self.text_maxlen, self.embedding_dim)],
            'text_embeds': [(self.xB*min_batch, 1280), (self.xB*batch_size, 1280), (self.xB*max_batch, 1280)],
            'time_ids': [(self.xB*min_batch, self.time_dim), (self.xB*batch_size, self.time_dim), (self.xB*max_batch, self.time_dim)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (self.xB*batch_size, 4, latent_height, latent_width),
            'text_embeds': (self.xB*batch_size, 1280),
            'time_ids': (self.xB*batch_size, self.time_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        # WAR to enable inference for H/W that are not multiples of 16
        # If building with Dynamic Shapes: ensure image height and width are not multiples of 16 for ONNX export and TensorRT engine build
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(self.xB*batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device),
            torch.tensor([1.], dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            {
                'added_cond_kwargs': {
                    'text_embeds': torch.randn(self.xB*batch_size, 1280, dtype=dtype, device=self.device),
                    'time_ids' : torch.randn(self.xB*batch_size, self.time_dim, dtype=dtype, device=self.device)
                }
            }
        )

    def optimize(self, onnx_graph):
        if self.fp8:
            return super().optimize(onnx_graph, modify_fp8_graph=True)
        if self.int8:
            return super().optimize(onnx_graph, fuse_mha_qkv_int8=True)
        return super().optimize(onnx_graph)


class UNetTemporalModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16 = False,
        fp8 = False,
        max_batch_size = 16,
        num_frames = 14,
        do_classifier_free_guidance = True,
    ):
        super(UNetTemporalModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, fp8=fp8, max_batch_size=max_batch_size, embedding_dim=get_unet_embedding_dim(version, pipeline))
        self.subfolder = 'unet'
        self.unet_dim = 4
        self.num_frames = num_frames
        self.out_channels = 4
        self.cross_attention_dim = 1024
        self.xB = 3 if do_classifier_free_guidance else 1 # batch multiplier

    def get_model(self, torch_inference=''):
        model_opts = {'torch_dtype': torch.float16} if self.fp16 else {}
        unet_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
            model = UNetSpatioTemporalConditionModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(unet_model_dir, **model_opts)
        else:
            print(f"[I] Load UNetSpatioTemporalConditionModel model from: {unet_model_dir}")
            model = UNetSpatioTemporalConditionModel.from_pretrained(unet_model_dir, **model_opts).to(self.device)
        
        # [TODO] 하드코딩 걷어내기
        add_ip_adapters(model, [32], [1.0])
        model.load_state_dict(
            torch.load("checkpoints/Sonic/unet.pth", map_location="cpu"),
            strict=True,
        )
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states', 'added_time_ids', 'ip_adapter_masks']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        xB = str(self.xB)+'B'
        return {
            'sample': {0: xB, 1: 'num_frames', 3: 'H', 4: 'W'},
            'encoder_hidden_states': {0: xB,},
            'added_time_ids': {0: xB},
            # 'ip_adapter_masks': {0: xB},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
        self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(self.xB*min_batch, self.num_frames, 2*self.out_channels, min_latent_height, min_latent_width), (self.xB*batch_size, self.num_frames, 2*self.out_channels, latent_height, latent_width), (self.xB*max_batch, self.num_frames, 2*self.out_channels, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(self.xB*min_batch, self.num_frames, 33, self.cross_attention_dim), (self.xB*batch_size, self.num_frames, 33, self.cross_attention_dim), (self.xB*max_batch, self.num_frames, 33, self.cross_attention_dim)], # 하드 코딩 (1 + 32)
            'added_time_ids': [(self.xB*min_batch, 3), (self.xB*batch_size, 3), (self.xB*max_batch, 3)],
            'ip_adapter_masks': [(min_batch, 1, image_height, image_width), (batch_size, 1, image_height, image_width), (max_batch, 1, image_height, image_width)],
        }


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (self.xB*batch_size, self.num_frames, 2*self.out_channels, latent_height, latent_width),
            'timestep': (1,),
            'encoder_hidden_states': (self.xB*batch_size, self.num_frames, 33, self.cross_attention_dim),
            'added_time_ids': (self.xB*batch_size, 3),
            'ip_adapter_masks': (batch_size, 1, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        # TODO chunk_size if forward_chunking is used
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)

        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(self.xB*batch_size, self.num_frames, 2*self.out_channels, latent_height, latent_width, dtype=dtype, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(self.xB*batch_size, self.num_frames, 33, self.cross_attention_dim, dtype=dtype, device=self.device), # hard-coded
            torch.randn(self.xB*batch_size, 3, dtype=dtype, device=self.device),
            torch.randn(batch_size, 1, image_height, image_width, dtype=dtype, device=self.device),
        )

    def optimize(self, onnx_graph):
        return super().optimize(onnx_graph, modify_fp8_graph=self.fp8)


class UNetCascadeModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16 = False,
        bf16 = False,
        max_batch_size = 16,
        text_maxlen = 77,
        do_classifier_free_guidance = False,
        compression_factor=42,
        latent_dim_scale=10.67,
        image_embedding_dim=768,
        lite=False
    ):
        super(UNetCascadeModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, bf16=bf16, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=get_unet_embedding_dim(version, pipeline), compression_factor=compression_factor)
        self.is_prior = True if pipeline.is_cascade_prior() else False
        self.subfolder = 'prior' if self.is_prior else 'decoder'
        if lite:
            self.subfolder += '_lite'
        self.prior_dim = 16
        self.decoder_dim = 4
        self.xB = 2 if do_classifier_free_guidance else 1 # batch multiplier
        self.latent_dim_scale = latent_dim_scale
        self.min_latent_shape = self.min_image_shape // self.compression_factor
        self.max_latent_shape = self.max_image_shape // self.compression_factor
        self.do_constant_folding = False
        self.image_embedding_dim = image_embedding_dim

    def get_model(self, torch_inference=''):
        # FP16 variant doesn't exist
        model_opts = {'torch_dtype': torch.float16} if self.fp16 else {}
        model_opts = {'variant': 'bf16', 'torch_dtype': torch.bfloat16} if self.bf16 else model_opts
        unet_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not is_model_cached(unet_model_dir, model_opts, self.hf_safetensor):
            model = StableCascadeUNet.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(unet_model_dir, **model_opts)
        else:
            print(f"[I] Load Stable Cascade UNet pytorch model from: {unet_model_dir}")
            model = StableCascadeUNet.from_pretrained(unet_model_dir, **model_opts).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        if self.is_prior:
            return ['sample', 'timestep_ratio', 'clip_text_pooled', 'clip_text', 'clip_img']
        else:
            return ['sample', 'timestep_ratio', 'clip_text_pooled', 'effnet']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        xB = '2B' if self.xB == 2 else 'B'
        if self.is_prior:
            return {
                'sample': {0: xB, 2: 'H', 3: 'W'},
                'timestep_ratio': {0: xB},
                'clip_text_pooled': {0: xB},
                'clip_text': {0: xB},
                'clip_img': {0: xB},
                'latent': {0: xB, 2: 'H', 3: 'W'}
            }
        else:
            return {
                'sample': {0: xB, 2: 'H', 3: 'W'},
                'timestep_ratio': {0: xB},
                'clip_text_pooled': {0: xB},
                'effnet': {0: xB, 2: 'H_effnet', 3: 'W_effnet'},
                'latent': {0: xB, 2: 'H', 3: 'W'}
            }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        if self.is_prior:
            return {
                'sample': [(self.xB*min_batch, self.prior_dim, min_latent_height, min_latent_width), (self.xB*batch_size, self.prior_dim, latent_height, latent_width), (self.xB*max_batch, self.prior_dim, max_latent_height, max_latent_width)],
                'timestep_ratio': [(self.xB*min_batch,), (self.xB*batch_size,), (self.xB*max_batch,)],
                'clip_text_pooled': [(self.xB*min_batch, 1, self.embedding_dim), (self.xB*batch_size, 1, self.embedding_dim), (self.xB*max_batch, 1, self.embedding_dim)],
                'clip_text': [(self.xB*min_batch, self.text_maxlen, self.embedding_dim), (self.xB*batch_size, self.text_maxlen, self.embedding_dim), (self.xB*max_batch, self.text_maxlen, self.embedding_dim)],
                'clip_img': [(self.xB*min_batch, 1, self.image_embedding_dim), (self.xB*batch_size, 1, self.image_embedding_dim), (self.xB*max_batch, 1, self.image_embedding_dim)],
            }
        else:
            return {
                'sample': [(self.xB*min_batch, self.decoder_dim, int(min_latent_height * self.latent_dim_scale), int(min_latent_width * self.latent_dim_scale)),
                    (self.xB*batch_size, self.decoder_dim, int(latent_height * self.latent_dim_scale), int(latent_width * self.latent_dim_scale)),
                    (self.xB*max_batch, self.decoder_dim, int(max_latent_height * self.latent_dim_scale), int(max_latent_width * self.latent_dim_scale))],
                'timestep_ratio': [(self.xB*min_batch,), (self.xB*batch_size,), (self.xB*max_batch,)],
                'clip_text_pooled': [(self.xB*min_batch, 1, self.embedding_dim), (self.xB*batch_size, 1, self.embedding_dim), (self.xB*max_batch, 1, self.embedding_dim)],
                'effnet': [(self.xB*min_batch, self.prior_dim, min_latent_height, min_latent_width), (self.xB*batch_size, self.prior_dim, latent_height, latent_width), (self.xB*max_batch, self.prior_dim, max_latent_height, max_latent_width)]
            }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        if self.is_prior:
            return {
                'sample': (self.xB*batch_size, self.prior_dim, latent_height, latent_width),
                'timestep_ratio': (self.xB*batch_size,),
                'clip_text_pooled': (self.xB*batch_size, 1, self.embedding_dim),
                'clip_text': (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
                'clip_img': (self.xB*batch_size, 1, self.image_embedding_dim),
                'latent': (self.xB*batch_size, self.prior_dim, latent_height, latent_width)
            }
        else:
            return {
                'sample': (self.xB*batch_size, self.decoder_dim, int(latent_height * self.latent_dim_scale), int(latent_width * self.latent_dim_scale)),
                'timestep_ratio': (self.xB*batch_size,),
                'clip_text_pooled': (self.xB*batch_size, 1, self.embedding_dim),
                'effnet': (self.xB*batch_size, self.prior_dim, latent_height, latent_width),
                'latent': (self.xB*batch_size, self.decoder_dim, int(latent_height * self.latent_dim_scale), int(latent_width * self.latent_dim_scale))
            }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        if self.is_prior:
            return (
                torch.randn(batch_size, self.prior_dim, latent_height, latent_width, dtype=dtype, device=self.device),
                torch.tensor([1.]*batch_size, dtype=dtype, device=self.device),
                torch.randn(batch_size, 1, self.embedding_dim, dtype=dtype, device=self.device),
                {
                    'clip_text': torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
                    'clip_img': torch.randn(batch_size, 1, self.image_embedding_dim, dtype=dtype, device=self.device),
                }
            )
        else:
            return (
                torch.randn(batch_size, self.decoder_dim, int(latent_height * self.latent_dim_scale), int(latent_width * self.latent_dim_scale), dtype=dtype, device=self.device),
                torch.tensor([1.]*batch_size, dtype=dtype, device=self.device),
                torch.randn(batch_size, 1, self.embedding_dim, dtype=dtype, device=self.device),
                {
                    'effnet': torch.randn(batch_size, self.prior_dim, latent_height, latent_width, dtype=dtype, device=self.device),
                }
            )

class FluxTransformerModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16 = False,
        tf32=False,
        int8 = False,
        fp8 = False,
        bf16 = False,
        max_batch_size = 16,
        text_maxlen = 77,
        build_strongly_typed=False,
        weight_streaming=False,
        weight_streaming_budget_percentage=None,
    ):
        super(FluxTransformerModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, tf32=tf32, int8=int8, fp8=fp8, bf16=bf16, max_batch_size=max_batch_size, text_maxlen=text_maxlen)
        self.subfolder = 'transformer'
        self.transformer_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not os.path.exists(self.transformer_model_dir):
            self.config = FluxTransformer2DModel.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load FluxTransformer2DModel config from: {self.transformer_model_dir}")
            self.config = FluxTransformer2DModel.load_config(self.transformer_model_dir)
        self.build_strongly_typed = build_strongly_typed
        self.weight_streaming = weight_streaming
        self.weight_streaming_budget_percentage = weight_streaming_budget_percentage
        self.out_channels = self.config.get('out_channels') or self.config['in_channels']

    def get_model(self, torch_inference=''):
        model_opts = {'torch_dtype': torch.float16} if self.fp16 else {'torch_dtype': torch.bfloat16} if self.bf16 else {}
        if not is_model_cached(self.transformer_model_dir, model_opts, self.hf_safetensor):
            model = FluxTransformer2DModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(self.transformer_model_dir, **model_opts)
        else:
            print(f"[I] Load FluxTransformer2DModel model from: {self.transformer_model_dir}")
            model = FluxTransformer2DModel.from_pretrained(self.transformer_model_dir, **model_opts).to(self.device)
        if torch_inference:
            model.to(memory_format=torch.channels_last)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['hidden_states', 'encoder_hidden_states', 'pooled_projections', 'timestep', 'img_ids', 'txt_ids', 'guidance']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        dynamic_axes = {
            'hidden_states': {0: 'B', 1: 'latent_dim'},
            'encoder_hidden_states': {0: 'B'},
            'pooled_projections': {0: 'B'},
            'timestep': {0: 'B'},
            'img_ids': {0: 'latent_dim'},
        }
        if self.config['guidance_embeds']:
            dynamic_axes['guidance'] = {0: 'B'}
        return dynamic_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        input_profile = {
            'hidden_states': [(min_batch, (min_latent_height // 2) * (min_latent_width // 2), self.config['in_channels']), (batch_size, (latent_height // 2) * (latent_width // 2), self.config['in_channels']), (max_batch, (max_latent_height // 2) * (max_latent_width // 2), self.config['in_channels'])],
            'encoder_hidden_states': [(min_batch, self.text_maxlen, self.config['joint_attention_dim']), (batch_size, self.text_maxlen, self.config['joint_attention_dim']), (max_batch, self.text_maxlen, self.config['joint_attention_dim'])],
            'pooled_projections': [(min_batch, self.config['pooled_projection_dim']), (batch_size, self.config['pooled_projection_dim']), (max_batch, self.config['pooled_projection_dim'])],
            'timestep': [(min_batch,), (batch_size,), (max_batch,)],
            'img_ids': [((min_latent_height // 2) * (min_latent_width // 2), 3), ((latent_height // 2) * (latent_width // 2), 3), ((max_latent_height // 2) * (max_latent_width // 2), 3)],
            'txt_ids': [(self.text_maxlen, 3), (self.text_maxlen, 3), (self.text_maxlen, 3)],
        }
        if self.config['guidance_embeds']:
            input_profile['guidance'] = [(min_batch,), (batch_size,), (max_batch,)]
        return input_profile


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            'hidden_states': (batch_size, (latent_height // 2) * (latent_width // 2), self.config['in_channels']),
            'encoder_hidden_states': (batch_size, self.text_maxlen, self.config['joint_attention_dim']),
            'pooled_projections': (batch_size, self.config['pooled_projection_dim']),
            'timestep': (batch_size,),
            'img_ids': ((latent_height // 2) * (latent_width // 2), 3),
            'txt_ids': (self.text_maxlen, 3),
            'latent': (batch_size, (latent_height // 2) * (latent_width // 2), self.out_channels),
        }
        if self.config['guidance_embeds']:
            shape_dict['guidance'] = (batch_size,)
        return shape_dict


    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float32
        assert not (self.fp16 and self.bf16), "fp16 and bf16 cannot be enabled simultaneously"
        tensor_dtype = torch.bfloat16 if self.bf16 else (torch.float16 if self.fp16 else torch.float32)

        sample_input = (
            torch.randn(batch_size, (latent_height // 2) * (latent_width // 2), self.config['in_channels'], dtype=tensor_dtype, device=self.device),
            torch.randn(batch_size, self.text_maxlen, self.config['joint_attention_dim'], dtype=tensor_dtype, device=self.device),
            torch.randn(batch_size, self.config['pooled_projection_dim'], dtype=tensor_dtype, device=self.device),
            torch.tensor([1.]*batch_size, dtype=tensor_dtype, device=self.device),
            torch.randn((latent_height // 2) * (latent_width // 2), 3, dtype=dtype, device=self.device),
            torch.randn(self.text_maxlen, 3, dtype=dtype, device=self.device),
            { }
        )
        if self.config['guidance_embeds']:
            sample_input[-1]['guidance'] = torch.tensor([1.]*batch_size, dtype=dtype, device=self.device)
        return sample_input

    def optimize(self, onnx_graph):
        if self.fp8:
            return super().optimize(onnx_graph)
        if self.int8:
            return super().optimize(onnx_graph, fuse_mha_qkv_int8=True)
        return super().optimize(onnx_graph)


class VAEModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch_size=16,
    ):
        super(VAEModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, tf32=tf32, bf16=bf16, max_batch_size=max_batch_size)
        self.subfolder = 'vae'
        self.vae_decoder_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not os.path.exists(self.vae_decoder_model_dir):
            self.config = AutoencoderKL.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load AutoencoderKL (decoder) config from: {self.vae_decoder_model_dir}")
            self.config = AutoencoderKL.load_config(self.vae_decoder_model_dir)

    def get_model(self, torch_inference=''):
        model_opts = {'torch_dtype': torch.float16} if self.fp16 else {'torch_dtype': torch.bfloat16} if self.bf16 else {}
        if not is_model_cached(self.vae_decoder_model_dir, model_opts, self.hf_safetensor):
            model = AutoencoderKL.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(self.vae_decoder_model_dir, **model_opts)
        else:
            print(f"[I] Load AutoencoderKL (decoder) model from: {self.vae_decoder_model_dir}")
            model = AutoencoderKL.from_pretrained(self.vae_decoder_model_dir, **model_opts).to(self.device)
        model.forward = model.decode
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, self.config['latent_channels'], min_latent_height, min_latent_width),
                (batch_size, self.config['latent_channels'], latent_height, latent_width),
                (max_batch, self.config['latent_channels'], max_latent_height, max_latent_width)
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, self.config['latent_channels'], latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        return torch.randn(batch_size, self.config['latent_channels'], latent_height, latent_width, dtype=dtype, device=self.device)

class SD3_VAEDecoderModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        fp16=False,
    ):
        super(SD3_VAEDecoderModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, max_batch_size=max_batch_size)
        self.subfolder = 'sd3'

    def get_model(self, torch_inference=''):
        dtype = torch.float16 if self.fp16 else torch.float32
        sd3_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        sd3_filename="sd3_medium.safetensors"
        sd3_model_path = f"{sd3_model_dir}/{sd3_filename}"
        if not os.path.exists(sd3_model_path):
            hf_hub_download(repo_id=self.path, filename=sd3_filename, local_dir=sd3_model_dir)
        with safe_open(sd3_model_path, framework="pt", device=self.device) as f:
            model = SDVAE(device=self.device, dtype=dtype).eval().cuda()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, model, prefix, self.device, dtype)
        model.forward = model.decode
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 16, min_latent_height, min_latent_width), (batch_size, 16, latent_height, latent_width), (max_batch, 16, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 16, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return torch.randn(batch_size, 16, latent_height, latent_width, dtype=dtype, device=self.device)

class VAEDecTemporalModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size = 16,
        decode_chunk_size = 14,
    ):
        super(VAEDecTemporalModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size)
        self.subfolder = 'vae'
        self.decode_chunk_size = decode_chunk_size

    def get_model(self, torch_inference=''):
        vae_decoder_model_path = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not os.path.exists(vae_decoder_model_path):
            model = AutoencoderKLTemporalDecoder.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token).to(self.device)
            model.save_pretrained(vae_decoder_model_path)
        else:
            print(f"[I] Load AutoencoderKLTemporalDecoder model from: {vae_decoder_model_path}")
            model = AutoencoderKLTemporalDecoder.from_pretrained(vae_decoder_model_path).to(self.device)
        model.forward = model.decode
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['latent', 'num_frames_in']

    def get_output_names(self):
       return ['frames']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'num_frames_in', 2: 'H', 3: 'W'},
            'frames': {0: 'num_frames_in', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        assert batch_size == 1
        _, _, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(1, 4, min_latent_height, min_latent_width), (self.decode_chunk_size, 4, latent_height, latent_width), (self.decode_chunk_size, 4, max_latent_height, max_latent_width)],
            'num_frames_in': [(1,), (1,), (1,)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        assert batch_size == 1
        return {
            'latent': (self.decode_chunk_size, 4, latent_height, latent_width),
            #'num_frames_in': (1,),
            'frames': (self.decode_chunk_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        assert batch_size == 1
        return (
            torch.randn(self.decode_chunk_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device),
            self.decode_chunk_size,
        )


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, version, pipeline, hf_token, device, path, framework_model_dir, subfolder, fp16=False, bf16=False, hf_safetensor=False):
        super().__init__()
        model_opts = {'torch_dtype': torch.float16} if fp16 else {'torch_dtype': torch.bfloat16} if bf16 else {}
        vae_encoder_model_dir = get_checkpoint_dir(framework_model_dir, version, pipeline, subfolder)
        if not is_model_cached(vae_encoder_model_dir, model_opts, hf_safetensor):
            self.vae_encoder = AutoencoderKL.from_pretrained(path,
                subfolder='vae',
                use_safetensors=hf_safetensor,
                token=hf_token,
                **model_opts).to(device)
            self.vae_encoder.save_pretrained(vae_encoder_model_dir, **model_opts)
        else:
            print(f"[I] Load AutoencoderKL (encoder) model from: {vae_encoder_model_dir}")
            self.vae_encoder = AutoencoderKL.from_pretrained(vae_encoder_model_dir, **model_opts).to(device)

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoderModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch_size=16,
    ):
        super(VAEEncoderModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, tf32=tf32, bf16=bf16, max_batch_size=max_batch_size)
        self.subfolder = 'vae'
        self.vae_encoder_model_dir = get_checkpoint_dir(framework_model_dir, version, self.pipeline, self.subfolder)
        if not os.path.exists(self.vae_encoder_model_dir):
            self.config = AutoencoderKL.load_config(self.path, subfolder=self.subfolder, token=self.hf_token)
        else:
            print(f"[I] Load AutoencoderKL (encoder) config from: {self.vae_encoder_model_dir}")
            self.config = AutoencoderKL.load_config(self.vae_encoder_model_dir)

    def get_model(self, torch_inference=''):
        vae_encoder = TorchVAEEncoder(self.version, self.pipeline, self.hf_token, self.device, self.path, self.framework_model_dir, self.subfolder, self.fp16, self.bf16, hf_safetensor=self.hf_safetensor)
        return vae_encoder

    def get_input_names(self):
        return ['images']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'images': [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, self.config['latent_channels'], latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        return torch.randn(batch_size, 3, image_height, image_width, dtype=dtype, device=self.device)

class SD3_VAEEncoderModel(VAEEncoderModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        fp16=False,
    ):
        super(SD3_VAEEncoderModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, max_batch_size=max_batch_size)
        self.subfolder = 'sd3'

    def get_model(self, torch_inference=''):
        dtype = torch.float16 if self.fp16 else torch.float32
        sd3_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        sd3_filename="sd3_medium.safetensors"
        sd3_model_path = f"{sd3_model_dir}/{sd3_filename}"
        if not os.path.exists(sd3_model_path):
            hf_hub_download(repo_id=self.path, filename=sd3_filename, local_dir=sd3_model_dir)
        with safe_open(sd3_model_path, framework="pt", device=self.device) as f:
            model = SDVAE(device=self.device, dtype=dtype).eval().cuda()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, model, prefix, self.device, dtype)
        model.forward = model.encode
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, _, _, _, _, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'images': [(min_batch, 3, image_height, image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, image_height, image_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 16, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        dtype = torch.float16 if self.fp16 else torch.float32
        return torch.randn(batch_size, 3, image_height, image_width, dtype=dtype, device=self.device)

class VQGANModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16=False,
        bf16=False,
        max_batch_size=16,
        compression_factor=42,
        latent_dim_scale=10.67,
        scale_factor=0.3764
    ):
        super(VQGANModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, bf16=bf16, max_batch_size=max_batch_size, compression_factor=compression_factor)
        self.subfolder = 'vqgan'
        self.latent_dim_scale = latent_dim_scale
        self.scale_factor = scale_factor

    def get_model(self, torch_inference=''):
        model_opts = {'variant': 'bf16', 'torch_dtype': torch.bfloat16} if self.bf16 else {}
        vqgan_model_dir = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder)
        if not is_model_cached(vqgan_model_dir, model_opts, self.hf_safetensor, model_name='model'):
            model = PaellaVQModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(vqgan_model_dir, **model_opts)
        else:
            print(f"[I] Load VQGAN pytorch model from: {vqgan_model_dir}")
            model = PaellaVQModel.from_pretrained(vqgan_model_dir, **model_opts).to(self.device)
        model.forward = model.decode
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }
    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.bfloat16 if self.bf16 else torch.float32
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=dtype, device=self.device)

    def check_dims(self, batch_size, image_height, image_width):
        latent_height, latent_width = super().check_dims(batch_size, image_height, image_width)
        latent_height = int(latent_height * self.latent_dim_scale)
        latent_width = int(latent_width * self.latent_dim_scale)
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            super().get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        min_latent_height = int(min_latent_height * self.latent_dim_scale)
        min_latent_width = int(min_latent_width * self.latent_dim_scale)
        max_latent_height = int(max_latent_height * self.latent_dim_scale)
        max_latent_width = int(max_latent_width * self.latent_dim_scale)
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)

def make_tokenizer(version, pipeline, hf_token, framework_model_dir, subfolder="tokenizer", tokenizer_type="clip"):
    if tokenizer_type == "clip":
        tokenizer_class = CLIPTokenizer
    elif tokenizer_type == "t5":
        tokenizer_class = T5TokenizerFast
    else:
        raise ValueError(f"Unsupported tokenizer_type {tokenizer_type}. Only tokenizer_type clip and t5 are currently supported")
    tokenizer_model_dir = get_checkpoint_dir(framework_model_dir, version, pipeline.name, subfolder)
    if not os.path.exists(tokenizer_model_dir):
        model = tokenizer_class.from_pretrained(get_path(version, pipeline),
                subfolder=subfolder,
                use_safetensors=pipeline.is_sd_xl(),
                token=hf_token)
        model.save_pretrained(tokenizer_model_dir)
    else:
        print(f"[I] Load {tokenizer_class.__name__} model from: {tokenizer_model_dir}")
        model = tokenizer_class.from_pretrained(tokenizer_model_dir)
    return model

def make_scheduler(cls, version, pipeline, hf_token, framework_model_dir, subfolder="scheduler"):
    scheduler_dir = os.path.join(framework_model_dir, version, pipeline.name, next(iter({cls.__name__})).lower(), subfolder)
    if not os.path.exists(scheduler_dir):
        scheduler = cls.from_pretrained(get_path(version, pipeline), subfolder=subfolder, token=hf_token)
        scheduler.save_pretrained(scheduler_dir)
    else:
        print(f"[I] Load Scheduler {cls.__name__} from: {scheduler_dir}")
        scheduler = cls.from_pretrained(scheduler_dir)
    return scheduler

@dataclass
class UNetSpatioTemporalConditionOutput(diffusers.utils.BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None

class CustomUNetSpatioTemporalConditionModel(UNetSpatioTemporalConditionModel):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        spatial_condition: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch*num_frames, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            spatial_condition (`torch.Tensor`, *optional*, defaults to `None`):
                The spatial_condition embedding with shape `(batch, num_frames, channel_in(320), height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        # cast to device
        batch_size, num_frames = batch_size.to(self.device), num_frames.to(self.device)
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        # import ipdb
        # ipdb.set_trace()
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames.to(self.device), dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        
        ### 20240731 process encoder_hidden_states ###
        if isinstance(encoder_hidden_states, tuple):
            # ip_hidden_states is a list
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            if encoder_hidden_states.shape[0]==batch_size:
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
            encoder_hidden_states = (encoder_hidden_states, ip_hidden_states)
        elif encoder_hidden_states.shape[0]==batch_size:
            ### if framewised feature is not provided, repeat_interleave
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
            

        # 2. pre-process
        sample = self.conv_in(sample)
        
        ### 20240731 add spatial_condition here ###
        if spatial_condition is not None:
            sample = sample + spatial_condition.flatten(0,1)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)
    

def add_ip_adapters(unet, num_adapter_embeds=[32,], scale=[1.0,]):
    
    assert len(num_adapter_embeds)==len(scale)
    
    
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        # if cross_attention_dim is None or "temporal_transformer_blocks" in name:
        if cross_attention_dim is None:
            attn_processor_class = (
                    AttnProcessor2_0 if hasattr(torch.nn.functional, "scaled_dot_product_attention") else AttnProcessor
                )
            attn_procs[name] = attn_processor_class()
        else:
            attn_processor_class = (
                    IPAdapterAttnProcessor2_0 if hasattr(torch.nn.functional, "scaled_dot_product_attention") else IPAdapterAttnProcessor
                )
            
            attn_procs[name] = attn_processor_class(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=num_adapter_embeds,
                        scale=scale
                    ).to(device=unet.device, dtype=unet.dtype)

            layer_name = name.split(".processor")[0]
            weights = {}
         
            for i in range(len(num_adapter_embeds)):
                weights.update({f"to_k_ip.{i}.weight": unet_sd[layer_name + ".to_k.weight"]})
                weights.update({f"to_v_ip.{i}.weight": unet_sd[layer_name + ".to_v.weight"]})
                    
    
            attn_procs[name].load_state_dict(weights)

    unet.set_attn_processor(attn_procs)

    adapter_modules = torch.nn.ModuleList([m for m in unet.attn_processors.values() if isinstance(m, IPAdapterAttnProcessor) or isinstance(m, IPAdapterAttnProcessor2_0)])
    return adapter_modules
    
