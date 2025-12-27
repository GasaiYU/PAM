# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import shutil
import sys
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Union, Optional

import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

import decord  # isort:skip
decord.bridge.set_bridge("torch")

from args import get_args  # isort:skip
from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop, VideoDatasetWithResizingTracking  # isort:skip
from dataset import HOIVideoDatasetResizing
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import get_gradient_norm, get_optimizer, prepare_rotary_positional_embeddings, print_memory, reset_memory  # isort:skip

from diffusers.utils import load_image
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking
from models.cogvidex_combination import CogVideoXImageToVideoPipelineCombination
from models.cogvideox_tracking import CogVideoXTransformer3DModelTracking, CogVideoXPipelineTracking
from models.cogvidex_combination import CogVideoXTransformer3DModelCombination, CogVideoXPipelineCombination

logger = get_logger(__name__)

def load_vae(args, device, weight_dtype):
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",  
        revision=args.revision,
        variant=args.variant
    )
    
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()
        
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    
    vae.eval()
    
    return vae

def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"video_{i}.mp4"},
                }
            )

    model_description = f"""
# CogVideoX Full Finetune

<Gallery />

## Model description

This is a full finetune of the CogVideoX model `{base_model}`.

The model was trained using [CogVideoX Factory](https://github.com/a-r-r-o-w/cogvideox-factory) - a repository containing memory-optimized training scripts for the CogVideoX family of models using [TorchAO](https://github.com/pytorch/ao) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). The scripts were adopted from [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("{repo_id}", torch_dtype=torch.bfloat16).to("cuda")

video = pipe("{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

For more details, checkout the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) for CogVideoX.

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) and [here](https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "cogvideox",
        "cogvideox-diffusers",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    accelerator: Accelerator,
    pipe: Union[CogVideoXPipeline, CogVideoXPipelineTracking, CogVideoXImageToVideoPipelineCombination],
    vae: Union[AutoencoderKLCogVideoX, None],
    dataset: Union[VideoDatasetWithResizingTracking, None],
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    epoch,
    is_final_validation: bool = False,
    random_flip: Optional[float] = None,
    initial_frames_num: Optional[int] = 1,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    tracking_map_path = pipeline_args.pop("tracking_map_path", None)
    normal_map_path = pipeline_args.pop("normal_map_path", None)
    depth_map_path = pipeline_args.pop("depth_map_path", None)
    seg_mask_path = pipeline_args.pop("seg_mask_path", None)
    hand_keypoints_path = pipeline_args.pop("hand_keypoints_path", None)

    try:
        tracking_maps = pipeline_args.pop("tracking_maps", None)
    except:
        tracking_maps = None
    
    try:
        normal_maps = pipeline_args.pop("normal_maps", None)
    except:
        normal_maps = None
    
    try:
        depth_maps = pipeline_args.pop("depth_maps", None)
    except:
        depth_maps = None
    
    try:
        seg_masks = pipeline_args.pop("seg_masks", None)
    except:
        seg_masks = None
    
    try:
        hand_keypoints = pipeline_args.pop("hand_keypoints", None)
    except:
        hand_keypoints = None
    
    if tracking_map_path:
        from torchvision.transforms.functional import resize
        from torchvision import transforms
        
        video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(dataset.identity_transform),
                transforms.Lambda(dataset.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        tracking_map_path = Path(tracking_map_path)
        tracking_reader = decord.VideoReader(uri=tracking_map_path.as_posix())
        frame_indices = list(range(0, len(tracking_reader)))
        tracking_frames = tracking_reader.get_batch(frame_indices)[:args.frame_buckets[0]]
        nearest_res = dataset._find_nearest_resolution(tracking_frames.shape[2], tracking_frames.shape[3])
        tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
        tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
        tracking_frames = torch.stack([video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)

        normal_map_path = Path(normal_map_path)
        normal_reader = decord.VideoReader(uri=normal_map_path.as_posix())
        normal_frames = normal_reader.get_batch(frame_indices[:args.frame_buckets[0]])
        normal_frames = normal_frames.permute(0, 3, 1, 2).contiguous()
        normal_frames_resized = torch.stack([resize(normal_frame, nearest_res) for normal_frame in normal_frames], dim=0)
        normal_frames = torch.stack([video_transforms(normal_frame) for normal_frame in normal_frames_resized], dim=0)

        depth_map_path = Path(depth_map_path)
        depth_reader = decord.VideoReader(uri=depth_map_path.as_posix())
        depth_frames = depth_reader.get_batch(frame_indices[:args.frame_buckets[0]])
        depth_frames = depth_frames.permute(0, 3, 1, 2).contiguous()
        depth_frames_resized = torch.stack([resize(depth_frame, nearest_res) for depth_frame in depth_frames], dim=0)
        depth_frames = torch.stack([video_transforms(depth_frame) for depth_frame in depth_frames_resized], dim=0)

        seg_mask_path = Path(seg_mask_path)
        seg_mask_reader = decord.VideoReader(uri=seg_mask_path.as_posix())
        seg_mask_frames = seg_mask_reader.get_batch(frame_indices[:args.frame_buckets[0]])
        seg_mask_frames = seg_mask_frames.permute(0, 3, 1, 2).contiguous()
        seg_mask_frames_resized = torch.stack([resize(seg_mask_frame, nearest_res) for seg_mask_frame in seg_mask_frames], dim=0)
        seg_mask_frames = torch.stack([video_transforms(seg_mask_frame) for seg_mask_frame in seg_mask_frames_resized], dim=0)

        hand_keypoints_path = Path(hand_keypoints_path)
        hand_keypoints_reader = decord.VideoReader(uri=hand_keypoints_path.as_posix())
        hand_keypoints_frames = hand_keypoints_reader.get_batch(frame_indices[:args.frame_buckets[0]])
        hand_keypoints_frames = hand_keypoints_frames.permute(0, 3, 1, 2).contiguous()
        hand_keypoints_frames_resized = torch.stack([resize(hand_keypoints_frame, nearest_res) for hand_keypoints_frame in hand_keypoints_frames], dim=0)
        hand_keypoints_frames = torch.stack([video_transforms(hand_keypoints_frame) for hand_keypoints_frame in hand_keypoints_frames_resized], dim=0)

        # if initial_frames_num > 1:
        #     pipeline_args["image"] = frames[:initial_frames_num].clone()

        tracking_image = tracking_frames[:initial_frames_num].clone()
        normal_image = normal_frames[:initial_frames_num].clone()
        depth_image = depth_frames[:initial_frames_num].clone()
        seg_image = seg_mask_frames[:initial_frames_num].clone()
        hand_keypoints_image = hand_keypoints_frames[:initial_frames_num].clone()

        pipeline_args["tracking_image"] = tracking_image
        pipeline_args["normal_image"] = normal_image
        pipeline_args["depth_image"] = depth_image
        pipeline_args["seg_image"] = seg_image
        pipeline_args["hand_keypoints_image"] = hand_keypoints_image
        
        # vae encode condition frames from path
        with torch.no_grad():
            tracking_frames = tracking_frames.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
            tracking_frames = tracking_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
            tracking_latent_dist = vae.encode(tracking_frames).latent_dist
            tracking_maps = tracking_latent_dist.sample() * vae.config.scaling_factor
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # to [B, F, C, H, W]
            tracking_maps = tracking_maps.to(memory_format=torch.contiguous_format, dtype=accelerator.unwrap_model(vae).dtype)

            normal_frames = normal_frames.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
            normal_frames = normal_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
            normal_latent_dist = vae.encode(normal_frames).latent_dist
            normal_maps = normal_latent_dist.sample() * vae.config.scaling_factor
            normal_maps = normal_maps.permute(0, 2, 1, 3, 4)  # to [B, F, C, H, W]
            normal_maps = normal_maps.to(memory_format=torch.contiguous_format, dtype=accelerator.unwrap_model(vae).dtype)

            depth_frames = depth_frames.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
            depth_frames = depth_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
            depth_latent_dist = vae.encode(depth_frames).latent_dist
            depth_maps = depth_latent_dist.sample() * vae.config.scaling_factor
            depth_maps = depth_maps.permute(0, 2, 1, 3, 4)  # to [B, F, C, H, W]
            depth_maps = depth_maps.to(memory_format=torch.contiguous_format, dtype=accelerator.unwrap_model(vae).dtype)
            
            seg_mask_frames = seg_mask_frames.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
            seg_mask_frames = seg_mask_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
            seg_mask_latent_dist = vae.encode(seg_mask_frames).latent_dist
            seg_masks = seg_mask_latent_dist.sample() * vae.config.scaling_factor
            seg_masks = seg_masks.permute(0, 2, 1, 3, 4)
            seg_masks = seg_masks.to(memory_format=torch.contiguous_format, dtype=accelerator.unwrap_model(vae).dtype)

            hand_keypoints_frames = hand_keypoints_frames.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
            hand_keypoints_frames = hand_keypoints_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
            hand_keypoints_latent_dist = vae.encode(hand_keypoints_frames).latent_dist
            hand_keypoints = hand_keypoints_latent_dist.sample() * vae.config.scaling_factor
            hand_keypoints = hand_keypoints.permute(0, 2, 1, 3, 4)
            hand_keypoints = hand_keypoints.to(memory_format=torch.contiguous_format, dtype=accelerator.unwrap_model(vae).dtype)

    pipe = pipe.to(accelerator.device)

    pipeline_args["tracking_maps"] = tracking_maps
    pipeline_args["normal_maps"] = normal_maps
    pipeline_args["depth_maps"] = depth_maps
    pipeline_args["seg_masks"] = seg_masks
    pipeline_args["hand_keypoints"] = hand_keypoints

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    if args.num_validation_videos > 2:
        args.num_validation_videos = 2  # 限制验证视频数量

    videos = []
    for _ in range(args.num_validation_videos):
        with torch.no_grad():
            video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_ep{epoch}_{i}th_{prompt}.mp4")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                export_to_video(video, filename, fps=24)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )
    torch.cuda.empty_cache()
    return videos


class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "videos": videos,
            "prompts": prompts,
        }

class CollateFunctionTracking:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_maps = [x["tracking_map"] for x in data[0]]
        tracking_maps = torch.stack(tracking_maps).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "videos": videos,
            "prompts": prompts,
            "tracking_maps": tracking_maps,
        }

class CollateFunctionImageTracking:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_maps = [x["tracking_map"] for x in data[0]]
        tracking_maps = torch.stack(tracking_maps).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "tracking_maps": tracking_maps,
        }

class CollateFunctionMultiCondition:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_maps = [x["tracking_map"] for x in data[0]]
        tracking_maps = torch.stack(tracking_maps).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_images = [x["tracking_image"] for x in data[0]]
        tracking_images = torch.stack(tracking_images).to(dtype=self.weight_dtype, non_blocking=True)
        
        depth_maps = [x["depth_map"] for x in data[0]]
        depth_maps = torch.stack(depth_maps).to(dtype=self.weight_dtype, non_blocking=True)

        depth_images = [x["depth_image"] for x in data[0]]
        depth_images = torch.stack(depth_images).to(dtype=self.weight_dtype, non_blocking=True)
        
        normal_maps = [x["normal_map"] for x in data[0]]
        normal_maps = torch.stack(normal_maps).to(dtype=self.weight_dtype, non_blocking=True)
        
        normal_images = [x['normal_image'] for x in data[0]]
        normal_images = torch.stack(normal_images).to(dtype=self.weight_dtype, non_blocking=True)
        
        seg_masks = [x["seg_mask"] for x in data[0]]
        seg_masks = torch.stack(seg_masks).to(dtype=self.weight_dtype, non_blocking=True)

        seg_mask_images = [x["seg_mask_image"] for x in data[0]]
        seg_mask_images = torch.stack(seg_mask_images).to(dtype=self.weight_dtype, non_blocking=True)
        
        hand_keypoints = [x["hand_keypoints"] for x in data[0]]
        hand_keypoints = torch.stack(hand_keypoints).to(dtype=self.weight_dtype, non_blocking=True)

        hand_keypoints_images = [x["hand_keypoints_image"] for x in data[0]]
        hand_keypoints_images = torch.stack(hand_keypoints_images).to(dtype=self.weight_dtype, non_blocking=True)
        
        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "tracking_maps": tracking_maps,
            "tracking_images": tracking_images,
            "depth_maps": depth_maps,
            "depth_images": depth_images,
            "normal_maps": normal_maps,
            "normal_images": normal_images,
            "seg_masks": seg_masks,
            "seg_mask_images": seg_mask_images,
            "hand_keypoints": hand_keypoints,
            "hand_keypoints_images": hand_keypoints_images,
        }

def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    if not args.tracking_column:
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
        )
    elif args.tracking_column and args.depth_column:
        transformer = CogVideoXTransformer3DModelCombination.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
            num_tracking_blocks=args.num_tracking_blocks,
        )
    elif args.tracking_column:
        transformer = CogVideoXTransformer3DModelTracking.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
            num_tracking_blocks=args.num_tracking_blocks,
        )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    # transformer.requires_grad_(True)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model: CogVideoXTransformer3DModelCombination
                    model = unwrap_model(model)
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = CogVideoXTransformer3DModel.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                )
                init_under_meta = True

        load_model = CogVideoXTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    def load_model_hook_tracking(models, input_dir):
        transformer_ = None
        init_under_meta = False

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = CogVideoXTransformer3DModelTracking.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer", num_tracking_blocks=args.num_tracking_blocks
                )
                init_under_meta = True

        load_model = CogVideoXTransformer3DModelTracking.from_pretrained(os.path.join(input_dir, "transformer"))
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook if args.resume_from_checkpoint is None else load_model_hook_tracking)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )


    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        pass
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            


    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    # Add initial validation before training starts
    if accelerator.is_main_process:
        accelerator.print("===== Memory before initial validation =====")
        print_memory(accelerator.device)
        torch.cuda.synchronize(accelerator.device)
        transformer.eval()

        if args.tracking_column is None:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unwrap_model(transformer),
                scheduler=scheduler,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
        elif args.tracking_column and args.depth_column:
            pipe = CogVideoXImageToVideoPipelineCombination.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unwrap_model(transformer),
                scheduler=scheduler,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
        elif args.tracking_column:
            pipe = CogVideoXImageToVideoPipelineTracking.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unwrap_model(transformer),
                scheduler=scheduler,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            
        dataset_init_kwargs = {
            "data_root": args.data_root,
            "dataset_file": args.dataset_file,
            "caption_column": args.caption_column,
            "tracking_column": args.tracking_column, 
            "normal_column": args.normal_column,
            "depth_column": args.depth_column,
            "label_column": args.label_column,
            "seg_mask_column": args.seg_mask_column,
            "hand_keypoints_column": args.hand_keypoints_column,
            "image_column": args.image_column,
            "tracking_image_column": args.tracking_image_column,
            "normal_image_column": args.normal_image_column,
            "depth_image_column": args.depth_image_column,
            "seg_mask_image_column": args.seg_mask_image_column,
            "hand_keypoints_image_column": args.hand_keypoints_image_column,
            "video_column": args.video_column,
            "max_num_frames": args.max_num_frames,
            "id_token": args.id_token,
            "height_buckets": args.height_buckets,
            "width_buckets": args.width_buckets,
            "frame_buckets": args.frame_buckets,
            "load_tensors": args.load_tensors,
            "random_flip": args.random_flip,
            "image_to_video": True,
            "random_mask": args.random_mask,
            "initial_frames_num": args.initial_frames_num,
        }
        train_dataset = HOIVideoDatasetResizing(**dataset_init_kwargs)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        
        validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
        validation_images = args.validation_images.split(args.validation_prompt_separator)
        
        for validation_image, validation_prompt in zip(validation_images, validation_prompts):
            pipeline_args = {
                "image": load_image(validation_image),
                "prompt": validation_prompt,
                "negative_prompt": "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
                "max_sequence_length": model_config.max_text_seq_length,
            }

            if args.tracking_column is not None:
                pipeline_args["tracking_map_path"] = args.tracking_map_path
            
            if args.depth_column is not None:
                pipeline_args["depth_map_path"] = args.depth_map_path
            
            if args.normal_column is not None:
                pipeline_args["normal_map_path"] = args.normal_map_path
            
            if args.seg_mask_column is not None or args.label_column is not None:
                pipeline_args["seg_mask_path"] = args.seg_mask_path
            
            if args.hand_keypoints_column is not None or args.label_column is not None:
                pipeline_args["hand_keypoints_path"] = args.hand_keypoints_path

            # vae = load_vae(args, device=accelerator.device, weight_dtype=weight_dtype)
            
            log_validation(
                accelerator=accelerator,
                pipe=pipe,
                vae=vae,
                dataset=train_dataset,
                args=args,
                pipeline_args=pipeline_args,
                epoch=0,
                is_final_validation=False,
                initial_frames_num=args.initial_frames_num
            )


if __name__ == "__main__":
    args = get_args()
    main(args)


