#!/usr/bin/env python3
# Only save for 1 frame referenece image
# For 3 frames reference image, please refer to 

import argparse
import functools
import json
import os
import pathlib
import queue
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.training_utils import set_seed
from diffusers.utils import export_to_video, get_logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


import decord  # isort:skip

from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop, HOIVideoDatasetResizing, Oakink2VideoDatasetResizing  # isort:skip


decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def check_height(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--height_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_width(x: Any) -> int:
    x = int(x)
    if x % 16 != 0:
        raise argparse.ArgumentTypeError(
            f"`--width_buckets` must be divisible by 16, but got {x} which does not fit criteria."
        )
    return x


def check_frames(x: Any) -> int:
    x = int(x)
    if x % 4 != 0 and x % 4 != 1:
        raise argparse.ArgumentTypeError(
            f"`--frames_buckets` must be of form `4 * k` or `4 * k + 1`, but got {x} which does not fit criteria."
        )
    return x


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-5b-I2V",
        help="Hugging Face model ID to use for tokenizer, text encoder and VAE.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to where training data is located.")
    parser.add_argument(
        "--dataset_file", type=str, default=None, help="Path to CSV file containing metadata about training data."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the captions. If using the folder structure format for data loading, this should be the name of the file containing line-separated captions (the file should be located in `--data_root`).",
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the video paths. If using the folder structure format for data loading, this should be the name of the file containing line-separated video paths (the file should be located in `--data_root`).",
    )
    parser.add_argument(
        "--tracking_column",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--normal_column",
        type=str,
        default=None
    )
    parser.add_argument(
        "--depth_column",
        type=str,
        default=None
    )
    parser.add_argument(
        "--seg_column",
        type=str,
        default=None
    )
    parser.add_argument(
        "--hand_column",
        type=str,
        default=None
    )
    parser.add_argument(
        "--id_token",
        type=str,
        default=None,
        help="Identifier token appended to the start of each prompt if provided.",
    )
    parser.add_argument(
        "--height_buckets",
        nargs="+",
        type=check_height,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--width_buckets",
        nargs="+",
        type=check_width,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--frame_buckets",
        nargs="+",
        type=check_frames,
        default=[49],
    )
    parser.add_argument(
        "--random_flip",
        type=float,
        default=None,
        help="If random horizontal flip augmentation is to be used, this should be the flip probability.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default=None,
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument(
        "--save_image_latents",
        action="store_true",
        help="Whether or not to encode and store image latents, which are required for image-to-video finetuning. The image latents are the first frame of input videos encoded with the VAE.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory where preprocessed videos/latents/embeddings will be saved.",
    )
    parser.add_argument("--max_num_frames", type=int, default=49, help="Maximum number of frames in output video.")
    parser.add_argument(
        "--max_sequence_length", type=int, default=226, help="Max sequence length of prompt embeddings."
    )
    parser.add_argument("--target_fps", type=int, default=8, help="Frame rate of output videos.")
    parser.add_argument(
        "--save_latents_and_embeddings",
        action="store_true",
        help="Whether to encode videos/captions to latents/embeddings and save them in pytorch serializable format.",
    )
    parser.add_argument(
        "--use_slicing",
        action="store_true",
        help="Whether to enable sliced encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument(
        "--use_tiling",
        action="store_true",
        help="Whether to enable tiled encoding/decoding in the VAE. Only used if `--save_latents_and_embeddings` is also used.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of videos to process at once in the VAE.")
    parser.add_argument(
        "--num_decode_threads",
        type=int,
        default=0,
        help="Number of decoding threads for `decord` to use. The default `0` means to automatically determine required number of threads.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Data type to use when generating latents and prompt embeddings.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--num_artifact_workers", type=int, default=4, help="Number of worker threads for serializing artifacts."
    )
    return parser.parse_args()


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompts: List[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompts,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompts,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


to_pil_image = transforms.ToPILImage(mode="RGB")


def save_image(image: torch.Tensor, path: pathlib.Path) -> None:
    image = to_pil_image(image)
    image.save(path)


def save_video(video: torch.Tensor, path: pathlib.Path, fps: int = 8) -> None:
    video = [to_pil_image(frame) for frame in video]
    export_to_video(video, path, fps=fps)


def save_prompt(prompt: str, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(prompt)


def save_metadata(metadata: Dict[str, Any], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(metadata))


@torch.no_grad()
def serialize_artifacts(
    batch_size: int,
    fps: int,
    images_dir: Optional[pathlib.Path] = None,
    image_latents_dir: Optional[pathlib.Path] = None,
    videos_dir: Optional[pathlib.Path] = None,
    video_latents_dir: Optional[pathlib.Path] = None,
    prompts_dir: Optional[pathlib.Path] = None,
    prompt_embeds_dir: Optional[pathlib.Path] = None,
    images: Optional[torch.Tensor] = None,
    image_latents: Optional[torch.Tensor] = None,
    videos: Optional[torch.Tensor] = None,
    video_latents: Optional[torch.Tensor] = None,
    prompts: Optional[List[str]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    tracking_maps: Optional[torch.Tensor] = None,
    tracking_map_latents: Optional[torch.Tensor] = None,
    tracking_images: Optional[torch.Tensor] = None,
    tracking_images_latents: Optional[torch.Tensor] = None,
    depth_maps: Optional[torch.Tensor] = None,
    depth_map_latents: Optional[torch.Tensor] = None,
    depth_images: Optional[torch.Tensor] = None,
    depth_images_latents: Optional[torch.Tensor] = None,
    normal_maps: Optional[torch.Tensor] = None,
    normal_map_latents: Optional[torch.Tensor] = None,
    normal_images: Optional[torch.Tensor] = None,
    normal_images_latents: Optional[torch.Tensor] = None,
    seg_masks: Optional[torch.Tensor] = None,
    seg_mask_latents: Optional[torch.Tensor] = None,
    seg_mask_images: Optional[torch.Tensor] = None,
    seg_mask_images_latents: Optional[torch.Tensor] = None,
    hand_keypoints: Optional[torch.Tensor] = None,
    hand_keypoints_latents: Optional[torch.Tensor] = None,
    hand_keypoints_images: Optional[torch.Tensor] = None,
    hand_keypoints_images_latents: Optional[torch.Tensor] = None,
    tracking_dir: Optional[pathlib.Path] = None,
    tracking_latents_dir: Optional[pathlib.Path] = None,
    tracking_images_dir: Optional[pathlib.Path] = None,
    tracking_images_latents_dir: Optional[pathlib.Path] = None,
    depth_dir: Optional[pathlib.Path] = None,
    depth_latents_dir: Optional[pathlib.Path] = None,
    depth_images_dir: Optional[pathlib.Path] = None,
    depth_images_latents_dir: Optional[pathlib.Path] = None,
    normal_dir: Optional[pathlib.Path] = None,
    normal_latents_dir: Optional[pathlib.Path] = None,
    normal_images_dir: Optional[pathlib.Path] = None,
    normal_images_latents_dir: Optional[pathlib.Path] = None,
    seg_mask_dir: Optional[pathlib.Path] = None,
    seg_mask_latents_dir: Optional[pathlib.Path] = None,
    seg_mask_images_dir: Optional[pathlib.Path] = None,
    seg_mask_images_latents_dir: Optional[pathlib.Path] = None,
    hand_keypoints_dir: Optional[pathlib.Path] = None,
    hand_keypoints_latents_dir: Optional[pathlib.Path] = None,
    hand_keypoints_images_dir: Optional[pathlib.Path] = None,
    hand_keypoints_images_latents_dir: Optional[pathlib.Path] = None,
) -> None:
    num_frames, height, width = videos.size(1), videos.size(3), videos.size(4)
    metadata = [{"num_frames": num_frames, "height": height, "width": width}]

    data_folder_mapper_list = [
        (images, images_dir, lambda img, path: save_image(img[0], path), "png"),
        (image_latents, image_latents_dir, torch.save, "pt"),
        (videos, videos_dir, functools.partial(save_video, fps=fps), "mp4"),
        (video_latents, video_latents_dir, torch.save, "pt"),
        (prompts, prompts_dir, save_prompt, "txt"),
        (prompt_embeds, prompt_embeds_dir, torch.save, "pt"),
        (metadata, videos_dir, save_metadata, "txt"),
        (tracking_maps, tracking_dir, functools.partial(save_video, fps=fps), "mp4"),
        (tracking_map_latents, tracking_latents_dir, torch.save, "pt"),
        (tracking_images, tracking_images_dir, lambda img, path: save_image(img[0], path), "png"),
        (tracking_images_latents, tracking_images_latents_dir, torch.save, "pt"),
        (depth_maps, depth_dir, functools.partial(save_video, fps=fps), "mp4"),
        (depth_map_latents, depth_latents_dir, torch.save, "pt"),
        (depth_images, depth_images_dir, lambda img, path: save_image(img[0], path), "png"),
        (depth_images_latents, depth_images_latents_dir, torch.save, "pt"),
        (normal_maps, normal_dir, functools.partial(save_video, fps=fps), "mp4"),
        (normal_map_latents, normal_latents_dir, torch.save, "pt"),
        (normal_images, normal_images_dir, lambda img, path: save_image(img[0], path), "png"),
        (normal_images_latents, normal_images_latents_dir, torch.save, "pt"),
        (seg_masks, seg_mask_dir, functools.partial(save_video, fps=fps), "mp4"),
        (seg_mask_latents, seg_mask_latents_dir, torch.save, "pt"),
        (seg_mask_images, seg_mask_images_dir, lambda img, path: save_image(img[0], path), "png"),
        (seg_mask_images_latents, seg_mask_images_latents_dir, torch.save, "pt"),
        (hand_keypoints, hand_keypoints_dir, functools.partial(save_video, fps=fps), "mp4"),
        (hand_keypoints_latents, hand_keypoints_latents_dir, torch.save, "pt"),
        (hand_keypoints_images, hand_keypoints_images_dir, lambda img, path: save_image(img[0], path), "png"),
        (hand_keypoints_images_latents, hand_keypoints_images_latents_dir, torch.save, "pt"),
    ]

    filenames = [uuid.uuid4() for _ in range(batch_size)]

    for data, folder, save_fn, extension in data_folder_mapper_list:
        if data is None:
            continue
        for slice, filename in zip(data, filenames):
            if isinstance(slice, torch.Tensor):
                slice = slice.clone().to("cpu")
            path = folder.joinpath(f"{filename}.{extension}")
            save_fn(slice, path)


def save_intermediates(output_queue: queue.Queue) -> None:
    while True:
        try:
            item = output_queue.get(timeout=30)
            if item is None:
                break
            serialize_artifacts(**item)

        except queue.Empty:
            continue


@torch.no_grad()
def main():
    args = get_args()
    set_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    tmp_dir = output_dir.joinpath("tmp")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create task queue for non-blocking serializing of artifacts
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=args.num_artifact_workers)
    save_future = save_thread.submit(save_intermediates, output_queue)

    # Initialize distributed processing
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        rank = 0
        torch.cuda.set_device(rank)

    # Create folders where intermediate tensors from each rank will be saved
    images_dir = tmp_dir.joinpath(f"images/{rank}")
    image_latents_dir = tmp_dir.joinpath(f"image_latents/{rank}")
    videos_dir = tmp_dir.joinpath(f"videos/{rank}")
    video_latents_dir = tmp_dir.joinpath(f"video_latents/{rank}")
    prompts_dir = tmp_dir.joinpath(f"prompts/{rank}")
    prompt_embeds_dir = tmp_dir.joinpath(f"prompt_embeds/{rank}")
    tracking_dir = tmp_dir.joinpath(f"tracking/{rank}")
    tracking_latents_dir = tmp_dir.joinpath(f"tracking_latents/{rank}")
    tracking_images_dir = tmp_dir.joinpath(f"tracking_images/{rank}")
    tracking_images_latents_dir = tmp_dir.joinpath(f"tracking_images_latents/{rank}")
    depth_dir = tmp_dir.joinpath(f"depth/{rank}")
    depth_latents_dir = tmp_dir.joinpath(f"depth_latents/{rank}")
    depth_images_dir = tmp_dir.joinpath(f"depth_images/{rank}")
    depth_images_latents_dir = tmp_dir.joinpath(f"depth_images_latents/{rank}")
    normal_dir = tmp_dir.joinpath(f"normal/{rank}")
    normal_latents_dir = tmp_dir.joinpath(f"normal_latents/{rank}")
    normal_images_dir = tmp_dir.joinpath(f"normal_images/{rank}")
    normal_images_latents_dir = tmp_dir.joinpath(f"normal_images_latents/{rank}")
    seg_mask_dir = tmp_dir.joinpath(f"seg_mask/{rank}")
    seg_mask_latents_dir = tmp_dir.joinpath(f"seg_mask_latents/{rank}")
    seg_mask_images_dir = tmp_dir.joinpath(f"seg_mask_images/{rank}")
    seg_mask_images_latents_dir = tmp_dir.joinpath(f"seg_mask_images_latents/{rank}")
    hand_keypoints_dir = tmp_dir.joinpath(f"hand_keypoints/{rank}")
    hand_keypoints_latents_dir = tmp_dir.joinpath(f"hand_keypoints_latents/{rank}")
    hand_keypoints_images_dir = tmp_dir.joinpath(f"hand_keypoints_images/{rank}")
    hand_keypoints_images_latents_dir = tmp_dir.joinpath(f"hand_keypoints_images_latents/{rank}")


    images_dir.mkdir(parents=True, exist_ok=True)
    image_latents_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_latents_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeds_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_latents_dir.mkdir(parents=True, exist_ok=True)
    tracking_images_dir.mkdir(parents=True, exist_ok=True)
    tracking_images_latents_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_latents_dir.mkdir(parents=True, exist_ok=True)
    depth_images_dir.mkdir(parents=True, exist_ok=True)
    depth_images_latents_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    normal_latents_dir.mkdir(parents=True, exist_ok=True)
    normal_images_dir.mkdir(parents=True, exist_ok=True)
    normal_images_latents_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_latents_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_images_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_images_latents_dir.mkdir(parents=True, exist_ok=True)
    hand_keypoints_dir.mkdir(parents=True, exist_ok=True)
    hand_keypoints_latents_dir.mkdir(parents=True, exist_ok=True)
    hand_keypoints_images_dir.mkdir(parents=True, exist_ok=True)
    hand_keypoints_images_latents_dir.mkdir(parents=True, exist_ok=True)


    weight_dtype = DTYPE_MAPPING[args.dtype]
    target_fps = args.target_fps

    # 1. Dataset
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": False,
        "random_flip": args.random_flip,
        "image_to_video": args.save_image_latents,
        "tracking_column": args.depth_column,
        "normal_column": args.depth_column,
        "depth_column": args.depth_column,
        "hand_keypoints_column": args.hand_column,
        "seg_mask_column": args.seg_column,
        "random_mask": False,
        "initial_frames_num": 1
    }
    if args.video_reshape_mode is None:
        dataset = Oakink2VideoDatasetResizing(**dataset_init_kwargs)
    else:
        dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
        )

    original_dataset_size = len(dataset)

    # Split data among GPUs
    if world_size > 1:
        samples_per_gpu = original_dataset_size // world_size
        start_index = rank * samples_per_gpu
        end_index = start_index + samples_per_gpu
        if rank == world_size - 1:
            end_index = original_dataset_size  # Make sure the last GPU gets the remaining data

        # Slice the data
        dataset.prompts = dataset.prompts[start_index:end_index]
        dataset.video_paths = dataset.video_paths[start_index:end_index]
        dataset.tracking_paths = dataset.tracking_paths[start_index:end_index]
        dataset.normal_paths = dataset.normal_paths[start_index:end_index]
        dataset.depth_paths = dataset.depth_paths[start_index:end_index]
        dataset.label_paths = dataset.label_paths[start_index:end_index]
    else:
        pass

    rank_dataset_size = len(dataset)

    # 2. Dataloader
    def collate_fn(data):
        prompts = [x["prompt"] for x in data[0]]

        images = None
        if args.save_image_latents:
            images = [x["image"] for x in data[0]]
            images = torch.stack(images).to(dtype=weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=weight_dtype, non_blocking=True)

        tracking_maps = [x["tracking_map"] for x in data[0]]
        tracking_maps = torch.stack(tracking_maps).to(dtype=weight_dtype, non_blocking=True)
        
        tracking_images = [x["tracking_image"] for x in data[0]]
        tracking_images = torch.stack(tracking_images).to(dtype=weight_dtype, non_blocking=True)
        
        normal_maps = [x["normal_map"] for x in data[0]]
        normal_maps = torch.stack(normal_maps).to(dtype=weight_dtype, non_blocking=True)

        normal_images = [x["normal_image"] for x in data[0]]
        normal_images = torch.stack(normal_images).to(dtype=weight_dtype, non_blocking=True)
        
        depth_maps = [x["depth_map"] for x in data[0]]
        depth_maps = torch.stack(depth_maps).to(dtype=weight_dtype, non_blocking=True)
        
        depth_images = [x["depth_image"] for x in data[0]]
        depth_images = torch.stack(depth_images).to(dtype=weight_dtype, non_blocking=True)
        
        seg_masks = [x["seg_mask"] for x in data[0]]
        seg_masks = torch.stack(seg_masks).to(dtype=weight_dtype, non_blocking=True)
        
        seg_mask_images = [x["seg_mask_image"] for x in data[0]]
        seg_mask_images = torch.stack(seg_mask_images).to(dtype=weight_dtype, non_blocking=True)
        
        hand_keypoints = [x["hand_keypoints"] for x in data[0]]
        hand_keypoints = torch.stack(hand_keypoints).to(dtype=weight_dtype, non_blocking=True)
        
        hand_keypoints_images = [x["hand_keypoints_image"] for x in data[0]]
        hand_keypoints_images = torch.stack(hand_keypoints_images).to(dtype=weight_dtype, non_blocking=True)
        
        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "tracking_maps": tracking_maps,
            "tracking_images": tracking_images,
            "normal_maps": normal_maps,
            "normal_images": normal_images,
            "depth_maps": depth_maps,
            "depth_images": depth_images,
            "seg_masks": seg_masks,
            "seg_mask_images": seg_mask_images,
            "hand_keypoints": hand_keypoints,
            "hand_keypoints_images": hand_keypoints_images,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=BucketSampler(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False),
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    # 3. Prepare models
    device = f"cuda:{rank}"

    if args.save_latents_and_embeddings:
        tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.model_id, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.to(device)

        vae = AutoencoderKLCogVideoX.from_pretrained(args.model_id, subfolder="vae", torch_dtype=weight_dtype)
        vae = vae.to(device)

        if args.use_slicing:
            vae.enable_slicing()
        if args.use_tiling:
            vae.enable_tiling()

    # 4. Compute latents and embeddings and save
    if rank == 0:
        iterator = tqdm(
            dataloader, desc="Encoding", total=(rank_dataset_size + args.batch_size - 1) // args.batch_size
        )
    else:
        iterator = dataloader

    for step, batch in enumerate(iterator):
        try:
            images = None
            image_latents = None
            video_latents = None
            prompt_embeds = None

            tracking_map_latents = None
            tracking_images_latents = None
            depth_map_latents = None
            depth_images_latents = None
            normal_map_latents = None
            normal_images_latents = None
            seg_mask_latents = None
            seg_mask_images_latents = None
            hand_keypoints_latents = None
            hand_keypoints_images_latents = None

            if args.save_image_latents:
                images = batch["images"].to(device, non_blocking=True)
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                
                tracking_images = batch["tracking_images"].to(device, non_blocking=True)
                tracking_images = tracking_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                
                depth_images = batch["depth_images"].to(device, non_blocking=True)
                depth_images = depth_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                
                normal_images = batch["normal_images"].to(device, non_blocking=True)
                normal_images = normal_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                
                seg_mask_images = batch["seg_mask_images"].to(device, non_blocking=True)
                seg_mask_images = seg_mask_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                
                hand_keypoints_images = batch["hand_keypoints_images"].to(device, non_blocking=True)
                hand_keypoints_images = hand_keypoints_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                
            videos = batch["videos"].to(device, non_blocking=True)
            videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            tracking_maps = batch["tracking_maps"].to(device, non_blocking=True)
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            depth_maps = batch["depth_maps"].to(device, non_blocking=True)
            depth_maps = depth_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            normal_maps = batch["normal_maps"].to(device, non_blocking=True)
            normal_maps = normal_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            seg_masks = batch["seg_masks"].to(device, non_blocking=True)
            seg_masks = seg_masks.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            hand_keypoints = batch["hand_keypoints"].to(device, non_blocking=True)
            hand_keypoints = hand_keypoints.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            prompts = batch["prompts"]

            # Encode videos & images
            if args.save_latents_and_embeddings:
                if args.use_slicing:
                    if args.save_image_latents:
                        encoded_slices = [vae._encode(image_slice) for image_slice in images.split(1)]
                        image_latents = torch.cat(encoded_slices)
                        image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        tracking_images_latents = torch.cat([vae._encode(tracking_image_slice) for tracking_image_slice in tracking_images.split(1)])
                        tracking_images_latents = tracking_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        depth_images_latents = torch.cat([vae._encode(depth_image_slice) for depth_image_slice in depth_images.split(1)])
                        depth_images_latents = depth_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        normal_images_latents = torch.cat([vae._encode(normal_image_slice) for normal_image_slice in normal_images.split(1)])
                        normal_images_latents = normal_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        seg_mask_images_latents = torch.cat([vae._encode(seg_mask_image_slice) for seg_mask_image_slice in seg_mask_images.split(1)])
                        seg_mask_images_latents = seg_mask_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        hand_keypoints_images_latents = torch.cat([vae._encode(hand_keypoints_image_slice) for hand_keypoints_image_slice in hand_keypoints_images.split(1)])
                        hand_keypoints_images_latents = hand_keypoints_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    encoded_slices = [vae._encode(video_slice) for video_slice in videos.split(1)]
                    video_latents = torch.cat(encoded_slices)

                    encoded_slices = [vae._encode(tracking_map_slice) for tracking_map_slice in tracking_maps.split(1)]
                    tracking_map_latents = torch.cat(encoded_slices)

                    encoded_slices = [vae._encode(depth_map_slice) for depth_map_slice in depth_maps.split(1)]
                    depth_map_latents = torch.cat(encoded_slices)

                    encoded_slices = [vae._encode(normal_map_slice) for normal_map_slice in normal_maps.split(1)]
                    normal_map_latents = torch.cat(encoded_slices)

                    encoded_slices = [vae._encode(seg_mask_slice) for seg_mask_slice in seg_masks.split(1)]
                    seg_mask_latents = torch.cat(encoded_slices)

                    encoded_slices = [vae._encode(hand_keypoints_slice) for hand_keypoints_slice in hand_keypoints.split(1)]
                    hand_keypoints_latents = torch.cat(encoded_slices)

                else:
                    if args.save_image_latents:
                        image_latents = vae._encode(images)
                        image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                        tracking_images_latents = vae._encode(tracking_images)
                        tracking_images_latents = tracking_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        depth_images_latents = vae._encode(depth_images)
                        depth_images_latents = depth_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        normal_images_latents = vae._encode(normal_images)
                        normal_images_latents = normal_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        seg_mask_images_latents = vae._encode(seg_mask_images)
                        seg_mask_images_latents = seg_mask_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                        hand_keypoints_images_latents = vae._encode(hand_keypoints_images)
                        hand_keypoints_images_latents = hand_keypoints_images_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                        
                    video_latents = vae._encode(videos)
                    tracking_map_latents = vae._encode(tracking_maps)
                    depth_map_latents = vae._encode(depth_maps)
                    normal_map_latents = vae._encode(normal_maps)
                    seg_mask_latents = vae._encode(seg_masks)
                    hand_keypoints_latents = vae._encode(hand_keypoints)

                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                tracking_map_latents = tracking_map_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                depth_map_latents = depth_map_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                normal_map_latents = normal_map_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                seg_mask_latents = seg_mask_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                hand_keypoints_latents = hand_keypoints_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # Encode prompts
                prompt_embeds = compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    prompts,
                    args.max_sequence_length,
                    device,
                    weight_dtype,
                    requires_grad=False,
                )

            if images is not None:
                images = (images.permute(0, 2, 1, 3, 4) + 1) / 2

            videos = (videos.permute(0, 2, 1, 3, 4) + 1) / 2
            tracking_maps = (tracking_maps.permute(0, 2, 1, 3, 4) + 1) / 2
            tracking_images = (tracking_images.permute(0, 2, 1, 3, 4) + 1) / 2
            depth_maps = (depth_maps.permute(0, 2, 1, 3, 4) + 1) / 2
            depth_images = (depth_images.permute(0, 2, 1, 3, 4) + 1) / 2
            normal_maps = (normal_maps.permute(0, 2, 1, 3, 4) + 1) / 2
            normal_images = (normal_images.permute(0, 2, 1, 3, 4) + 1) / 2
            seg_masks = (seg_masks.permute(0, 2, 1, 3, 4) + 1) / 2
            seg_mask_images = (seg_mask_images.permute(0, 2, 1, 3, 4) + 1) / 2
            hand_keypoints = (hand_keypoints.permute(0, 2, 1, 3, 4) + 1) / 2
            hand_keypoints_images = (hand_keypoints_images.permute(0, 2, 1, 3, 4) + 1) / 2
            
            output_queue.put(
                {
                    "batch_size": len(prompts),
                    "fps": target_fps,
                    "images_dir": images_dir,
                    "image_latents_dir": image_latents_dir,
                    "videos_dir": videos_dir,
                    "video_latents_dir": video_latents_dir,
                    "prompts_dir": prompts_dir,
                    "prompt_embeds_dir": prompt_embeds_dir,
                    "images": images,
                    "image_latents": image_latents,
                    "videos": videos,
                    "video_latents": video_latents,
                    "prompts": prompts,
                    "prompt_embeds": prompt_embeds,
                    "tracking_maps": tracking_maps,
                    "tracking_map_latents": tracking_map_latents,
                    "tracking_images": tracking_images,
                    "tracking_images_latents": tracking_images_latents,
                    "depth_maps": depth_maps,
                    "depth_map_latents": depth_map_latents,
                    "depth_images": depth_images,
                    "depth_images_latents": depth_images_latents,
                    "normal_maps": normal_maps,
                    "normal_map_latents": normal_map_latents,
                    "normal_images": normal_images,
                    "normal_images_latents": normal_images_latents,
                    "seg_masks": seg_masks,
                    "seg_mask_latents": seg_mask_latents,
                    "seg_mask_images": seg_mask_images,
                    "seg_mask_images_latents": seg_mask_images_latents,
                    "hand_keypoints": hand_keypoints,
                    "hand_keypoints_latents": hand_keypoints_latents,
                    "hand_keypoints_images": hand_keypoints_images,
                    "hand_keypoints_images_latents": hand_keypoints_images_latents,
                    "tracking_dir": tracking_dir,
                    "tracking_latents_dir": tracking_latents_dir,
                    "tracking_images_dir": tracking_images_dir,
                    "tracking_images_latents_dir": tracking_images_latents_dir,
                    "depth_dir": depth_dir,
                    "depth_latents_dir": depth_latents_dir,
                    "depth_images_dir": depth_images_dir,
                    "depth_images_latents_dir": depth_images_latents_dir,
                    "normal_dir": normal_dir,
                    "normal_latents_dir": normal_latents_dir,
                    "normal_images_dir": normal_images_dir,
                    "normal_images_latents_dir": normal_images_latents_dir,
                    "seg_mask_dir": seg_mask_dir,
                    "seg_mask_latents_dir": seg_mask_latents_dir,
                    "seg_mask_images_dir": seg_mask_images_dir,
                    "seg_mask_images_latents_dir": seg_mask_images_latents_dir,
                    "hand_keypoints_dir": hand_keypoints_dir,
                    "hand_keypoints_latents_dir": hand_keypoints_latents_dir,
                    "hand_keypoints_images_dir": hand_keypoints_images_dir,
                    "hand_keypoints_images_latents_dir": hand_keypoints_images_latents_dir
                }
            )

        except Exception:
            print("-------------------------")
            print(f"An exception occurred while processing data: {rank=}, {world_size=}, {step=}")
            traceback.print_exc()
            print("-------------------------")

    # 5. Complete distributed processing
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    output_queue.put(None)
    save_thread.shutdown(wait=True)
    save_future.result()

    # 6. Combine results from each rank
    if rank == 0:
        print(
            f"Completed preprocessing latents and embeddings. Temporary files from all ranks saved to `{tmp_dir.as_posix()}`"
        )
        exit()
        # Move files from each rank to common directory
        for subfolder, extension in [
            ("images", "png"),
            ("image_latents", "pt"),
            ("videos", "txt"),
            ("video_latents", "pt"),
            ("prompts", "txt"),
            ("prompt_embeds", "pt"),
            ("tracking", "mp4"),
            ("tracking_latents", "pt"),
            ("tracking_images", "png"),
            ("tracking_images_latents", "pt"),
            ("depth", "mp4"),
            ("depth_latents", "pt"),
            ("depth_images", "png"),
            ("depth_images_latents", "pt"),
            ("normal", "mp4"),
            ("normal_latents", "pt"),
            ("normal_images", "png"),
            ("normal_images_latents", "pt"),
            ("seg_mask", "mp4"),
            ("seg_mask_latents", "pt"),
            ("seg_mask_images", "png"),
            ("seg_mask_images_latents", "pt"),
            ("hand_keypoints", "mp4"),
            ("hand_keypoints_latents", "pt"),
            ("hand_keypoints_images", "png"),
            ("hand_keypoints_images_latents", "pt"),
        ]:
            tmp_subfolder = tmp_dir.joinpath(subfolder)
            combined_subfolder = output_dir.joinpath(subfolder)
            combined_subfolder.mkdir(parents=True, exist_ok=True)
            pattern = f"*.{extension}"

            for file in tmp_subfolder.rglob(pattern):
                file.replace(combined_subfolder / file.name)

        # Remove temporary directories
        def rmdir_recursive(dir: pathlib.Path) -> None:
            for child in dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rmdir_recursive(child)
            dir.rmdir()

        rmdir_recursive(tmp_dir)

        # Combine prompts and videos into individual text files and single jsonl
        prompts_folder = output_dir.joinpath("prompts")
        prompts = []
        stems = []

        for filename in prompts_folder.rglob("*.txt"):
            with open(filename, "r") as file:
                prompts.append(file.read().strip())
            stems.append(filename.stem)

        prompts_txt = output_dir.joinpath("prompts.txt")
        videos_txt = output_dir.joinpath("videos.txt")
        data_jsonl = output_dir.joinpath("data.jsonl")

        with open(prompts_txt, "w") as file:
            for prompt in prompts:
                file.write(f"{prompt}\n")

        with open(videos_txt, "w") as file:
            for stem in stems:
                file.write(f"videos/{stem}.mp4\n")

        with open(data_jsonl, "w") as file:
            for prompt, stem in zip(prompts, stems):
                video_metadata_txt = output_dir.joinpath(f"videos/{stem}.txt")
                with open(video_metadata_txt, "r", encoding="utf-8") as metadata_file:
                    metadata = json.loads(metadata_file.read())

                data = {
                    "prompt": prompt,
                    "prompt_embed": f"prompt_embeds/{stem}.pt",
                    "image": f"images/{stem}.png",
                    "image_latent": f"image_latents/{stem}.pt",
                    "video": f"videos/{stem}.mp4",
                    "video_latent": f"video_latents/{stem}.pt",
                    "tracking": f"tracking/{stem}.mp4",
                    "tracking_latent": f"tracking_latents/{stem}.pt",
                    "depth": f"depth/{stem}.mp4",
                    "depth_latent": f"depth_latents/{stem}.pt",
                    "normal": f"normal/{stem}.mp4",
                    "normal_latent": f"normal_latents/{stem}.pt",
                    "seg_mask": f"seg_mask/{stem}.mp4",
                    "seg_mask_latent": f"seg_mask_latents/{stem}.pt",
                    "hand_keypoints": f"hand_keypoints/{stem}.mp4",
                    "hand_keypoints_latent": f"hand_keypoints_latents/{stem}.pt",
                    "metadata": metadata,
                }
                file.write(json.dumps(data) + "\n")

        print(f"Completed preprocessing. All files saved to `{output_dir.as_posix()}`")


if __name__ == "__main__":
    main()
