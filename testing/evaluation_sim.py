import argparse
from typing import Any, Dict, List, Literal, Tuple
import pandas as pd
import os
import sys

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers.utils import export_to_video, load_image, load_video

import numpy as np
import random
import cv2
from pathlib import Path
import decord
from torchvision import transforms
from torchvision.transforms.functional import resize

import PIL.Image
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.hoi_utils import *

import argparse

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--start_frame_filelist", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser

def export_comb_video(
        generate_frames: List[PIL.Image.Image],
        gt_video: torch.Tensor, 
        condition_video_list: List[torch.Tensor], 
        output_video_path: str,
        fps: int = 8
    ):
    import imageio
    import os

    # Create subdirectories
    base_dir = os.path.dirname(output_video_path)
    filename = os.path.basename(output_video_path)
    generated_dir = os.path.join(base_dir, "generated")
    group_dir = os.path.join(base_dir, "group")
    gt_dir = os.path.join(base_dir, "gt")
    last_frame_dir = os.path.join(base_dir, "last_frame")

    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(group_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(last_frame_dir, exist_ok=True)

    # Convert original video tensor to numpy array and adjust format
    gt_frames = []
    for frame in gt_video:
        frame = frame.permute(1,2,0).to(dtype=torch.float32,device="cpu").numpy()
        frame = ((frame + 1.0) * 127.5).astype(np.uint8)
        gt_frames.append(frame)
    
    condition_frames_list = []
    for condition_video in condition_video_list:
        condition_frames = []
        for frame in condition_video:
            frame = frame.permute(1,2,0).to(dtype=torch.float32,device="cpu").numpy()
            frame = ((frame + 1.0) * 127.5).astype(np.uint8)
            condition_frames.append(frame)
        condition_frames_list.append(condition_frames)
    
    # Ensure all videos have same number of frames
    num_frames = min(len(generate_frames), len(gt_frames))
    for condition_frames in condition_frames_list:
        num_frames = min(num_frames, len(condition_frames))
    
    generate_frames = generate_frames[:num_frames]
    gt_frames = gt_frames[:num_frames]
    for i in range(len(condition_frames_list)):
        condition_frames_list[i] = condition_frames_list[i][:num_frames]
    
    # Convert generated PIL images to numpy arrays
    generate_frames_np = [np.array(frame) for frame in generate_frames]

    # Save generated video separately to generated folder
    gen_video_path = os.path.join(generated_dir, f"generated_{filename}")
    with imageio.get_writer(gen_video_path, fps=fps) as writer:
        for frame in generate_frames_np:
            writer.append_data(frame)
    
    # Save last frame
    image_filename = filename.replace(".mp4", ".png")
    last_frame_path = os.path.join(last_frame_dir, f"last_frame_{image_filename}")
    imageio.imwrite(last_frame_path, generate_frames_np[-1])

    # Concatenate frames vertically and save sampled frames
    concat_frames = []
    for i in range(num_frames):
        gen_frame = generate_frames_np[i]
        gt_frame = gt_frames[i]
        
        width = min(gen_frame.shape[1], gt_frame.shape[1])
        height = gt_frame.shape[0]
        
        gen_frame = Image.fromarray(gen_frame).resize((width, height))
        gen_frame = np.array(gen_frame)
        gt_frame = Image.fromarray(gt_frame).resize((width, height))
        gt_frame = np.array(gt_frame)
        
        condition_frame_list = []
        for condition_frames in condition_frames_list:
            condition_frame = condition_frames[i]
            # alpha blend condition frame and gt_frame
            condition_frame = cv2.addWeighted(gen_frame, 0.5, condition_frame, 0.5, 0)
            condition_frame = Image.fromarray(condition_frame).resize((width, height))
            condition_frame = np.array(condition_frame)
            condition_frame_list.append(condition_frame)
        
        concat_frame = np.concatenate([gen_frame, gt_frame] + condition_frame_list, axis=1)
        concat_frames.append(concat_frame)
    
    # Save concatenated frames to group folder
    group_video_path = os.path.join(group_dir, f"group_{filename}")
    with imageio.get_writer(group_video_path, fps=fps) as writer:
        for frame in concat_frames:
            writer.append_data(frame)

    # Convert gt_frames to PIL images and save to gt folder
    gt_video_path = os.path.join(gt_dir, f"gt_{filename}")
    with imageio.get_writer(gt_video_path, fps=fps) as writer:
        for frame in gt_frames:
            writer.append_data(frame)

    return group_video_path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking, CogVideoXPipelineTracking, CogVideoXVideoToVideoPipelineTracking, CogVideoXTransformer3DModelTracking
from training.dataset import VideoDataset, VideoDatasetWithResizingTracking, HOIVideoDatasetResizing
from models.cogvidex_combination import CogVideoXImageToVideoPipelineCombination, CogVideoXTransformer3DModelCombination


video_transforms = transforms.Compose(
    [
        transforms.Lambda(VideoDataset.identity_transform),
        transforms.Lambda(VideoDataset.scale_transform),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)

def load_pipeline_and_args(model_path, transformer_path, validation_prompt,
                           guidance_scale=6.0, seed=42, num_inference_steps=50, dtype=torch.bfloat16, 
                           device='cuda'):
    transformer = CogVideoXTransformer3DModelCombination.from_pretrained(transformer_path, torch_dtype=dtype, num_tracking_blocks=12)
    pipe = CogVideoXImageToVideoPipelineCombination.from_pretrained(model_path, 
                                                                transformer=transformer,
                                                                torch_dtype=dtype)
    
    # Set model parameters
    pipe.to(device, dtype=dtype)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.transformer.gradient_checkpointing = False
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    pipeline_args = {
        "prompt": validation_prompt,
        "negative_prompt": "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        "num_inference_steps": num_inference_steps,
        "num_frames": 49,
        "use_dynamic_cfg": True,
        "guidance_scale": guidance_scale,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "height": 480,
        "width": 720
    }
    
    return pipe, pipeline_args

def generate_dexycb_mask_from_labels(dexycb_label_dir, save_path='output.mp4'):
    label_files = []
    for file in os.listdir(dexycb_label_dir):
        if file.startswith("label"):
            label_files.append(file)

    label_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    colored_masks = []
    for index in range(49):
        file = label_files[index]
        label = np.load(os.path.join(dexycb_label_dir, file))
        colored_masks.append(convert_gray_to_color(label["seg"]))  
    
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    for mask in colored_masks:
        mask = cv2.resize(mask, (640, 480))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        video_writer.write(mask)
    
    video_writer.release()

def log_validation(pipe, pipeline_args, video_path, tracking_map_path, depth_map_path, seg_mask_path, hand_keypoints_path, output_file,
                   image_path=None, frame=49, res=(480, 720), initial_frames_num=1, device='cuda', dtype=torch.bfloat16):
    vae = pipe.vae
    
    video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    frame_indices = list(range(0, len(video_reader)))
    video_frames = video_reader.get_batch(frame_indices)[:frame]
    video_frames = video_frames.permute(0, 3, 1, 2).contiguous()
    video_frames_resized = torch.stack([resize(video_frame, res) for video_frame in video_frames], dim=0)
    video_frames = torch.stack([video_transforms(video_frame) for video_frame in video_frames_resized], dim=0)
 
    tracking_map_path = Path(tracking_map_path)
    tracking_reader = decord.VideoReader(uri=tracking_map_path.as_posix())
    frame_indices = list(range(0, len(tracking_reader)))
    tracking_frames = tracking_reader.get_batch(frame_indices)[:frame]
    tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
    tracking_frames_resized = torch.stack([resize(tracking_frame, res) for tracking_frame in tracking_frames], dim=0)
    tracking_frames = torch.stack([video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)
    
    normal_frames = torch.zeros_like(tracking_frames).to(tracking_frames.device).to(tracking_frames.dtype)
    
    depth_map_path = Path(depth_map_path)
    depth_reader = decord.VideoReader(uri=depth_map_path.as_posix())
    depth_frames = depth_reader.get_batch(frame_indices[:frame])
    depth_frames = depth_frames.permute(0, 3, 1, 2).contiguous()
    depth_frames_resized = torch.stack([resize(depth_frame, res) for depth_frame in depth_frames], dim=0)
    depth_frames = torch.stack([video_transforms(depth_frame) for depth_frame in depth_frames_resized], dim=0)
    
    seg_mask_path = Path(seg_mask_path)
    seg_mask_reader = decord.VideoReader(uri=seg_mask_path.as_posix())
    seg_mask_frames = seg_mask_reader.get_batch(frame_indices[:frame])
    seg_mask_frames = seg_mask_frames.permute(0, 3, 1, 2).contiguous()
    seg_mask_frames_resized = torch.stack([resize(seg_mask_frame, res) for seg_mask_frame in seg_mask_frames], dim=0)
    seg_mask_frames = torch.stack([video_transforms(seg_mask_frame) for seg_mask_frame in seg_mask_frames_resized], dim=0)

    hand_keypoints_path = Path(hand_keypoints_path)
    hand_keypoints_reader = decord.VideoReader(uri=hand_keypoints_path.as_posix())
    hand_keypoints_frames = hand_keypoints_reader.get_batch(frame_indices[:frame])
    hand_keypoints_frames = hand_keypoints_frames.permute(0, 3, 1, 2).contiguous()
    hand_keypoints_frames_resized = torch.stack([resize(hand_keypoints_frame, res) for hand_keypoints_frame in hand_keypoints_frames], dim=0)
    hand_keypoints_frames = torch.stack([video_transforms(hand_keypoints_frame) for hand_keypoints_frame in hand_keypoints_frames_resized], dim=0)
    
    if image_path is not None:
        image = load_image(image_path)
    else:
        image = video_frames[:initial_frames_num].clone()
        
    tracking_image = tracking_frames[:initial_frames_num].clone()
    normal_image = normal_frames[:initial_frames_num].clone()
    depth_image = depth_frames[:initial_frames_num].clone()
    seg_image = seg_mask_frames[:initial_frames_num].clone()
    hand_keypoints_image = hand_keypoints_frames[:initial_frames_num].clone()
    
    if not image_path:
        image = (image + 1.0) / 2.0
    tracking_image = (tracking_image + 1.0) / 2.0
    normal_image = (normal_image + 1.0) / 2.0
    depth_image = (depth_image + 1.0) / 2.0
    seg_image = (seg_image + 1.0) / 2.0
    hand_keypoints_image = (hand_keypoints_image + 1.0) / 2.0
    
    pipeline_args["image"] = image
    pipeline_args["tracking_image"] = tracking_image
    pipeline_args["normal_image"] = normal_image
    pipeline_args["depth_image"] = depth_image
    pipeline_args["seg_image"] = seg_image
    pipeline_args["hand_keypoints_image"] = hand_keypoints_image
    
    # Preprocess the videos
    print("encoding condition maps")
    with torch.no_grad():
        video_frames = video_frames.to(device).to(dtype)
        video = video_frames.clone()
        video_frames  = video_frames.unsqueeze(0)
        video_frames = video_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]

        tracking_frames = tracking_frames.to(device).to(dtype)
        tracking_video = tracking_frames.clone()
        tracking_frames = tracking_frames.unsqueeze(0)
        tracking_frames = tracking_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
        tracking_latent_dist = vae.encode(tracking_frames).latent_dist
        tracking_maps = tracking_latent_dist.sample() * vae.config.scaling_factor
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # to [B, F, C, H, W]

        normal_frames = normal_frames.to(device).to(dtype)
        normal_video = normal_frames.clone()
        normal_frames = normal_frames.unsqueeze(0)
        normal_frames = normal_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
        normal_latent_dist = vae.encode(normal_frames).latent_dist
        normal_maps = normal_latent_dist.sample() * vae.config.scaling_factor
        normal_maps = normal_maps.permute(0, 2, 1, 3, 4)  # to [B, F, C, H, W]

        depth_frames = depth_frames.to(device).to(dtype)
        depth_video = depth_frames.clone()
        depth_frames = depth_frames.unsqueeze(0)
        depth_frames = depth_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
        depth_latent_dist = vae.encode(depth_frames).latent_dist
        depth_maps = depth_latent_dist.sample() * vae.config.scaling_factor
        depth_maps = depth_maps.permute(0, 2, 1, 3, 4)  # to [B, F, C, H, W]
        
        seg_mask_frames = seg_mask_frames.to(device).to(dtype)  
        seg_mask_video = seg_mask_frames.clone()
        seg_mask_frames = seg_mask_frames.unsqueeze(0)
        seg_mask_frames = seg_mask_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
        seg_mask_latent_dist = vae.encode(seg_mask_frames).latent_dist
        seg_masks = seg_mask_latent_dist.sample() * vae.config.scaling_factor
        seg_masks = seg_masks.permute(0, 2, 1, 3, 4)

        hand_keypoints_frames = hand_keypoints_frames.to(device).to(dtype)
        hand_keypoints_video = hand_keypoints_frames.clone()
        hand_keypoints_frames = hand_keypoints_frames.unsqueeze(0)
        hand_keypoints_frames = hand_keypoints_frames.permute(0, 2, 1, 3, 4)  # to [B, C, F, H, W]
        hand_keypoints_latent_dist = vae.encode(hand_keypoints_frames).latent_dist
        hand_keypoints = hand_keypoints_latent_dist.sample() * vae.config.scaling_factor
        hand_keypoints = hand_keypoints.permute(0, 2, 1, 3, 4)
        
    pipeline_args["tracking_maps"] = tracking_maps
    pipeline_args["depth_maps"] = depth_maps
    pipeline_args["normal_maps"] = normal_maps
    pipeline_args["seg_masks"] = seg_masks
    pipeline_args["hand_keypoints"] = hand_keypoints
    
    with torch.no_grad():
        video_generate = pipe(**pipeline_args).frames[0]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    export_comb_video(video_generate, video, [seg_mask_video, hand_keypoints_video, depth_video, tracking_video], output_file, fps=15)

def get_data_dirs(root_dir):
    data_dirs = []
    for data_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, data_dir)):
            data_dirs.append(os.path.join(root_dir, data_dir))
    
    data_dirs.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1]))

    return data_dirs

def get_sel_data_dirs(data_dirs, sel_filelist):
    with open(sel_filelist, "r") as file:
        sel_data_indexex = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
    
    sel_data_dirs = []
    sel_first_image_paths = []
    for sel_data_index in sel_data_indexex:
        sel_first_image_paths.append(sel_data_index)
        sel_data_index = sel_data_index.split('/')[-1].split('.')[0].split('_')[-2]
        sel_data_dirs.append(data_dirs[int(sel_data_index)])

    return sel_data_dirs, sel_first_image_paths
    


if __name__ == "__main__":    
    validation_prompt = """A man is in a lab trying to grasp something."""

    args = create_argparser().parse_args()
    root_dir = args.root_dir
    start_frame_filelist = args.start_frame_filelist
    transformer_path = args.transformer_path
    output_dir = args.output_dir

    data_dirs = get_data_dirs(root_dir)
    sel_data_dirs, sel_first_image_paths = get_sel_data_dirs(data_dirs, start_frame_filelist)

    pipe, pipeline_args = load_pipeline_and_args(
        model_path='THUDM/CogVideoX-5b-I2V',
        transformer_path=transformer_path,
        validation_prompt=validation_prompt,
        num_inference_steps=50
    )

    for i, (sel_data_dir, sel_first_image_path) in enumerate(zip(sel_data_dirs, sel_first_image_paths)):

        video_path = os.path.join(sel_data_dir, "rgb.mp4")
        tracking_map_path = os.path.join(sel_data_dir, "tracking.mp4")
        depth_map_path = os.path.join(sel_data_dir, "depth.mp4")
        seg_mask_path = os.path.join(sel_data_dir, "seg_mask.mp4")
        hand_keypoints_path = os.path.join(sel_data_dir, "hand_keypoints.mp4")
        
        output_file = os.path.join(output_dir, f"{os.path.basename(sel_data_dir)}.mp4")
        
        first_image = Image.open(sel_first_image_path)
        first_image.crop((0, 0, 640, 480)).save("tmp.png")
   
        if not os.path.exists(seg_mask_path):
            print(f"Segmentation mask path {seg_mask_path} does not exist. Generating from labels.")
            generate_dexycb_mask_from_labels(os.path.join(root_dir, "labels"), seg_mask_path)

        log_validation(
            pipe=pipe,
            pipeline_args=pipeline_args,
            video_path=video_path,
            tracking_map_path=tracking_map_path,
            depth_map_path=depth_map_path,
            seg_mask_path=seg_mask_path,
            hand_keypoints_path=hand_keypoints_path,
            output_file = output_file,
            image_path = "tmp.png"
        )
        pass
