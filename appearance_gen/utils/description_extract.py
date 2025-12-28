import os
import json
import torch
import random
import decord
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

from threading import Thread
from multiprocessing import Process

class VideoDescriber:
    def __init__(self, device="cuda", frame_interval=8, max_frames=49):
        self.device = device
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self._init_llava()
        
        # 初始化视频处理参数
        self.height_buckets = [480]
        self.width_buckets = [640]
        self.resolutions = [(h, w) for h in self.height_buckets for w in self.width_buckets]

    def _init_llava(self):
        """初始化LLaVA模型"""
        self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def _find_nearest_resolution(self, height, width):
        """找到最近的预设分辨率"""
        return min(self.resolutions, key=lambda x: abs(x[0] - height) + abs(x[1] - width))

    def _preprocess_frame(self, frame):
        """预处理视频帧"""
        # 转换为Tensor并调整通道顺序
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # HWC -> CHW
        frame = frame / 255.0  # 归一化到[0,1]
        
        # 调整到最近的分辨率
        h, w = self._find_nearest_resolution(frame.shape[1], frame.shape[2])
        frame = resize(frame, [h, w], interpolation=InterpolationMode.BICUBIC)
        return frame

    def generate_description(self, frame_tensor):
        """生成单帧描述"""
        # 转换为PIL图像
        frame_np = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_np).convert("RGB")
        
        # 生成描述
        prompt = "<image>\nDescribe this image in detail."
        inputs = self.llava_processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llava_model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.3,
            )
        
        description = self.llava_processor.decode(outputs[0], skip_special_tokens=True)
        return description.replace("Describe this image in detail.", "").strip()

    def process_video(self, video_path):
        """处理单个视频"""
        output_path = str(video_path).replace("video.mp4", "descriptions.json")
        # if os.path.exists(output_path):
        #     return  # 跳过已处理的视频

        try:
            # 读取视频
            vr = decord.VideoReader(str(video_path))
            total_frames = len(vr)
            
            # 计算采样帧索引 (0, 8, 16,...,48)
            frame_indices = list(range(0, min(total_frames, self.max_frames), self.frame_interval))
            frames = vr.get_batch(frame_indices)
            frames=frames.asnumpy()
            print(frames.shape)
            print(f"Processing {video_path} ({len(frame_indices)} frames)")
            frames=[frames[i] for i in range(len(frame_indices))]
            # 处理并生成描述
            descriptions = {}
            for idx, frame in zip(frame_indices, frames):
                print("Processing frame",idx)
                processed_frame = self._preprocess_frame(frame)
                description = self.generate_description(processed_frame)
                descriptions[video_path] = description

            # 保存结果
            with open(output_path, "w") as f:
                json.dump(descriptions, f, indent=2)

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")

    def process_filelist(self, filelist_path, base_dir, num_workers=1):
        """处理文件列表中的所有视频"""
        with open(filelist_path, "r") as f:
            video_paths = [os.path.join(base_dir,line.strip()) for line in f if line.strip()]
        
        # 单线程处理（可根据需要扩展为多线程）
        for path in tqdm(video_paths, desc="Processing videos"):
            self.process_video(Path(path))
            

def infer_prompt(video_paths, gpu_id):
    describer = VideoDescriber(device=f"cuda:{gpu_id}")  # 使用GPU加速
    if gpu_id == 0:
        p_bar = tqdm(total=len(video_paths), desc="Processing videos")
        
    for video_path in video_paths:
        describer.process_video(video_path)
        if gpu_id == 0:
            p_bar.update(1)
        
    
if __name__ == "__main__":
    num_gpus = 1
    input_filelist = "/path/to/video_filelist.txt"
    
    video_paths = []
    with open(input_filelist, "r") as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    processes = []
    split_num = len(video_paths) // num_gpus
    for i in range(num_gpus):
        process = Process(target=infer_prompt, args=(video_paths[i*split_num:(i+1)*split_num], i))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
        
