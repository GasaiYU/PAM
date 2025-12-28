import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import json
from torchvision.transforms import InterpolationMode
import cv2
from torchvision.transforms.functional import resize
import PIL.Image as Image
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset.hoi_utils import showHandJoints, CLASS_PROTOCAL, convert_gray_to_color


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]

def split_video_path(video_path):
    video_path = os.path.dirname(video_path)
    return video_path.split("dexycb_videos")[-1][1:]

class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        # if len(self.video_paths) != len(self.prompts):
        #     raise ValueError(
        #         f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
        #     )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
            frames = video_reader.get_batch(indices)
            frames = frames[: self.max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_valid=kwargs.pop("is_valid", False)

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

class VideoDatasetWithResizingTracking(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column = kwargs.pop("tracking_column", None)
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path, tracking_path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path, tracking_path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            tracking_reader = decord.VideoReader(uri=tracking_path.as_posix())
            tracking_frames = tracking_reader.get_batch(frame_indices[:nearest_frame_bucket])
            tracking_frames = tracking_frames[:nearest_frame_bucket].float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)

            return image, frames, tracking_frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        tracking_path = self.data_root.joinpath(self.tracking_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )
        if not tracking_path.exists() or not tracking_path.is_file():
            raise ValueError(
                "Expected `--tracking_column` to be path to a file in `--data_root` containing line-separated tracking information."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(tracking_path, "r", encoding="utf-8") as file:
            tracking_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        self.tracking_paths = tracking_paths
        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        tracking_paths = df[self.tracking_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]
        tracking_paths = [self.data_root.joinpath(line.strip()) for line in tracking_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found at least one path that is not a valid file."
            )

        self.tracking_paths = tracking_paths
        return prompts, video_paths
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index], self.tracking_paths[index])

            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "tracking_map": tracking_map,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, tracking_map, _ = self._preprocess_video(self.video_paths[index], self.tracking_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "tracking_map": tracking_map,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }
    
    def _load_preprocessed_latents_and_embeds(self, path: Path, tracking_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        tracking_map_path = path.parent.parent.joinpath("tracking_map")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or not tracking_map_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains folders named `video_latents`, `prompt_embeds`, and `tracking_map`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        tracking_map_filepath = tracking_map_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file() or not tracking_map_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            tracking_map_filepath = tracking_map_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} or {tracking_map_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        tracking_map = torch.load(tracking_map_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, tracking_map, embeds

class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []


class HOIVideoDatasetResizing(VideoDataset):
    def visualize_samples(self, num=5, save_dir="debug_samples"):
        """保存部分样本的图片、掩码、法线、深度等到本地，便于人工检查"""
        import os
        import torchvision.utils as vutils
        os.makedirs(save_dir, exist_ok=True)
        for i in range(num):
            try:
                sample = self[i]
                # 保存图片
                if "video" in sample:
                    img = sample["video"]
                    if hasattr(img, 'shape') and img.shape[0] > 0:
                        vutils.save_image(img[0].float()/255.0, os.path.join(save_dir, f"sample_{i}_video.png"))
                if "normal_map" in sample:
                    normal = sample["normal_map"]
                    vutils.save_image((normal[0]+1)/2, os.path.join(save_dir, f"sample_{i}_normal.png"))
                if "depth_map" in sample:
                    depth = sample["depth_map"]
                    vutils.save_image(depth[0:1], os.path.join(save_dir, f"sample_{i}_depth.png"))
                if "seg_mask" in sample:
                    mask = sample["seg_mask"]
                    if hasattr(mask, 'shape') and mask.shape[0] > 0:
                        vutils.save_image(mask[0:1].float()/mask.max(), os.path.join(save_dir, f"sample_{i}_mask.png"))
            except Exception as e:
                print(f"[visualize_samples] Error at idx {i}: {e}")

    def _check_sample(self, sample, idx=None):
        """对单个样本做 shape、nan、inf 检查，并打印关键信息"""
        import torch
        info = f"[SampleCheck idx={idx}] "
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                info += f"{k}: shape={tuple(v.shape)} min={v.min().item() if hasattr(v,'min') else '?'} max={v.max().item() if hasattr(v,'max') else '?'}; "
                if torch.is_tensor(v):
                    if torch.isnan(v).any():
                        info += f"{k} has NaN; "
                    if torch.isinf(v).any():
                        info += f"{k} has Inf; "
        print(info)

    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column = kwargs.pop("tracking_column", None)
        self.normal_column = kwargs.pop("normal_column", None)
        self.depth_column = kwargs.pop("depth_column", None)
        self.label_column = kwargs.pop("label_column", None)
        self.device = 'cuda'
        self.is_valid = kwargs.pop("is_valid", False)
        self.used_condition = kwargs.pop("used_condition", set({'hand','tracking','normal','depth','mask'}))
        filepath = kwargs.pop("filter_file",None)
        self.filter_file = None
        self.color_transform = kwargs.pop("color_transform",{})
        if filepath:
            self.filter_file={}
            with open(filepath, 'r') as f:
                data = [json.loads(line.strip()) for line in f]
                for d in data:
                    self.filter_file[d['image_path']] = d['loss']
            
        self.loss_threshold = kwargs.pop("loss_threshold", 100.0)

        # For bg and human seg mask
        self.bg_json_path = kwargs.pop("bg_json_path", None)
        self.human_label_root = kwargs.pop("human_label_root", None)

        super().__init__(*args, **kwargs)
        
    def _init_llava(self):
        self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def load_bg_and_human(self, seg_mask, video_path, frame_index, human_label_num=120):
        # load human label (It usually detect banana as arm..)
        if self.human_label_root is not None and ((seg_mask == 10).sum() == 0 or (seg_mask == 255).sum() > 0):
            human_label_path = os.path.join(self.human_label_root, 
                                            split_video_path(video_path), 
                                            f"mask_{frame_index:06d}.npy")
            if os.path.exists(human_label_path):
                human_label = np.load(human_label_path)
                # For detection error
                if (seg_mask == 255).sum() == 0 or ((seg_mask == 255) & human_label).sum() > 0:
                    # seg_mask [B, H, W]
                    seg_mask[human_label] = human_label_num
                    # Ignore disalignment
                    seg_mask[seg_mask == 255] = 0
                else:
                    seg_mask[seg_mask == 255] = human_label_num
        else:
            seg_mask[seg_mask == 255] = human_label_num
            
        # load bg label
        if self.bg_json_path is not None:
            with open(self.bg_json_path, "r") as f:
                for line in f.readlines():
                    bg_label = json.loads(line)
                    if split_video_path(video_path) == split_video_path(bg_label['video_path']):
                        seg_mask[seg_mask == 0] = bg_label['bg_label']

        return seg_mask

    def _load_preprocessed_latents_and_embeds(
                        self, 
                        path : Path,
                        tracking_path : Path,
                        normal_path : Path,
                        depth_path : Path,
                        label_path : Path
                    ):
        
        raise NotImplementedError

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def generate_description(self, image):
        """使用 LLaVA v1.6 Mistral-7B 模型生成图像描述"""
        try:
            self.llava_processor
        except AttributeError:
            self._init_llava()
        image = image.convert("RGB")
        prompt = "<image>\nDescribe this image in detail."
        inputs = self.llava_processor(
            text=prompt,
            images=image,
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
        description = description.replace("Describe this image in detail.", "").strip()
        return description
    
    def _preprocess_video(self, 
                          path : Path,
                          tracking_path : Path,
                          normal_path : Path,
                          depth_path : Path,
                          label_path : Path,
                          ) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path, tracking_path, normal_path, depth_path, label_path)
        else:
            # Read rgb video
            video_reader = decord.VideoReader(uri=path.as_posix())

            # 计算采样帧的索引
            frame_interval = 8  # 每 8 帧采一张
            frame_indices = list(range(0, 49, frame_interval))

            # 确保不超过最大帧数
            if self.max_num_frames is not None:
                frame_indices = frame_indices[:self.max_num_frames]

            # 读取采样帧
            frames = video_reader.get_batch(frame_indices)
            frames = frames.float()  # 转换为浮点数
            frames = frames.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0) 
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None
            # Read tracking videos
            if (tracking_path is not None and random.random() < 0.8 or self.is_valid) and "tracking" in self.used_condition:
                tracking_reader = decord.VideoReader(uri=tracking_path.as_posix())
                tracking_frames = tracking_reader.get_batch(frame_indices)
                tracking_frames = tracking_frames.float()
                tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
                tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
                tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)
            else:
                tracking_frames = -torch.ones_like(frames)
            
            # Read normal videos
            if (normal_path is not None or self.is_valid) and "normal" in self.used_condition:
                normal_reader = decord.VideoReader(uri=normal_path.as_posix())
                normal_frames = normal_reader.get_batch(frame_indices)
                normal_frames = normal_frames.float()
                normal_frames = normal_frames.permute(0, 3, 1, 2).contiguous()
                normal_frames_resized = torch.stack([resize(normal_frame, nearest_res) for normal_frame in normal_frames], dim=0)
                normal_frames = torch.stack([self.video_transforms(normal_frame) for normal_frame in normal_frames_resized], dim=0)
            else:
                normal_frames = -torch.ones_like(frames)
            
            # Read depth videos
            if (depth_path is not None and random.random() < 0.8 or self.is_valid) and "depth" in self.used_condition:
                depth_reader = decord.VideoReader(uri=depth_path.as_posix())
                depth_frames = depth_reader.get_batch(frame_indices)
                depth_frames = depth_frames.float()
                depth_frames = depth_frames.permute(0, 3, 1, 2).contiguous()
                depth_frames_resized = torch.stack([resize(depth_frame, nearest_res) for depth_frame in depth_frames], dim=0)
                depth_frames = torch.stack([self.video_transforms(depth_frame) for depth_frame in depth_frames_resized], dim=0)
            else:
                depth_frames = -torch.ones_like(frames)

            # Read hand pose videos and segmentation videos
            masks = []
            hand_keypoints = []
            keypoints_2d = []
            colored_masks = []
            if label_path is not None:
                label_files = []
                for file in os.listdir(label_path.as_posix()):
                    if file.startswith("label"):
                        label_files.append(file)
                label_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
                
                for index in frame_indices:
                    file = label_files[index]
                    label = np.load(label_path.joinpath(file))
                    masks.append(label["seg"].copy())
                    if self.human_label_root is not None:
                        colored_masks.append(convert_gray_to_color(self.load_bg_and_human(label["seg"], path.as_posix(), index)))  
                    else:
                        seg_wo_object = label["seg"].copy()
                        # seg_wo_object[seg_wo_object != 255] = 0
                        colored_masks.append(convert_gray_to_color(seg_wo_object))
                    # import cv2
                    # cv2.imwrite('test.png', colored_masks[-1])
                    # breakpoint()
                    hand_keypoints.append(showHandJoints(np.zeros([480,640,3], dtype=np.uint8), label["joint_2d"][0]))
                    keypoints_2d.append(torch.from_numpy(label["joint_2d"][0]))

                # Get colored semantic masks
                colored_masks = torch.from_numpy(np.stack(colored_masks, axis=0)).float()
                colored_masks = colored_masks.permute(0, 3, 1, 2).contiguous()
                colored_masks = torch.stack([resize(colored_mask, nearest_res, interpolation=InterpolationMode.NEAREST) for colored_mask in colored_masks], dim=0)
                colored_masks = torch.stack([self.video_transforms(colored_mask) for colored_mask in colored_masks], dim=0)

                # Get Hand Keypoints masks
                hand_keypoints = torch.from_numpy(np.stack(hand_keypoints, axis=0)).float()
                hand_keypoints = hand_keypoints.permute(0, 3, 1, 2).contiguous()
                hand_keypoints = torch.stack([resize(hand_keypoint, nearest_res, interpolation=InterpolationMode.NEAREST) for hand_keypoint in hand_keypoints], dim=0)
                hand_keypoints = torch.stack([self.video_transforms(hand_keypoint) for hand_keypoint in hand_keypoints], dim=0)

                keypoints_2d = torch.stack(keypoints_2d)
                # Mask depth and normal frames
                masks = torch.from_numpy(np.stack(masks, axis=0))
                
                masks = torch.stack([resize(mask.unsqueeze(0), nearest_res, interpolation=InterpolationMode.NEAREST) for mask in masks], dim=0)

                hand_masks = (masks == 255)
                masks = masks > 0
                extended_mask=masks.repeat(1,3,1,1)
                depth_frames[~extended_mask] = -1
                normal_frames[~extended_mask] = -1
                # frames 不再加掩码，保留原始视频帧

                if (random.random() > 0.8 and not self.is_valid) or 'mask' not in self.used_condition:
                    colored_masks = -torch.ones_like(frames)
                
                if (random.random() > 0.95 and not self.is_valid) or 'hand' not in self.used_condition:
                    hand_keypoints = -torch.ones_like(frames)

            else:
                colored_masks = -torch.ones_like(frames)
                hand_keypoints = -torch.ones_like(frames)

            return {
                "image": image,
                "frames": frames,
                "normal_frames": normal_frames,
                "tracking_frames": tracking_frames,
                "depth_frames": depth_frames,
                "colored_masks": colored_masks,
                "hand_keypoints": hand_keypoints,
                "hand_mask": hand_masks,
                "front_mask":extended_mask,
                "keypoints_2d": keypoints_2d
            }
    
    def _load_dataset_from_local_path(self):
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")
        
        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        tracking_path = self.data_root.joinpath(self.tracking_column) if self.tracking_column is not None else None
        normal_path = self.data_root.joinpath(self.normal_column) if self.normal_column is not None else None
        depth_path = self.data_root.joinpath(self.depth_column) if self.depth_column is not None else None
        label_path = self.data_root.joinpath(self.label_column) if self.label_column is not None else None

        prompts, video_paths, tracking_paths, normal_paths, depth_paths, label_paths = None, None, None, None, None, None

        if not prompt_path.exists() or not prompt_path.is_file():
            # print(prompt_path)
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        else:
            with open(prompt_path, "r") as file:
                prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            ) 
        else:
            with open(video_path, "r") as file:
                video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        if tracking_path is not None and (not tracking_path.exists() or not tracking_path.is_file()):
            raise ValueError(
                "Expected `--tracking_column` to be path to a file in `--data_root` containing line-separated tracking information."
            )
        elif tracking_path is not None:
            with open(tracking_path, "r") as file:
                tracking_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        if normal_path is not None and (not normal_path.exists() or not normal_path.is_file()):
            raise ValueError(
                "Expected `--normal_column` to be path to a file in `--data_root` containing line-separated normal information."
            )
        elif normal_path is not None:
            with open(normal_path, "r") as file:
                normal_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        if depth_path is not None and (not depth_path.exists() or not depth_path.is_file()):
            raise ValueError(
                "Expected `--depth_column` to be path to a file in `--data_root` containing line-separated depth information."
            )
        elif depth_path is not None:
            with open(depth_path, "r") as file:
                # depth_paths = [self.data_root.joinpath(line.strip().replace('masked_color.mp4','depth.mp4')) for line in file.readlines() if len(line.strip()) > 0]
                depth_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        if label_path is not None and (not label_path.exists() or not label_path.is_file()):
            raise ValueError(
                "Expected `--label_column` to be path to a directory in `--data_root` containing semantic segmentation information."
            )
        elif label_path is not None:
            with open(label_path, "r") as file:
                label_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        self.tracking_paths = tracking_paths
        self.normal_paths = normal_paths
        self.depth_paths = depth_paths
        self.label_paths = label_paths

        return prompts, video_paths
    
    def _check_valid(self,index:int):
        video_index=index//7
        frame_index=(index%7)*8
        base_path=self.video_paths[video_index]
        path_query=os.path.join(os.path.dirname(base_path),"color_"+"0"*(6-len(str(frame_index)))+str(frame_index)+'.jpg')
        if path_query in self.filter_file:
            if self.filter_file[path_query] > self.loss_threshold:
                return False
        return True
    
    def _load_dataset_from_csv(self):
        raise NotImplementedError
    
    def _get_description(self, index):
        return str(self.video_paths[index]).replace("video.mp4", "descriptions.json")
    
    def __len__(self)->int:
        return len(self.video_paths*7)
    
    def __getitem__(self, index):
        if isinstance(index, list):
            return index
        if self.load_tensors:
            raise NotImplementedError
        else:
            if self.filter_file != None:
                if not self._check_valid(index):
                    return self.__getitem__(index+1)
            index_video = index // 7
            index_frame = (index % 7) * 8
            preprocess_dict = self._preprocess_video(
                self.video_paths[index_video],
                self.tracking_paths[index_video] if self.tracking_paths is not None else None,
                self.normal_paths[index_video] if self.normal_paths is not None else None,
                self.depth_paths[index_video] if self.depth_paths is not None else None,
                self.label_paths[index_video] if self.label_paths is not None else None,
            )
            if os.path.exists(self._get_description(index_video)):
                with open(self._get_description(index_video), "r") as file:
                    descriptions = json.load(file)

            sample = {
                "prompt": descriptions[str(index_frame)],
                # "prompt": "A hand is trying to grasp something",
                "video": preprocess_dict["frames"][index_frame//8],
                "tracking_map": preprocess_dict["tracking_frames"][index_frame//8],
                "depth_map": preprocess_dict["depth_frames"][index_frame//8],
                "normal_map": preprocess_dict["normal_frames"][index_frame//8],
                "seg_mask": preprocess_dict["colored_masks"][index_frame//8],
                "hand_keypoints": preprocess_dict["hand_keypoints"][index_frame//8],
                "hand_mask":preprocess_dict["hand_mask"][index_frame//8],
                "video_metadata": {
                    "height": preprocess_dict["frames"][index_frame//8].shape[1],
                    "width": preprocess_dict["frames"][index_frame//8].shape[2],
                },
                "mask" : preprocess_dict["front_mask"][index_frame//8],
                "keypoints_2d": preprocess_dict["keypoints_2d"][index_frame//8],
            }
            # 数据验证
            # self._check_sample(sample, idx=index)
            return sample

    def __post_init__(self):
        """实例化后自动采样若干条数据做可视化和验证"""
        print("[HOIVideoDatasetResizing] 自动采样部分数据做可视化和验证...")
        self.visualize_samples(num=3, save_dir="debug_samples")
        for i in range(3):
            try:
                sample = self[i]
                self._check_sample(sample, idx=i)
            except Exception as e:
                print(f"[__post_init__] Error at idx {i}: {e}")
                
def get_train_set(data_root, train_caption_column, train_video_column, train_depth_column, train_label_column):
    return HOIVideoDatasetResizing(
        data_root=Path(data_root),
        caption_column=Path(train_caption_column),
        video_column=Path(train_video_column),
        depth_column=Path(train_depth_column),
        label_column=Path(train_label_column),
        image_to_video=True,
        load_tensors=False,
        max_num_frames=72,
        frame_buckets=[8],
        height_buckets=[480],
        width_buckets=[640],
        is_valid=False,
        used_condition={'depth','mask','hand'},
    )
    
def get_valid_set(data_root, valid_caption_column, valid_video_column, valid_depth_column, valid_label_column):
    return HOIVideoDatasetResizing(
        data_root=Path(data_root),
        caption_column=Path(valid_caption_column),
        video_column=Path(valid_video_column),
        depth_column=Path(valid_depth_column),
        label_column=Path(valid_label_column),
        image_to_video=True,
        load_tensors=False,
        max_num_frames=72,
        frame_buckets=[8],
        height_buckets=[480],
        width_buckets=[640],
        is_valid=True,
        used_condition={'depth','mask','hand'},
    )
