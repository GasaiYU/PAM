import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.hoi_utils import showHandJoints, CLASS_PROTOCAL, convert_gray_to_color, mask_conditions


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


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

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

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
    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column = kwargs.pop("tracking_column", None)
        self.normal_column = kwargs.pop("normal_column", None)
        self.depth_column = kwargs.pop("depth_column", None)
        # For rgb
        self.label_column = kwargs.pop("label_column", None)
        # For latent
        self.seg_mask = kwargs.pop("seg_mask_column", False)
        self.hand_keypoints = kwargs.pop("hand_keypoints_column", False)
        self.image_column = kwargs.pop("image_column", None)
        self.tracking_image_column = kwargs.pop("tracking_image_column", None)
        self.normal_image_column = kwargs.pop("normal_image_column", None)
        self.depth_image_column = kwargs.pop("depth_image_column", None)
        self.seg_mask_image_column = kwargs.pop("seg_mask_image_column", None)
        self.hand_keypoints_image_column = kwargs.pop("hand_keypoints_image_column", None)
        # For random mask
        self.random_mask = kwargs.pop("random_mask", False)
        # For initial frame num
        self.initial_frames_num = kwargs.pop("initial_frames_num", 1)
        # For used conditions
        self.used_conditions = kwargs.pop("used_conditions", None)
        
        super().__init__(*args, **kwargs)
    
    def _load_preprocessed_latents_and_embeds(
                        self, 
                        image_latent_path: Path,
                        video_latent_path : Path,
                        tracking_latent_path : Path,
                        tracking_image_latent_path : Path,
                        normal_latent_path : Path,
                        normal_image_latent_path : Path,
                        depth_latent_path : Path,
                        depth_image_latent_path : Path,
                        seg_mask_latent_path : Path,
                        seg_mask_image_latent_path : Path,
                        hand_keypoints_latent_path : Path,
                        hand_keypoints_image_latent_path : Path,
                        prompt_embed_path : Path,
                    ):
        if image_latent_path is not None:
            image_latents = torch.load(image_latent_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Image latent path is not provided.")

        if video_latent_path is not None:
            video_latents = torch.load(video_latent_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Video latent path is not provided.")
        
        if (tracking_latent_path is not None and tracking_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) and \
                    (self.used_conditions is None or "tracking" in self.used_conditions):
            tracking_latents = torch.load(tracking_latent_path, map_location="cpu", weights_only=True)
            tracking_image_latents = torch.load(tracking_image_latent_path, map_location="cpu", weights_only=True)
        else:
            tracking_latents = torch.zeros_like(video_latents)
            tracking_image_latents = torch.zeros_like(image_latents)
        
        if (normal_latent_path is not None and normal_image_latent_path is not None) and \
                (random.random() < 0.7 or not self.random_mask) \
                    and (self.used_conditions is None or "normal" in self.used_conditions): 
            normal_latents = torch.load(normal_latent_path, map_location="cpu", weights_only=True)
            normal_image_latents = torch.load(normal_image_latent_path, map_location="cpu", weights_only=True)
        else:
            normal_latents = torch.zeros_like(video_latents)
            normal_image_latents = torch.zeros_like(image_latents)
        
        if (depth_latent_path is not None and depth_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) and \
                    (self.used_conditions is None or "depth" in self.used_conditions):
            depth_latents = torch.load(depth_latent_path, map_location="cpu", weights_only=True)
            depth_image_latents = torch.load(depth_image_latent_path, map_location="cpu", weights_only=True)
        else:
            depth_latents = torch.zeros_like(video_latents)
            depth_image_latents = torch.zeros_like(image_latents)

        if (seg_mask_latent_path is not None and seg_mask_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) \
                    and (self.used_conditions is None or "seg_mask" in self.used_conditions):
            seg_mask_latents = torch.load(seg_mask_latent_path, map_location="cpu", weights_only=True)
            seg_mask_image_latents = torch.load(seg_mask_image_latent_path, map_location="cpu", weights_only=True)
        else:
            seg_mask_latents = torch.zeros_like(video_latents)
            seg_mask_image_latents = torch.zeros_like(image_latents)
        
        if (hand_keypoints_latent_path is not None and hand_keypoints_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) \
                    and (self.used_conditions is None or "hand_keypoints" in self.used_conditions):
            hand_keypoints_latents = torch.load(hand_keypoints_latent_path, map_location="cpu", weights_only=True)
            hand_keypoints_image_latents = torch.load(hand_keypoints_image_latent_path, map_location="cpu", weights_only=True)
        else:
            hand_keypoints_latents = torch.zeros_like(video_latents)
            hand_keypoints_image_latents = torch.zeros_like(image_latents)
        
        if prompt_embed_path is not None:
            prompt_embeds = torch.load(prompt_embed_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Prompt embed path is not provided.")

        return {
            "prompt": prompt_embeds,
            "image": image_latents,
            "video": video_latents,
            "tracking_map": tracking_latents,
            "tracking_image": tracking_image_latents,
            "depth_map": depth_latents,
            "depth_image": depth_image_latents,
            "normal_map": normal_latents,
            "normal_image": normal_image_latents,
            "seg_mask": seg_mask_latents,
            "seg_mask_image": seg_mask_image_latents,
            "hand_keypoints": hand_keypoints_latents,
            "hand_keypoints_image": hand_keypoints_image_latents,
        }


    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _preprocess_video(self, 
                          path : Path,
                          tracking_path : Path,
                          normal_path : Path,
                          depth_path : Path,
                          label_path : Path) -> torch.Tensor:
       
        # Read rgb video
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        nearest_frame_bucket = min(
            self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
        )

        frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:nearest_frame_bucket].float()
        frames = frames.permute(0, 3, 1, 2).contiguous() # (T, C, H, W)

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0) 
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        # Read tracking videos
        if tracking_path is not None and (random.random() < 0.8 or not self.random_mask):
            tracking_reader = decord.VideoReader(uri=tracking_path.as_posix())
            tracking_frames = tracking_reader.get_batch(frame_indices[:nearest_frame_bucket])
            tracking_frames = tracking_frames[:nearest_frame_bucket].float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)
        else:
            tracking_frames = torch.zeros_like(frames)
        
        # Read normal videos
        if normal_path is not None and (random.random() < 0.7 or not self.random_mask) and False:
            normal_reader = decord.VideoReader(uri=normal_path.as_posix())
            normal_frames = normal_reader.get_batch(frame_indices[:nearest_frame_bucket])
            normal_frames = normal_frames[:nearest_frame_bucket].float()
            normal_frames = normal_frames.permute(0, 3, 1, 2).contiguous()
            normal_frames_resized = torch.stack([resize(normal_frame, nearest_res) for normal_frame in normal_frames], dim=0)
            normal_frames = torch.stack([self.video_transforms(normal_frame) for normal_frame in normal_frames_resized], dim=0)
        else:
            normal_frames = torch.zeros_like(frames)
        
        # Read depth videos
        if depth_path is not None and (random.random() < 0.8 or not self.random_mask):
            depth_reader = decord.VideoReader(uri=depth_path.as_posix())
            depth_frames = depth_reader.get_batch(frame_indices[:nearest_frame_bucket])
            depth_frames = depth_frames[:nearest_frame_bucket].float()
            depth_frames = depth_frames.permute(0, 3, 1, 2).contiguous()
            depth_frames_resized = torch.stack([resize(depth_frame, nearest_res) for depth_frame in depth_frames], dim=0)
            depth_frames = torch.stack([self.video_transforms(depth_frame) for depth_frame in depth_frames_resized], dim=0)
        else:
            depth_frames = torch.zeros_like(frames)

        # Read hand pose videos and segmentation videos
        masks = []
        hand_keypoints = []
        colored_masks = []
        if label_path is not None:
            label_files = []
            for file in os.listdir(label_path.as_posix()):
                if file.startswith("label"):
                    label_files.append(file)
            label_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

            for index in frame_indices[:nearest_frame_bucket]:
                file = label_files[index]
                label = np.load(label_path.joinpath(file))
                masks.append(label["seg"])
                colored_masks.append(convert_gray_to_color(label["seg"]))  
                hand_keypoints.append(showHandJoints(np.zeros([480, 640, 3], dtype=np.uint8), label["joint_2d"][0], label["joint_3d"][0]))
            
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

            # Mask depth and normal frames
            masks = torch.from_numpy(np.stack(masks, axis=0))
            masks = torch.stack([resize(mask.unsqueeze(0), nearest_res, interpolation=InterpolationMode.NEAREST) for mask in masks], dim=0)
            masks[masks > 0] = 1
            masks = masks.repeat(1, 3, 1, 1)
            depth_frames[masks == 0] = -1.0
            # normal_frames[masks == 0] = -1.0

            if self.random_mask and random.random() > 0.8:
                colored_masks = torch.zeros_like(frames)
            
            if self.random_mask and random.random() > 0.8:
                hand_keypoints = torch.zeros_like(frames)

        else:
            colored_masks = torch.zeros_like(frames)
            hand_keypoints = torch.zeros_like(frames)

        image = frames[:self.initial_frames_num].clone() if self.image_to_video else None
        tracking_image = tracking_frames[:self.initial_frames_num].clone() if self.image_to_video else None
        normal_image = normal_frames[:self.initial_frames_num].clone() if self.image_to_video else None
        depth_image = depth_frames[:self.initial_frames_num].clone() if self.image_to_video else None
        colored_mask_image = colored_masks[:self.initial_frames_num].clone() if self.image_to_video else None
        hand_keypoints_image = hand_keypoints[:self.initial_frames_num].clone() if self.image_to_video else None
        
        return {
            "image": image,
            "frames": frames,
            "tracking_frames": tracking_frames,
            "tracking_image": tracking_image,
            "normal_frames": normal_frames,
            "normal_image": normal_image,
            "depth_frames": depth_frames,
            "depth_image": depth_image,
            "colored_masks": colored_masks,
            "colored_mask_image": colored_mask_image,
            "hand_keypoints": hand_keypoints,
            "hand_keypoints_image": hand_keypoints_image
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
        # For load latents
        seg_mask_path = self.data_root.joinpath(self.seg_mask) if self.seg_mask else None
        hand_keypoints_path = self.data_root.joinpath(self.hand_keypoints) if self.hand_keypoints else None
        image_path = self.data_root.joinpath(self.image_column) if self.image_column is not None else None
        tracking_image_path = self.data_root.joinpath(self.tracking_image_column) if self.tracking_image_column is not None else None
        depth_image_path = self.data_root.joinpath(self.depth_image_column) if self.depth_image_column is not None else None
        normal_image_path = self.data_root.joinpath(self.normal_image_column) if self.normal_image_column is not None else None
        seg_mask_image_path = self.data_root.joinpath(self.seg_mask_image_column) if self.seg_mask_image_column is not None else None
        hand_keypoints_image_path = self.data_root.joinpath(self.hand_keypoints_image_column) if self.hand_keypoints_image_column is not None else None

        if self.load_tensors:
            assert seg_mask_path and hand_keypoints_path and image_path, "Expected `--seg_mask` and `--hand_keypoints` to be set to when `--load_tensors` is set to True."

        prompts, video_paths, tracking_paths, normal_paths, depth_paths, label_paths = None, None, None, None, None, None
        seg_mask_paths, hand_keypoints_paths, image_paths = None, None, None

        if not prompt_path.exists() or not prompt_path.is_file():
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
                depth_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        if not self.load_tensors:
            if label_path is not None and (not label_path.exists() or not label_path.is_file()):
                raise ValueError(
                    "Expected `--label_column` to be path to a directory in `--data_root` containing semantic segmentation information."
                )
            elif label_path is not None:
                with open(label_path, "r") as file:
                    label_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        else:
            if seg_mask_path is not None and (not seg_mask_path.exists() or not seg_mask_path.is_file()):
                raise ValueError(
                    "Expected `--seg_mask` to be path to a directory in `--data_root` containing semantic segmentation information."
                )
            elif seg_mask_path is not None:
                with open(seg_mask_path, "r") as file:
                    seg_mask_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

            if hand_keypoints_path is not None and (not hand_keypoints_path.exists() or not hand_keypoints_path.is_file()):
                raise ValueError(
                    "Expected `--hand_keypoints` to be path to a directory in `--data_root` containing hand keypoints information."
                )
            elif hand_keypoints_path is not None:
                with open(hand_keypoints_path, "r") as file:
                    hand_keypoints_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

            if image_path is not None and (not image_path.exists() or not image_path.is_file()):
                raise ValueError(
                    "Expected `--image` to be path to a directory in `--data_root` containing image information."
                )
            elif image_path is not None:
                with open(image_path, "r") as file:
                    image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

            if tracking_image_path is not None and (not tracking_image_path.exists() or not tracking_image_path.is_file()):
                raise ValueError(
                    "Expected `--tracking_image` to be path to a directory in `--data_root` containing tracking image information."
                )
            elif tracking_image_path is not None:
                with open(tracking_image_path, "r") as file:
                    tracking_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
            if depth_image_path is not None and (not depth_image_path.exists() or not depth_image_path.is_file()):
                raise ValueError(
                    "Expected `--depth_image` to be path to a directory in `--data_root` containing depth image information."
                )
            elif depth_image_path is not None:
                with open(depth_image_path, "r") as file:
                    depth_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
            if normal_image_path is not None and (not normal_image_path.exists() or not normal_image_path.is_file()):
                raise ValueError(
                    "Expected `--normal_image` to be path to a directory in `--data_root` containing normal image information."
                )
            elif normal_image_path is not None:
                with open(normal_image_path, "r") as file:
                    normal_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
            if seg_mask_image_path is not None and (not seg_mask_image_path.exists() or not seg_mask_image_path.is_file()):
                raise ValueError(
                    "Expected `--seg_mask_image` to be path to a directory in `--data_root` containing seg mask image information."
                )
            elif seg_mask_image_path is not None:
                with open(seg_mask_image_path, "r") as file:
                    seg_mask_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
                
            if hand_keypoints_image_path is not None and (not hand_keypoints_image_path.exists() or not hand_keypoints_image_path.is_file()):
                raise ValueError(
                    "Expected `--hand_keypoints_image` to be path to a directory in `--data_root` containing hand keypoints image information."
                )
            elif hand_keypoints_image_path is not None:
                with open(hand_keypoints_image_path, "r") as file:
                    hand_keypoints_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
        self.tracking_paths = tracking_paths
        self.normal_paths = normal_paths
        self.depth_paths = depth_paths
        self.label_paths = label_paths

        # For latent
        if self.load_tensors:
            self.seg_mask_paths = seg_mask_paths
            self.hand_keypoints_paths = hand_keypoints_paths
            self.image_paths = image_paths
            self.tracking_image_paths = tracking_image_paths
            self.normal_image_paths = normal_image_paths
            self.depth_image_paths = depth_image_paths
            self.seg_mask_image_paths = seg_mask_image_paths
            self.hand_keypoints_image_paths = hand_keypoints_image_paths
        

        return prompts, video_paths
    
    def _load_dataset_from_csv(self):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, list):
            return index
        
        if self.load_tensors:
            res_dict = self._load_preprocessed_latents_and_embeds(
                self.image_paths[index],
                self.video_paths[index],
                self.tracking_paths[index] if self.tracking_paths is not None else None,
                self.tracking_image_paths[index] if self.tracking_image_paths is not None else None,
                self.normal_paths[index] if self.normal_paths is not None else None,
                self.normal_image_paths[index] if self.normal_image_paths is not None else None,
                self.depth_paths[index] if self.depth_paths is not None else None,
                self.depth_image_paths[index] if self.depth_image_paths is not None else None,
                self.seg_mask_paths[index] if self.seg_mask_paths is not None else None,
                self.seg_mask_image_paths[index] if self.seg_mask_image_paths is not None else None,
                self.hand_keypoints_paths[index] if self.hand_keypoints_paths is not None else None,
                self.hand_keypoints_image_paths[index] if self.hand_keypoints_image_paths is not None else None,
                self.prompts[index],
            )

            video_latents = res_dict["video"]
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            res_dict['video_metadata'] = {
                "num_frames": num_frames,
                "height": height,
                "width": width,
            }
            return res_dict
        
        else:
            preprocess_dict = self._preprocess_video(
                self.video_paths[index],
                self.tracking_paths[index] if self.tracking_paths is not None else None,
                self.normal_paths[index] if self.normal_paths is not None else None,
                self.depth_paths[index] if self.depth_paths is not None else None,
                self.label_paths[index] if self.label_paths is not None else None,
            )

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": preprocess_dict["image"],
                "video": preprocess_dict["frames"],
                "tracking_map": preprocess_dict["tracking_frames"],
                "tracking_image": preprocess_dict["tracking_image"],
                "depth_map": preprocess_dict["depth_frames"],
                "depth_image": preprocess_dict["depth_image"],
                "normal_map": preprocess_dict["normal_frames"],
                "normal_image": preprocess_dict["normal_image"],
                "seg_mask": preprocess_dict["colored_masks"],
                "seg_mask_image": preprocess_dict["colored_mask_image"],
                "hand_keypoints": preprocess_dict["hand_keypoints"],
                "hand_keypoints_image": preprocess_dict["hand_keypoints_image"],
                "video_metadata": {
                    "num_frames": preprocess_dict["frames"].shape[0],
                    "height": preprocess_dict["frames"].shape[2],
                    "width": preprocess_dict["frames"].shape[3],
                },
            }

class Oakink2VideoDatasetResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column = kwargs.pop("tracking_column", None)
        self.normal_column = kwargs.pop("normal_column", None)
        self.depth_column = kwargs.pop("depth_column", None)
        self.seg_mask = kwargs.pop("seg_mask_column", False)
        self.hand_keypoints = kwargs.pop("hand_keypoints_column", False)
        # For latent
        self.image_column = kwargs.pop("image_column", None)
        self.tracking_image_column = kwargs.pop("tracking_image_column", None)
        self.normal_image_column = kwargs.pop("normal_image_column", None)
        self.depth_image_column = kwargs.pop("depth_image_column", None)
        self.seg_mask_image_column = kwargs.pop("seg_mask_image_column", None)
        self.hand_keypoints_image_column = kwargs.pop("hand_keypoints_image_column", None)
        # For random mask
        self.random_mask = kwargs.pop("random_mask", False)
        # For initial frame num
        self.initial_frames_num = kwargs.pop("initial_frames_num", 1)
        # For used conditions
        self.used_conditions = kwargs.pop("used_conditions", None)
        
        super().__init__(*args, **kwargs)
    
    def _load_preprocessed_latents_and_embeds(
                        self, 
                        image_latent_path: Path,
                        video_latent_path : Path,
                        tracking_latent_path : Path,
                        tracking_image_latent_path : Path,
                        normal_latent_path : Path,
                        normal_image_latent_path : Path,
                        depth_latent_path : Path,
                        depth_image_latent_path : Path,
                        seg_mask_latent_path : Path,
                        seg_mask_image_latent_path : Path,
                        hand_keypoints_latent_path : Path,
                        hand_keypoints_image_latent_path : Path,
                        prompt_embed_path : Path,
                    ):
        if image_latent_path is not None:
            image_latents = torch.load(image_latent_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Image latent path is not provided.")

        if video_latent_path is not None:
            video_latents = torch.load(video_latent_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Video latent path is not provided.")
        
        if (tracking_latent_path is not None and tracking_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) and \
                    (self.used_conditions is None or "tracking" in self.used_conditions):
            tracking_latents = torch.load(tracking_latent_path, map_location="cpu", weights_only=True)
            tracking_image_latents = torch.load(tracking_image_latent_path, map_location="cpu", weights_only=True)
        else:
            tracking_latents = torch.zeros_like(video_latents)
            tracking_image_latents = torch.zeros_like(image_latents)
        
        if (normal_latent_path is not None and normal_image_latent_path is not None) and \
                (random.random() < 0.7 or not self.random_mask) \
                    and (self.used_conditions is None or "normal" in self.used_conditions): 
            normal_latents = torch.load(normal_latent_path, map_location="cpu", weights_only=True)
            normal_image_latents = torch.load(normal_image_latent_path, map_location="cpu", weights_only=True)
        else:
            normal_latents = torch.zeros_like(video_latents)
            normal_image_latents = torch.zeros_like(image_latents)
        
        if (depth_latent_path is not None and depth_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) and \
                    (self.used_conditions is None or "depth" in self.used_conditions):
            depth_latents = torch.load(depth_latent_path, map_location="cpu", weights_only=True)
            depth_image_latents = torch.load(depth_image_latent_path, map_location="cpu", weights_only=True)
        else:
            depth_latents = torch.zeros_like(video_latents)
            depth_image_latents = torch.zeros_like(image_latents)

        if (seg_mask_latent_path is not None and seg_mask_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) \
                    and (self.used_conditions is None or "seg_mask" in self.used_conditions):
            seg_mask_latents = torch.load(seg_mask_latent_path, map_location="cpu", weights_only=True)
            seg_mask_image_latents = torch.load(seg_mask_image_latent_path, map_location="cpu", weights_only=True)
        else:
            seg_mask_latents = torch.zeros_like(video_latents)
            seg_mask_image_latents = torch.zeros_like(image_latents)
        
        if (hand_keypoints_latent_path is not None and hand_keypoints_image_latent_path is not None) and \
                (random.random() < 0.8 or not self.random_mask) \
                    and (self.used_conditions is None or "hand_keypoints" in self.used_conditions):
            hand_keypoints_latents = torch.load(hand_keypoints_latent_path, map_location="cpu", weights_only=True)
            hand_keypoints_image_latents = torch.load(hand_keypoints_image_latent_path, map_location="cpu", weights_only=True)
        else:
            hand_keypoints_latents = torch.zeros_like(video_latents)
            hand_keypoints_image_latents = torch.zeros_like(image_latents)
        
        if prompt_embed_path is not None:
            prompt_embeds = torch.load(prompt_embed_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Prompt embed path is not provided.")

        return {
            "prompt": prompt_embeds,
            "image": image_latents,
            "video": video_latents,
            "tracking_map": tracking_latents,
            "tracking_image": tracking_image_latents,
            "depth_map": depth_latents,
            "depth_image": depth_image_latents,
            "normal_map": normal_latents,
            "normal_image": normal_image_latents,
            "seg_mask": seg_mask_latents,
            "seg_mask_image": seg_mask_image_latents,
            "hand_keypoints": hand_keypoints_latents,
            "hand_keypoints_image": hand_keypoints_image_latents,
        }


    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _preprocess_video(self, 
                          path : Path,
                          tracking_path : Path,
                          normal_path : Path,
                          depth_path : Path,
                          hand_path : Path,
                          seg_path: Path) -> torch.Tensor:
       
        # Read rgb video
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        nearest_frame_bucket = min(
            self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
        )

        frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:nearest_frame_bucket].float()
        frames = frames.permute(0, 3, 1, 2).contiguous() # (T, C, H, W)

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0) 
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        # Read tracking videos
        if tracking_path is not None and (random.random() < 0.8 or not self.random_mask):
            tracking_reader = decord.VideoReader(uri=tracking_path.as_posix())
            tracking_frames = tracking_reader.get_batch(frame_indices[:nearest_frame_bucket])
            tracking_frames = tracking_frames[:nearest_frame_bucket].float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)
        else:
            tracking_frames = torch.zeros_like(frames)
        
        # Read normal videos
        if normal_path is not None and (random.random() < 0.7 or not self.random_mask) and False:
            normal_reader = decord.VideoReader(uri=normal_path.as_posix())
            normal_frames = normal_reader.get_batch(frame_indices[:nearest_frame_bucket])
            normal_frames = normal_frames[:nearest_frame_bucket].float()
            normal_frames = normal_frames.permute(0, 3, 1, 2).contiguous()
            normal_frames_resized = torch.stack([resize(normal_frame, nearest_res) for normal_frame in normal_frames], dim=0)
            normal_frames = torch.stack([self.video_transforms(normal_frame) for normal_frame in normal_frames_resized], dim=0)
        else:
            normal_frames = torch.zeros_like(frames)
        
        # Read depth videos
        if depth_path is not None and (random.random() < 0.8 or not self.random_mask):
            depth_reader = decord.VideoReader(uri=depth_path.as_posix())
            depth_frames = depth_reader.get_batch(frame_indices[:nearest_frame_bucket])
            depth_frames = depth_frames[:nearest_frame_bucket].float()
            depth_frames = depth_frames.permute(0, 3, 1, 2).contiguous()
            depth_frames_resized = torch.stack([resize(depth_frame, nearest_res) for depth_frame in depth_frames], dim=0)
            depth_frames = torch.stack([self.video_transforms(depth_frame) for depth_frame in depth_frames_resized], dim=0)
        else:
            depth_frames = torch.zeros_like(frames)

        # Read hand pose videos and segmentation videos
        # masks = []
        # hand_keypoints = []
        # colored_masks = []
        # if label_path is not None:
        #     label_files = []
        #     for file in os.listdir(label_path.as_posix()):
        #         if file.startswith("label"):
        #             label_files.append(file)
        #     label_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

        #     for index in frame_indices[:nearest_frame_bucket]:
        #         file = label_files[index]
        #         label = np.load(label_path.joinpath(file))
        #         masks.append(label["seg"])
        #         colored_masks.append(convert_gray_to_color(label["seg"]))  
        #         hand_keypoints.append(showHandJoints(np.zeros([480, 640, 3], dtype=np.uint8), label["joint_2d"][0], label["joint_3d"][0]))
            
        #     # Get colored semantic masks
        #     colored_masks = torch.from_numpy(np.stack(colored_masks, axis=0)).float()
        #     colored_masks = colored_masks.permute(0, 3, 1, 2).contiguous()
        #     colored_masks = torch.stack([resize(colored_mask, nearest_res, interpolation=InterpolationMode.NEAREST) for colored_mask in colored_masks], dim=0)
        #     colored_masks = torch.stack([self.video_transforms(colored_mask) for colored_mask in colored_masks], dim=0)

        #     # Get Hand Keypoints masks
        #     hand_keypoints = torch.from_numpy(np.stack(hand_keypoints, axis=0)).float()
        #     hand_keypoints = hand_keypoints.permute(0, 3, 1, 2).contiguous()
        #     hand_keypoints = torch.stack([resize(hand_keypoint, nearest_res, interpolation=InterpolationMode.NEAREST) for hand_keypoint in hand_keypoints], dim=0)
        #     hand_keypoints = torch.stack([self.video_transforms(hand_keypoint) for hand_keypoint in hand_keypoints], dim=0)

        #     # Mask depth and normal frames
        #     masks = torch.from_numpy(np.stack(masks, axis=0))
        #     masks = torch.stack([resize(mask.unsqueeze(0), nearest_res, interpolation=InterpolationMode.NEAREST) for mask in masks], dim=0)
        #     masks[masks > 0] = 1
        #     masks = masks.repeat(1, 3, 1, 1)
        #     depth_frames[masks == 0] = -1.0
        #     # normal_frames[masks == 0] = -1.0

        #     if self.random_mask and random.random() > 0.8:
        #         colored_masks = torch.zeros_like(frames)
            
        #     if self.random_mask and random.random() > 0.8:
        #         hand_keypoints = torch.zeros_like(frames)

        # else:
        #     colored_masks = torch.zeros_like(frames)
        #     hand_keypoints = torch.zeros_like(frames)

        # Real hand keypoints
        if hand_path is not None and (random.random() < 0.8 or not self.random_mask):
            hand_reader = decord.VideoReader(uri=hand_path.as_posix())
            hand_keypoints = hand_reader.get_batch(frame_indices[:nearest_frame_bucket])
            hand_keypoints = hand_keypoints[:nearest_frame_bucket].float()
            hand_keypoints = hand_keypoints.permute(0, 3, 1, 2).contiguous()
            hand_keypoints_resized = torch.stack([resize(hand_keypoint, nearest_res) for hand_keypoint in hand_keypoints], dim=0)
            hand_keypoints = torch.stack([self.video_transforms(hand_keypoint) for hand_keypoint in hand_keypoints_resized], dim=0)
        else:
            hand_keypoints = torch.zeros_like(frames)
        
        # Real segmentation mask
        if seg_path is not None and (random.random() < 0.8 or not self.random_mask):
            seg_reader = decord.VideoReader(uri=seg_path.as_posix())
            colored_masks = seg_reader.get_batch(frame_indices[:nearest_frame_bucket])
            colored_masks = colored_masks[:nearest_frame_bucket].float()
            colored_masks = colored_masks.permute(0, 3, 1, 2).contiguous()
            colored_masks_resized = torch.stack([resize(colored_mask, nearest_res) for colored_mask in colored_masks], dim=0)
            colored_masks = torch.stack([self.video_transforms(colored_mask) for colored_mask in colored_masks_resized], dim=0)
        else:
            colored_masks = torch.zeros_like(frames)

        depth_frames = mask_conditions(depth_frames, colored_masks)

        image = frames[:self.initial_frames_num].clone() if self.image_to_video else None
        tracking_image = tracking_frames[:self.initial_frames_num].clone() if self.image_to_video else None
        normal_image = normal_frames[:self.initial_frames_num].clone() if self.image_to_video else None
        depth_image = depth_frames[:self.initial_frames_num].clone() if self.image_to_video else None
        colored_mask_image = colored_masks[:self.initial_frames_num].clone() if self.image_to_video else None
        hand_keypoints_image = hand_keypoints[:self.initial_frames_num].clone() if self.image_to_video else None
        
        return {
            "image": image,
            "frames": frames,
            "tracking_frames": tracking_frames,
            "tracking_image": tracking_image,
            "normal_frames": normal_frames,
            "normal_image": normal_image,
            "depth_frames": depth_frames,
            "depth_image": depth_image,
            "colored_masks": colored_masks,
            "colored_mask_image": colored_mask_image,
            "hand_keypoints": hand_keypoints,
            "hand_keypoints_image": hand_keypoints_image
        }
    
    def _load_dataset_from_local_path(self):
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")
        
        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        tracking_path = self.data_root.joinpath(self.tracking_column) if self.tracking_column is not None else None
        normal_path = self.data_root.joinpath(self.normal_column) if self.normal_column is not None else None
        depth_path = self.data_root.joinpath(self.depth_column) if self.depth_column is not None else None
        seg_mask_path = self.data_root.joinpath(self.seg_mask) if self.seg_mask else None
        hand_keypoints_path = self.data_root.joinpath(self.hand_keypoints) if self.hand_keypoints else None
        # For load latents
        image_path = self.data_root.joinpath(self.image_column) if self.image_column is not None else None
        tracking_image_path = self.data_root.joinpath(self.tracking_image_column) if self.tracking_image_column is not None else None
        depth_image_path = self.data_root.joinpath(self.depth_image_column) if self.depth_image_column is not None else None
        normal_image_path = self.data_root.joinpath(self.normal_image_column) if self.normal_image_column is not None else None
        seg_mask_image_path = self.data_root.joinpath(self.seg_mask_image_column) if self.seg_mask_image_column is not None else None
        hand_keypoints_image_path = self.data_root.joinpath(self.hand_keypoints_image_column) if self.hand_keypoints_image_column is not None else None

        if self.load_tensors:
            assert seg_mask_path and hand_keypoints_path and image_path, "Expected `--seg_mask` and `--hand_keypoints` to be set to when `--load_tensors` is set to True."

        prompts, video_paths, tracking_paths, normal_paths, depth_paths = None, None, None, None, None
        seg_mask_paths, hand_keypoints_paths, image_paths = None, None, None

        if not prompt_path.exists() or not prompt_path.is_file():
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
                depth_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        
        if seg_mask_path is not None and (not seg_mask_path.exists() or not seg_mask_path.is_file()):
            raise ValueError(
                "Expected `--seg_mask` to be path to a directory in `--data_root` containing semantic segmentation information."
            )
        elif seg_mask_path is not None:
            with open(seg_mask_path, "r") as file:
                seg_mask_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if hand_keypoints_path is not None and (not hand_keypoints_path.exists() or not hand_keypoints_path.is_file()):
            raise ValueError(
                "Expected `--hand_keypoints` to be path to a directory in `--data_root` containing hand keypoints information."
            )
        elif hand_keypoints_path is not None:
            with open(hand_keypoints_path, "r") as file:
                hand_keypoints_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if self.load_tensors:
            if image_path is not None and (not image_path.exists() or not image_path.is_file()):
                raise ValueError(
                    "Expected `--image` to be path to a directory in `--data_root` containing image information."
                )
            elif image_path is not None:
                with open(image_path, "r") as file:
                    image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

            if tracking_image_path is not None and (not tracking_image_path.exists() or not tracking_image_path.is_file()):
                raise ValueError(
                    "Expected `--tracking_image` to be path to a directory in `--data_root` containing tracking image information."
                )
            elif tracking_image_path is not None:
                with open(tracking_image_path, "r") as file:
                    tracking_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
            if depth_image_path is not None and (not depth_image_path.exists() or not depth_image_path.is_file()):
                raise ValueError(
                    "Expected `--depth_image` to be path to a directory in `--data_root` containing depth image information."
                )
            elif depth_image_path is not None:
                with open(depth_image_path, "r") as file:
                    depth_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
            if normal_image_path is not None and (not normal_image_path.exists() or not normal_image_path.is_file()):
                raise ValueError(
                    "Expected `--normal_image` to be path to a directory in `--data_root` containing normal image information."
                )
            elif normal_image_path is not None:
                with open(normal_image_path, "r") as file:
                    normal_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
            if seg_mask_image_path is not None and (not seg_mask_image_path.exists() or not seg_mask_image_path.is_file()):
                raise ValueError(
                    "Expected `--seg_mask_image` to be path to a directory in `--data_root` containing seg mask image information."
                )
            elif seg_mask_image_path is not None:
                with open(seg_mask_image_path, "r") as file:
                    seg_mask_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
                
            if hand_keypoints_image_path is not None and (not hand_keypoints_image_path.exists() or not hand_keypoints_image_path.is_file()):
                raise ValueError(
                    "Expected `--hand_keypoints_image` to be path to a directory in `--data_root` containing hand keypoints image information."
                )
            elif hand_keypoints_image_path is not None:
                with open(hand_keypoints_image_path, "r") as file:
                    hand_keypoints_image_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
            
        self.tracking_paths = tracking_paths
        self.normal_paths = normal_paths
        self.depth_paths = depth_paths
        self.seg_mask_paths = seg_mask_paths
        self.hand_keypoints_paths = hand_keypoints_paths

        # For latent
        if self.load_tensors:
            self.image_paths = image_paths
            self.tracking_image_paths = tracking_image_paths
            self.normal_image_paths = normal_image_paths
            self.depth_image_paths = depth_image_paths
            self.seg_mask_image_paths = seg_mask_image_paths
            self.hand_keypoints_image_paths = hand_keypoints_image_paths
        

        return prompts, video_paths
    
    def _load_dataset_from_csv(self):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, list):
            return index
        
        if self.load_tensors:
            res_dict = self._load_preprocessed_latents_and_embeds(
                self.image_paths[index],
                self.video_paths[index],
                self.tracking_paths[index] if self.tracking_paths is not None else None,
                self.tracking_image_paths[index] if self.tracking_image_paths is not None else None,
                self.normal_paths[index] if self.normal_paths is not None else None,
                self.normal_image_paths[index] if self.normal_image_paths is not None else None,
                self.depth_paths[index] if self.depth_paths is not None else None,
                self.depth_image_paths[index] if self.depth_image_paths is not None else None,
                self.seg_mask_paths[index] if self.seg_mask_paths is not None else None,
                self.seg_mask_image_paths[index] if self.seg_mask_image_paths is not None else None,
                self.hand_keypoints_paths[index] if self.hand_keypoints_paths is not None else None,
                self.hand_keypoints_image_paths[index] if self.hand_keypoints_image_paths is not None else None,
                self.prompts[index],
            )

            video_latents = res_dict["video"]
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            res_dict['video_metadata'] = {
                "num_frames": num_frames,
                "height": height,
                "width": width,
            }
            return res_dict
        
        else:
            preprocess_dict = self._preprocess_video(
                self.video_paths[index],
                self.tracking_paths[index] if self.tracking_paths is not None else None,
                self.normal_paths[index] if self.normal_paths is not None else None,
                self.depth_paths[index] if self.depth_paths is not None else None,
                self.hand_keypoints_paths[index] if self.hand_keypoints_paths is not None else None,
                self.seg_mask_paths[index] if self.seg_mask_paths is not None else None
            )

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": preprocess_dict["image"],
                "video": preprocess_dict["frames"],
                "tracking_map": preprocess_dict["tracking_frames"],
                "tracking_image": preprocess_dict["tracking_image"],
                "depth_map": preprocess_dict["depth_frames"],
                "depth_image": preprocess_dict["depth_image"],
                "normal_map": preprocess_dict["normal_frames"],
                "normal_image": preprocess_dict["normal_image"],
                "seg_mask": preprocess_dict["colored_masks"],
                "seg_mask_image": preprocess_dict["colored_mask_image"],
                "hand_keypoints": preprocess_dict["hand_keypoints"],
                "hand_keypoints_image": preprocess_dict["hand_keypoints_image"],
                "video_metadata": {
                    "num_frames": preprocess_dict["frames"].shape[0],
                    "height": preprocess_dict["frames"].shape[2],
                    "width": preprocess_dict["frames"].shape[3],
                },
            }

        
if __name__ == "__main__":
    oakini2_dataset = Oakink2VideoDatasetResizing(
        data_root=Path("."),
        caption_column=Path("/share/project/cwm/mingju.gao/video-diff/workspace/HOI-DiffusionAsShader/OakInk-v2_data/filelist/prompts.txt"),
        video_column=Path("/share/project/cwm/mingju.gao/video-diff/workspace/HOI-DiffusionAsShader/OakInk-v2_data/filelist/videos.txt"),
        tracking_column=Path("/share/project/cwm/mingju.gao/video-diff/workspace/HOI-DiffusionAsShader/OakInk-v2_data/filelist/depths.txt"),
        normal_column=Path("/share/project/cwm/mingju.gao/video-diff/workspace/HOI-DiffusionAsShader/OakInk-v2_data/filelist/depths.txt"),
        depth_column=Path("/share/project/cwm/mingju.gao/video-diff/workspace/HOI-DiffusionAsShader/OakInk-v2_data/filelist/depths.txt"),
        seg_mask_column=Path("/share/project/cwm/mingju.gao/video-diff/workspace/HOI-DiffusionAsShader/OakInk-v2_data/filelist/seg_masks.txt"),
        hand_keypoints_column=Path("/share/project/cwm/mingju.gao/video-diff/workspace/HOI-DiffusionAsShader/OakInk-v2_data/filelist/hand_keypoints.txt"),
        image_to_video=True,
        load_tensors=False,
        max_num_frames=49,
        frame_buckets=[49],
        height_buckets=[480],
        width_buckets=[720],
        used_conditions=["depth", "seg_mask", "hand_keypoints"]
    )

    a = oakini2_dataset[0]

    rgb_video = a['video']
    tracking_video = a['tracking_map']
    depth_video = a['depth_map']
    normal_video = a['normal_map']
    seg_mask = a['seg_mask']
    hand_keypoints = a['hand_keypoints']

    rgb_video = ((rgb_video.permute(0, 2, 3, 1).numpy() + 1.0)* 127.5).astype(np.uint8)
    tracking_video = ((tracking_video.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    depth_video = ((depth_video.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    normal_video = ((normal_video.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    seg_mask = ((seg_mask.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    hand_keypoints = ((hand_keypoints.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)

    import cv2
    alpha_depth = cv2.addWeighted(depth_video, 0.5, rgb_video, 0.5, gamma=0)
    alpha_seg_mask = cv2.addWeighted(seg_mask, 0.5, rgb_video, 0.5, gamma=0)
    alpha_hand= cv2.addWeighted(hand_keypoints, 0.5, rgb_video, 0.5, gamma=0)
    cat_video1 = np.concatenate([rgb_video, tracking_video, alpha_depth], axis=2)
    cat_video2 = np.concatenate([normal_video, alpha_seg_mask, alpha_hand], axis=2)
    cat_video = np.concatenate([cat_video1, cat_video2], axis=1)

    import cv2
    video_writer = cv2.VideoWriter("temp/test_1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (cat_video.shape[2], cat_video.shape[1]))
    for frame in cat_video:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
    # hoi_dataset = HOIVideoDatasetResizing(
    #     data_root=Path("."),
    #     caption_column=Path("data/dexycb_filelist/training/training_prompts.txt"),
    #     video_column=Path("data/dexycb_filelist/training/training_videos.txt"),
    #     tracking_column=Path("data/dexycb_filelist/training/training_trackings.txt"),
    #     normal_column=Path("data/dexycb_filelist/training/training_normals.txt"),
    #     depth_column=Path("data/dexycb_filelist/training/training_depths.txt"),
    #     label_column=Path("data/dexycb_filelist/training/training_labels.txt"),
    #     image_to_video=True,
    #     load_tensors=False,
    #     max_num_frames=49,
    #     frame_buckets=[49],
    #     height_buckets=[480],
    #     width_buckets=[720],
    # )

    # random.seed(42)
    # a = hoi_dataset[0]

    # rgb_video = a['video']
    # tracking_video = a['tracking_map']
    # depth_video = a['depth_map']
    # normal_video = a['normal_map']
    # seg_mask = a['seg_mask']
    # hand_keypoints = a['hand_keypoints']

    # rgb_video = ((rgb_video.permute(0, 2, 3, 1).numpy() + 1.0)* 127.5).astype(np.uint8)
    # tracking_video = ((tracking_video.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    # depth_video = ((depth_video.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    # normal_video = ((normal_video.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    # seg_mask = ((seg_mask.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    # hand_keypoints = ((hand_keypoints.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).astype(np.uint8)  
    
    # import cv2
    # rgb_video_writer = cv2.VideoWriter("rgb_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (rgb_video.shape[2], rgb_video.shape[1]))
    # for frame in rgb_video:
    #     rgb_video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # rgb_video_writer.release()
    # # Save hand keypoints
    # video_writer = cv2.VideoWriter("hand_keypoints.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (hand_keypoints.shape[2], hand_keypoints.shape[1]))
    # for frame in hand_keypoints:
    #     video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # video_writer.release()
    
    # # Save seg mask
    # video_writer = cv2.VideoWriter("seg_mask.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (seg_mask.shape[2], seg_mask.shape[1]))
    # for frame in seg_mask:
    #     video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # video_writer.release()

    # cat_video1 = np.concatenate([rgb_video, tracking_video, depth_video], axis=2)
    # cat_video2 = np.concatenate([normal_video, seg_mask, hand_keypoints], axis=2)
    # cat_video = np.concatenate([cat_video1, cat_video2], axis=1)

    # import cv2
    # video_writer = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (cat_video.shape[2], cat_video.shape[1]))
    # for frame in cat_video:
    #     video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # video_writer.release()
