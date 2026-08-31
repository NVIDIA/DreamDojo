import math
import random
from random import choices, randint
from typing import Any, Callable, Dict, List, Mapping, Sequence
from pathlib import Path

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2 ** 13)
import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info
from tqdm import tqdm

from lam.manifest import (
    VideoManifestEntry,
    choose_entry,
    frame_stride_for_gap,
    group_entries_by_source,
    load_video_manifests,
    normalized_source_weights,
    validate_time_gaps,
)
from lam.lance_dataset import AcceptedLanceVideoDataset


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


def filter_video_files(file_names: List, xdof: bool = False) -> List:
    if xdof:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
            and "320_240" in str(f).lower()
        ]
    else:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
        ]


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 8,
        num_workers: int = 32,
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        val_batch_size: int = None,
        worker_init_fn: Callable = None,
        collate_fn: Callable = None,
        train_sampler: Callable = None,
        test_sampler: Callable = None,
        val_sampler: Callable = None,
        multiprocessing_context: str | None = None,
        persistent_workers: bool = False,
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.persistent_workers = persistent_workers

    def _multiprocessing_context(self) -> str | None:
        if self.num_workers == 0:
            return None
        return self.multiprocessing_context

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=self._multiprocessing_context(),
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=self._multiprocessing_context(),
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=self._multiprocessing_context(),
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )


class VideoDataset(Dataset):
    def __init__(
        self,
        subset_path: str,
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        output_format: str = "t h w c",
        color_aug: bool = True
    ) -> None:
        super(VideoDataset, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.color_aug = color_aug

        # Get all the file path based on the split path
        mp4_list = list(Path(subset_path).rglob("*.mp4"))
        if len(mp4_list) > 0:
            if "xdof" in subset_path:
                self.file_names = filter_video_files(mp4_list, xdof=True)
            else:
                self.file_names = filter_video_files(mp4_list)
        else:
            raise ValueError(f"No video files found in {subset_path}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0
                )
                return self.build_data_dict(video)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
        self,
        video_path: str,
        num_frames: int,
        start_frame: int = None,
        frame_skip: int = 1
    ) -> Tensor:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_skip = randint(1, 4)
        num_frames = num_frames * frame_skip

        start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - num_frames))
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Frame was successfully read, parse it
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                # Reach the end of video, deal with padding and return
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                else:
                    raise ValueError(f"Invalid padding type: {self.padding}")
                break
        cap.release()

        video = torch.stack(frames[::frame_skip]) / 255.0

        target_ratio = 640 / 480
        if video.shape[2] / video.shape[1] > target_ratio:
            target_height = video.shape[1]
            target_width = int(video.shape[1] * target_ratio)
        elif video.shape[2] / video.shape[1] < target_ratio:
            target_height = int(video.shape[2] / target_ratio)
            target_width = video.shape[2]
        else:
            target_height = video.shape[1]
            target_width = video.shape[2]
        h_crop = (video.shape[1] - target_height) // 2
        w_crop = (video.shape[2] - target_width) // 2
        video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        video = rearrange(video, f"c t h w -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            # Brightness jitter
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)

        data_dict = {
            "videos": video
        }
        return data_dict


class OriginalVideoDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list,
        split: str = "train",
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        output_format: str = "t h w c",
        color_aug: bool = True
    ) -> None:
        super(OriginalVideoDataset, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.color_aug = color_aug

        self.file_names = []
        for subset_path in dataset_paths:
            mp4_list = list(Path(subset_path).rglob("*.mp4"))
            if len(mp4_list) > 0:
                if "xdof" in subset_path:
                    self.file_names.extend(filter_video_files(mp4_list, xdof=True))
                else:
                    self.file_names.extend(filter_video_files(mp4_list))
            else:
                raise ValueError(f"No video files found in {subset_path}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0
                )
                return self.build_data_dict(video)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
        self,
        video_path: str,
        num_frames: int,
        start_frame: int = None,
        frame_skip: int = 1
    ) -> Tensor:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_skip = randint(1, 4)
        num_frames = num_frames * frame_skip

        start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - num_frames))

        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Frame was successfully read, parse it
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                # Reach the end of video, deal with padding and return
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                else:
                    raise ValueError(f"Invalid padding type: {self.padding}")
                break
        cap.release()

        video = torch.stack(frames[::frame_skip]) / 255.0

        target_ratio = 640 / 480
        if video.shape[2] / video.shape[1] > target_ratio:
            target_height = video.shape[1]
            target_width = int(video.shape[1] * target_ratio)
        elif video.shape[2] / video.shape[1] < target_ratio:
            target_height = int(video.shape[2] / target_ratio)
            target_width = video.shape[2]
        else:
            target_height = video.shape[1]
            target_width = video.shape[2]
        h_crop = (video.shape[1] - target_height) // 2
        w_crop = (video.shape[2] - target_width) // 2
        video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        video = rearrange(video, f"c t h w -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            # Brightness jitter
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)

        data_dict = {
            "videos": video
        }
        return data_dict


class MultiSourceSamplerDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list,
        split: str = "train",
        samples_per_epoch: int = 1000000,
        sampling_strategy: str = "sample",
        color_aug: bool = True,
        **kwargs
    ) -> None:
        self.samples_per_epoch = samples_per_epoch

        self.subsets = []
        for subset_path in tqdm(dataset_paths, desc="Loading subsets..."):
            self.subsets.append(VideoDataset(subset_path=subset_path, color_aug=color_aug, **kwargs))
            print("Subset:", subset_path, "Number of samples:", len(self.subsets[-1]))
        print("Number of subsets:", len(self.subsets))

        if sampling_strategy == "sample":
            # Sample uniformly from all samples
            probs = [len(d) for d in self.subsets]
        elif sampling_strategy == "dataset":
            # Sample uniformly from all datasets
            probs = [1 for _ in self.subsets]
        elif sampling_strategy == "log":
            # Generate probabilities according to the scale of each dataset
            probs = [math.log(len(d)) if len(d) else 0 for d in self.subsets]
        elif sampling_strategy == "pi":
            # Generate probabilities according to the scale of each dataset
            probs = [len(d) ** 0.43 for d in self.subsets]
        # elif sampling_strategy == "manual":
        #     probs = [1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        else:
            raise ValueError(f"Unavailable sampling strategy: {sampling_strategy}")
        total_prob = sum(probs)
        assert total_prob > 0, "No sample is available"
        self.sample_probs = [x / total_prob for x in probs]
        print("Sample probabilities:", self.sample_probs)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
        index (int): Index (ignored since we sample randomly).

        Returns:
        TensorDict: Dict containing all the data blocks.
        """

        # Randomly select a subset based on weights
        subset = choices(self.subsets, self.sample_probs)[0]

        # Sample a valid sample with a random index
        sample_idx = randint(0, len(subset) - 1)
        sample_item = subset[sample_idx]
        return sample_item


class ManifestVideoDataset(Dataset):
    """Sample LAM clips from an explicit, FPS-aware JSONL manifest."""

    def __init__(
        self,
        manifest_paths: Sequence[str],
        padding: str = "repeat",
        randomize: bool = True,
        num_frames: int = 2,
        output_format: str = "t h w c",
        samples_per_epoch: int = 1000000,
        source_weights: Mapping[str, float] = None,
        camera_allowlist: Sequence[str] = None,
        time_gaps_s: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
        color_aug: bool = True,
        sample_randomly: bool = True,
        max_decode_attempts: int = 10,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.samples_per_epoch = samples_per_epoch
        self.color_aug = color_aug
        self.sample_randomly = sample_randomly
        self.max_decode_attempts = max_decode_attempts
        self.time_gaps_s = validate_time_gaps(time_gaps_s)
        self.entries = load_video_manifests(manifest_paths, camera_allowlist=camera_allowlist)
        self.grouped_entries = group_entries_by_source(self.entries)
        self.source_names, self.source_probabilities = normalized_source_weights(
            self.grouped_entries, source_weights
        )

        print("Manifest sources:", dict(zip(self.source_names, self.source_probabilities)))
        print("Manifest cameras:", sorted({entry.camera for entry in self.entries}))
        print("Manifest videos:", len(self.entries))

    def __len__(self) -> int:
        return self.samples_per_epoch if self.sample_randomly else len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        last_error = None
        for _ in range(self.max_decode_attempts):
            if self.sample_randomly:
                entry = choose_entry(
                    self.grouped_entries,
                    self.source_names,
                    self.source_probabilities,
                    random,
                )
                gap_s = random.choice(self.time_gaps_s)
            else:
                entry = self.entries[idx % len(self.entries)]
                gap_s = self.time_gaps_s[0]
            try:
                video = self.load_video_slice(entry, gap_s)
                return self.build_data_dict(video, entry, gap_s)
            except Exception as exc:
                last_error = exc
                if not self.sample_randomly:
                    break
        raise RuntimeError(
            f"Unable to decode a valid LAM clip after {self.max_decode_attempts} attempts"
        ) from last_error

    def load_video_slice(self, entry: VideoManifestEntry, gap_s: float) -> Tensor:
        cap = cv.VideoCapture(str(entry.video_path))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_skip = frame_stride_for_gap(entry.fps, gap_s)
        span = 1 + (self.num_frames - 1) * frame_skip
        first_frame = int(math.ceil(entry.start_time_s * entry.fps))
        last_frame_exclusive = total_frames
        if entry.end_time_s is not None:
            last_frame_exclusive = min(last_frame_exclusive, int(math.floor(entry.end_time_s * entry.fps)))
        max_start = last_frame_exclusive - span
        if max_start < first_frame:
            cap.release()
            raise ValueError(
                f"Video interval is too short for {self.num_frames} frames at a {gap_s:.3f}s gap: "
                f"{entry.video_path}"
            )

        start_frame = random.randint(first_frame, max_start) if self.randomize else first_frame
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for offset in range(span):
            ret, frame = cap.read()
            if ret and offset % frame_skip == 0:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame))
            elif not ret:
                break
        cap.release()

        if len(frames) < self.num_frames:
            if not frames or self.padding == "none":
                raise ValueError(f"Could not decode enough frames from {entry.video_path}")
            missing = self.num_frames - len(frames)
            if self.padding == "repeat":
                frames.extend([frames[-1]] * missing)
            elif self.padding == "zero":
                frames.extend([torch.zeros_like(frames[-1])] * missing)
            elif self.padding == "random":
                frames.extend([torch.rand_like(frames[-1])] * missing)
            else:
                raise ValueError(f"Invalid padding type: {self.padding}")

        video = torch.stack(frames[:self.num_frames]) / 255.0
        target_ratio = 640 / 480
        if video.shape[2] / video.shape[1] > target_ratio:
            target_height = video.shape[1]
            target_width = int(video.shape[1] * target_ratio)
        elif video.shape[2] / video.shape[1] < target_ratio:
            target_height = int(video.shape[2] / target_ratio)
            target_width = video.shape[2]
        else:
            target_height = video.shape[1]
            target_width = video.shape[2]
        h_crop = (video.shape[1] - target_height) // 2
        w_crop = (video.shape[2] - target_width) // 2
        video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        return rearrange(video, f"c t h w -> {self.output_format}")

    def build_data_dict(
        self, video: Tensor, entry: VideoManifestEntry, gap_s: float
    ) -> Dict:
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return {
            "videos": video,
            "source": entry.source,
            "camera": entry.camera,
            "time_gap_s": gap_s,
        }


class LightningVideoDataset(LightningDataset):
    def __init__(
        self,
        dataset_paths: list = None,
        manifest_paths: list = None,
        lance_dataset_path: str = None,
        lance_dataset_version: int = None,
        source_weights: Dict[str, float] = None,
        camera_allowlist: list = None,
        time_gaps_s: list = None,
        episode_split: str = "all",
        eval_fraction: float = 0.01,
        split_seed: str = "dreamdojo-lam-eval-v1",
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        output_format: str = "t h w c",
        samples_per_epoch: int = 1000000,
        sampling_strategy: str = "sample",
        **kwargs
    ) -> None:
        super(LightningVideoDataset, self).__init__(**kwargs)
        self.dataset_paths = dataset_paths
        self.manifest_paths = manifest_paths
        self.lance_dataset_path = lance_dataset_path
        self.lance_dataset_version = lance_dataset_version
        self.source_weights = source_weights
        self.camera_allowlist = camera_allowlist
        self.time_gaps_s = time_gaps_s or [0.1, 0.2, 0.3, 0.4]
        self.episode_split = episode_split
        self.eval_fraction = eval_fraction
        self.split_seed = split_seed
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.samples_per_epoch = samples_per_epoch
        self.sampling_strategy = sampling_strategy

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        configured_inputs = sum(
            bool(value)
            for value in (
                self.dataset_paths,
                self.manifest_paths,
                self.lance_dataset_path,
            )
        )
        if configured_inputs != 1:
            raise ValueError(
                "Configure exactly one of dataset_paths, manifest_paths, or "
                "lance_dataset_path"
            )
        if stage == "fit":
            if self.lance_dataset_path:
                self.train_dataset = AcceptedLanceVideoDataset(
                    dataset_path=self.lance_dataset_path,
                    dataset_version=self.lance_dataset_version,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    samples_per_epoch=self.samples_per_epoch,
                    source_weights=self.source_weights,
                    camera_allowlist=self.camera_allowlist
                    or ["left_wrist", "right_wrist", "overhead"],
                    time_gaps_s=self.time_gaps_s,
                    episode_split=self.episode_split,
                    eval_fraction=self.eval_fraction,
                    split_seed=self.split_seed,
                )
            elif self.manifest_paths:
                self.train_dataset = ManifestVideoDataset(
                    manifest_paths=self.manifest_paths,
                    padding=self.padding,
                    randomize=self.randomize,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    samples_per_epoch=self.samples_per_epoch,
                    source_weights=self.source_weights,
                    camera_allowlist=self.camera_allowlist,
                    time_gaps_s=self.time_gaps_s,
                )
            else:
                self.train_dataset = MultiSourceSamplerDataset(
                    dataset_paths=self.dataset_paths,
                    split="train",
                    padding=self.padding,
                    randomize=self.randomize,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    samples_per_epoch=self.samples_per_epoch,
                    sampling_strategy=self.sampling_strategy
                )
            # self.val_dataset = MultiSourceSamplerDataset(
            #     dataset_paths=self.dataset_paths,
            #     split="test",
            #     padding=self.padding,
            #     randomize=self.randomize,
            #     num_frames=self.num_frames,
            #     output_format=self.output_format,
            #     samples_per_epoch=self.samples_per_epoch // 1000,
            #     sampling_strategy=self.sampling_strategy,
            #     color_aug=False
            # )
        elif stage == "test":
            if self.lance_dataset_path:
                self.test_dataset = AcceptedLanceVideoDataset(
                    dataset_path=self.lance_dataset_path,
                    dataset_version=self.lance_dataset_version,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    samples_per_epoch=self.samples_per_epoch,
                    source_weights=self.source_weights,
                    camera_allowlist=self.camera_allowlist
                    or ["left_wrist", "right_wrist", "overhead"],
                    time_gaps_s=self.time_gaps_s,
                    color_aug=False,
                    episode_split=self.episode_split,
                    eval_fraction=self.eval_fraction,
                    split_seed=self.split_seed,
                )
            elif self.manifest_paths:
                self.test_dataset = ManifestVideoDataset(
                    manifest_paths=self.manifest_paths,
                    padding=self.padding,
                    randomize=False,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    samples_per_epoch=self.samples_per_epoch,
                    source_weights=self.source_weights,
                    camera_allowlist=self.camera_allowlist,
                    time_gaps_s=self.time_gaps_s,
                    color_aug=False,
                    sample_randomly=False,
                )
            else:
                self.test_dataset = OriginalVideoDataset(
                    dataset_paths=self.dataset_paths,
                    split="test",
                    padding=self.padding,
                    randomize=self.randomize,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    color_aug=False
                )
        else:
            raise ValueError(f"Invalid stage: {stage}")
