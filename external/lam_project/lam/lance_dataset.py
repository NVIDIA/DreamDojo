"""Read accepted UMI/YAM camera videos from a pinned Lance snapshot."""

from __future__ import annotations

import math
import random
import tempfile
from hashlib import sha256
from collections.abc import Mapping, Sequence
from typing import Any

import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from lam.manifest import validate_time_gaps


# Deliberately not configurable: the Lance path is an accepted-only training input.
ACCEPTED_FILTER = (
    "quality.capture_status = 'accepted' AND quality.capture_valid = true"
)
CAMERA_VALIDITY_FIELDS = {
    "left_wrist": "left_wrist_cam_validity",
    "right_wrist": "right_wrist_cam_validity",
    "overhead": "overhead_cam_validity",
}


def deterministic_episode_split(
    grouped_rows: Mapping[str, Sequence[int]],
    split: str,
    eval_fraction: float = 0.01,
    split_seed: str = "dreamdojo-lam-eval-v1",
) -> dict[str, list[int]]:
    """Return a stable, per-source episode split for a pinned Lance version.

    Row IDs are stable inside the pinned snapshot.  Hash ranking instead of a
    threshold gives every source with at least two episodes an eval example,
    while keeping train and eval strictly disjoint.
    """

    if split not in {"all", "train", "eval"}:
        raise ValueError("split must be one of: all, train, eval")
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")

    selected: dict[str, list[int]] = {}
    for source in sorted(grouped_rows):
        rows = sorted({int(row_id) for row_id in grouped_rows[source]})
        if split == "all":
            selected[source] = rows
            continue

        ranked = sorted(
            rows,
            key=lambda row_id: sha256(
                f"{split_seed}\0{source}\0{row_id}".encode("utf-8")
            ).digest(),
        )
        eval_count = 0 if len(ranked) < 2 else min(
            len(ranked) - 1,
            max(1, int(round(len(ranked) * eval_fraction))),
        )
        eval_rows = set(ranked[:eval_count])
        chosen = [
            row_id
            for row_id in rows
            if (row_id in eval_rows) == (split == "eval")
        ]
        if chosen:
            selected[source] = chosen
    return selected


def valid_window_starts(
    validity: Sequence[bool], num_frames: int, frame_stride: int
) -> list[int]:
    """Return starts whose complete sampled window passes per-frame QC."""

    if num_frames < 1 or frame_stride < 1:
        raise ValueError("num_frames and frame_stride must be positive")
    span = 1 + (num_frames - 1) * frame_stride
    return [
        start
        for start in range(max(0, len(validity) - span + 1))
        if all(validity[start + offset * frame_stride] for offset in range(num_frames))
    ]


def normalized_row_source_weights(
    grouped_rows: Mapping[str, Sequence[int]],
    source_weights: Mapping[str, float] | None,
) -> tuple[list[str], list[float]]:
    """Weight sources by row count by default, or validate explicit weights."""

    source_names = sorted(grouped_rows)
    if source_weights is None:
        raw_weights = [float(len(grouped_rows[source])) for source in source_names]
    else:
        missing = sorted(set(source_names) - set(source_weights))
        unknown = sorted(set(source_weights) - set(source_names))
        if missing or unknown:
            detail = []
            if missing:
                detail.append("missing sources: " + ", ".join(missing))
            if unknown:
                detail.append("unknown sources: " + ", ".join(unknown))
            raise ValueError(
                "source_weights do not match Lance sources (" + "; ".join(detail) + ")"
            )
        raw_weights = [float(source_weights[source]) for source in source_names]
        if any(not math.isfinite(weight) or weight <= 0 for weight in raw_weights):
            raise ValueError("source_weights must be finite and positive")
    total = sum(raw_weights)
    return source_names, [weight / total for weight in raw_weights]


def _decode_window(video: bytes, indices: Sequence[int]) -> Tensor:
    """Decode a short RGB window from an embedded MP4 without writing to the volume."""

    with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
        source.write(video)
        source.flush()
        capture = cv.VideoCapture(source.name)
        frames = []
        try:
            for frame_index in indices:
                capture.set(cv.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"failed to decode embedded MP4 frame {frame_index}")
                frames.append(torch.from_numpy(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
        finally:
            capture.release()
    return torch.stack(frames)


def _letterbox(video: Tensor, height: int = 240, width: int = 320) -> Tensor:
    """Resize T,H,W,C video without cutting off wrist-workspace content."""

    video = rearrange(video.float() / 255.0, "t h w c -> t c h w")
    source_height, source_width = video.shape[-2:]
    scale = min(height / source_height, width / source_width)
    resized_height = max(1, int(round(source_height * scale)))
    resized_width = max(1, int(round(source_width * scale)))
    video = F.interpolate(
        video,
        (resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height = height - resized_height
    pad_width = width - resized_width
    return F.pad(
        video,
        (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ),
        mode="replicate",
    )


class AcceptedLanceVideoDataset(Dataset):
    """Sample LAM clips exclusively from accepted rows in a Lance dataset."""

    def __init__(
        self,
        dataset_path: str,
        dataset_version: int,
        num_frames: int = 2,
        output_format: str = "t h w c",
        samples_per_epoch: int = 1_000_000,
        source_weights: Mapping[str, float] | None = None,
        camera_allowlist: Sequence[str] = (
            "left_wrist",
            "right_wrist",
            "overhead",
        ),
        time_gaps_s: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
        color_aug: bool = True,
        max_decode_attempts: int = 10,
        episode_split: str = "all",
        eval_fraction: float = 0.01,
        split_seed: str = "dreamdojo-lam-eval-v1",
    ) -> None:
        super().__init__()
        if dataset_version is None or int(dataset_version) < 1:
            raise ValueError("a positive dataset_version is required to pin the snapshot")
        if num_frames < 1 or samples_per_epoch < 1 or max_decode_attempts < 1:
            raise ValueError(
                "num_frames, samples_per_epoch, and max_decode_attempts must be positive"
            )
        if not camera_allowlist:
            raise ValueError("camera_allowlist must contain at least one camera")
        unknown_cameras = set(camera_allowlist) - set(CAMERA_VALIDITY_FIELDS)
        if unknown_cameras:
            raise ValueError(
                "unsupported cameras: " + ", ".join(sorted(unknown_cameras))
            )

        self.dataset_path = dataset_path
        self.dataset_version = int(dataset_version)
        self.num_frames = num_frames
        self.output_format = output_format
        self.samples_per_epoch = samples_per_epoch
        self.camera_allowlist = tuple(camera_allowlist)
        self.time_gaps_s = validate_time_gaps(time_gaps_s)
        self.color_aug = color_aug
        self.max_decode_attempts = max_decode_attempts
        self.episode_split = episode_split
        self.eval_fraction = eval_fraction
        self.split_seed = split_seed
        self._dataset: Any | None = None

        dataset = self._open_dataset()
        metadata = dataset.to_table(
            columns=["_rowid", "source_dataset"],
            filter=ACCEPTED_FILTER,
        ).to_pylist()
        grouped_rows: dict[str, list[int]] = {}
        for row in metadata:
            grouped_rows.setdefault(str(row["source_dataset"]), []).append(
                int(row["_rowid"])
            )
        if not grouped_rows:
            raise ValueError("the pinned Lance snapshot has no accepted episodes")
        self.grouped_rows = deterministic_episode_split(
            grouped_rows,
            split=episode_split,
            eval_fraction=eval_fraction,
            split_seed=split_seed,
        )
        if not self.grouped_rows:
            raise ValueError(f"the {episode_split!r} episode split is empty")

        self.source_names, self.source_probabilities = normalized_row_source_weights(
            self.grouped_rows, source_weights
        )
        selected_count = sum(len(rows) for rows in self.grouped_rows.values())
        print(
            f"Accepted Lance episodes: {selected_count}/{len(metadata)} "
            f"(split={episode_split})"
        )
        print(
            "Accepted Lance sources:",
            dict(zip(self.source_names, self.source_probabilities)),
        )

    def _open_dataset(self) -> Any:
        if self._dataset is None:
            try:
                import lance
            except ImportError as exc:
                raise ImportError(
                    "Accepted Lance input requires pylance==10.0.0"
                ) from exc
            self._dataset = lance.dataset(
                self.dataset_path,
                version=self.dataset_version,
            )
        return self._dataset

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_dataset"] = None
        return state

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _: int) -> dict[str, Tensor]:
        last_error: Exception | None = None
        for _ in range(self.max_decode_attempts):
            try:
                return {"videos": self._sample_clip()}
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(
            "unable to fetch a valid accepted Lance clip after "
            f"{self.max_decode_attempts} attempts"
        ) from last_error

    def _sample_clip(self) -> Tensor:
        source = random.choices(
            self.source_names,
            weights=self.source_probabilities,
            k=1,
        )[0]
        row_id = random.choice(self.grouped_rows[source])
        payload = self._open_dataset()._take_rows(
            [row_id],
            columns=[
                "quality",
                "cameras",
                *CAMERA_VALIDITY_FIELDS.values(),
            ],
        ).to_pylist()
        if len(payload) != 1:
            raise RuntimeError(f"Lance row-ID lookup returned {len(payload)} rows")
        row = payload[0]

        quality = row["quality"]
        if quality["capture_status"] != "accepted" or not quality["capture_valid"]:
            raise RuntimeError("accepted-only query returned an ineligible episode")

        cameras = {
            camera["name"]: camera
            for camera in row["cameras"]
            if camera["name"] in self.camera_allowlist and camera.get("video")
        }
        if not cameras:
            raise ValueError("accepted episode has no allowed camera with embedded video")
        camera_name = random.choice(sorted(cameras))
        camera = cameras[camera_name]
        fps = float(camera["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("camera FPS must be finite and positive")
        frame_stride = max(1, int(round(fps * random.choice(self.time_gaps_s))))

        validity = row[CAMERA_VALIDITY_FIELDS[camera_name]] or []
        frame_count = min(int(camera["num_frames"]), len(validity))
        starts = valid_window_starts(
            [bool(value) for value in validity[:frame_count]],
            self.num_frames,
            frame_stride,
        )
        if not starts:
            raise ValueError("camera has no fully valid window at the sampled time gap")
        start = random.choice(starts)
        indices = [start + offset * frame_stride for offset in range(self.num_frames)]
        video = _letterbox(_decode_window(bytes(camera["video"]), indices))
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return rearrange(video, f"t c h w -> {self.output_format}")
