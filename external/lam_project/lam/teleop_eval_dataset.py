"""Fixed accepted-only Pantheon teleoperation benchmark for LAM evaluation."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import cv2 as cv
import numpy as np
import torch
from einops import rearrange
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from lam.lance_dataset import _letterbox
from lam.manifest import validate_time_gaps


TELEOP_EVAL_MANIFEST_SCHEMA = "dreamdojo-lam-fixed-pantheon-teleop-eval-v1"
TELEOP_ACCEPTED_FILTER = 'disposition == "ACCEPT"'
TELEOP_ACTION_DIMENSION_NAMES = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
)
TELEOP_ACTION_GROUPS = {
    "left_joints": (0, 1, 2, 3, 4, 5),
    "left_gripper": (6,),
    "right_joints": (7, 8, 9, 10, 11, 12),
    "right_gripper": (13,),
}


def _stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _robot_source(source_dir: str) -> str:
    for part in Path(source_dir).parts:
        if part.startswith("robot") and part.endswith("-teleop"):
            return part
    return "pantheon-teleop"


def _contract(
    dataset_root: Path,
    source_manifest: Path,
    time_gaps_s: Sequence[float],
    train_samples_per_episode: int,
    validation_samples_per_episode: int,
    split_seed: str,
) -> dict[str, Any]:
    return {
        "schema": TELEOP_EVAL_MANIFEST_SCHEMA,
        "dataset_root": str(dataset_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest.read_bytes()).hexdigest(),
        "accepted_filter": TELEOP_ACCEPTED_FILTER,
        "probe_split_field": "split",
        "metric_split": "validation",
        "time_gaps_s": list(validate_time_gaps(time_gaps_s)),
        "train_samples_per_episode": int(train_samples_per_episode),
        "validation_samples_per_episode": int(validation_samples_per_episode),
        "split_seed": split_seed,
        "action_target_contract": "issued absolute joint-command delta between frame endpoints",
        "action_dimension_names": list(TELEOP_ACTION_DIMENSION_NAMES),
        "action_groups": {
            key: list(indices) for key, indices in TELEOP_ACTION_GROUPS.items()
        },
    }


def build_fixed_teleop_eval_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    time_gaps_s: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    train_samples_per_episode: int = 4,
    validation_samples_per_episode: int = 64,
    split_seed: str = "dreamdojo-lam-teleop-eval-v1",
) -> dict[str, Any]:
    """Build a reproducible probe-train/metric-validation teleop manifest."""

    if train_samples_per_episode < 1 or validation_samples_per_episode < 1:
        raise ValueError("samples per episode must be positive")
    root = Path(dataset_root).resolve()
    source_manifest = root / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"missing Pantheon teleop manifest: {source_manifest}")
    contract = _contract(
        root,
        source_manifest,
        time_gaps_s,
        train_samples_per_episode,
        validation_samples_per_episode,
        split_seed,
    )
    target = Path(manifest_path)
    if target.exists():
        manifest = json.loads(target.read_text())
        for key, value in contract.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"existing teleop eval manifest mismatch for {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        return manifest

    source = json.loads(source_manifest.read_text())
    accepted = [
        episode
        for episode in source.get("episodes", [])
        if episode.get("disposition") == "ACCEPT"
        and episode.get("split") in {"train", "validation"}
    ]
    if not accepted:
        raise ValueError("Pantheon teleop manifest has no accepted train/validation episodes")

    samples: list[dict[str, Any]] = []
    episodes_by_split = {"train": 0, "validation": 0}
    samples_by_split = {"train": 0, "validation": 0}
    for row_id, episode in enumerate(sorted(accepted, key=lambda item: item["episode_id"])):
        split = str(episode["split"])
        episodes_by_split[split] += 1
        fps = float(episode["fps"])
        input_video_fps = float(episode["input_video_fps"])
        frames = int(episode["frames"])
        if (
            not math.isfinite(fps)
            or not math.isfinite(input_video_fps)
            or fps <= 0
            or input_video_fps <= 0
            or frames < 2
        ):
            continue
        choices = []
        for gap_s in contract["time_gaps_s"]:
            stride = max(1, int(round(fps * float(gap_s))))
            choices.extend((float(gap_s), stride, start) for start in range(frames - stride))
        if not choices:
            continue
        requested = (
            train_samples_per_episode if split == "train" else validation_samples_per_episode
        )
        ranked = sorted(
            choices,
            key=lambda choice: _stable_int(
                split_seed, episode["episode_id"], choice[0], choice[2]
            ),
        )
        for gap_s, stride, start in ranked[: min(requested, len(ranked))]:
            command_indices = [int(start), int(start + stride)]
            offset_s = float(episode.get("video_start_offset_s", 0.0))
            video_indices = [
                max(0, int(round((index / fps + offset_s) * input_video_fps)))
                for index in command_indices
            ]
            identity = {
                "episode_id": str(episode["episode_id"]),
                "command_indices": command_indices,
                "video_indices": video_indices,
                "time_gap_s": gap_s,
            }
            samples.append(
                {
                    **identity,
                    "sample_id": sha256(
                        json.dumps(identity, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:16],
                    "row_id": row_id,
                    "source": _robot_source(str(episode.get("source_dir", ""))),
                    "task": str(episode.get("task", "")),
                    "camera": "exo",
                    "split": split,
                    "video_path": str((root / episode["video"]).resolve()),
                    "commands_path": str((root / episode["commands_path"]).resolve()),
                }
            )
            samples_by_split[split] += 1

    if not samples_by_split["train"] or not samples_by_split["validation"]:
        raise ValueError("accepted teleop benchmark requires both train and validation samples")
    manifest = {
        **contract,
        "source_schema": source.get("schema"),
        "accepted_episode_count": len(accepted),
        "episodes_by_split": episodes_by_split,
        "samples_by_split": samples_by_split,
        "samples": samples,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return manifest


def _decode_file_window(path: str, indices: Sequence[int]) -> Tensor:
    capture = cv.VideoCapture(path)
    frames = []
    try:
        for frame_index in indices:
            capture.set(cv.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"failed to decode {path} frame {frame_index}")
            frames.append(torch.from_numpy(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return torch.stack(frames)


class FixedTeleopEvalDataset(Dataset):
    """Decode fixed external-video clips and aligned issued joint-command deltas."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get("schema") != TELEOP_EVAL_MANIFEST_SCHEMA:
            raise ValueError("unsupported Pantheon teleop eval manifest schema")
        if self.manifest.get("accepted_filter") != TELEOP_ACCEPTED_FILTER:
            raise ValueError("teleop eval manifest is not accepted-only")
        self.samples = list(self.manifest["samples"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int | float]:
        spec = self.samples[index]
        commands = np.load(spec["commands_path"], mmap_mode="r")
        first, second = (int(value) for value in spec["command_indices"])
        if commands.ndim != 2 or commands.shape[1] != 14 or second >= len(commands):
            raise ValueError(f"invalid commands_14d array for {spec['episode_id']}")
        endpoint = np.asarray(commands[[first, second]], dtype=np.float64)
        validity = np.isfinite(endpoint).all(axis=0)
        target = endpoint[1] - endpoint[0]
        target[~validity] = 0.0
        video = _letterbox(_decode_file_window(spec["video_path"], spec["video_indices"]))
        return {
            "videos": rearrange(video, "t c h w -> t h w c"),
            "sample_id": spec["sample_id"],
            "source": spec["source"],
            "row_id": int(spec["row_id"]),
            "episode_id": spec["episode_id"],
            "task": spec["task"],
            "camera": spec["camera"],
            "time_gap_s": float(spec["time_gap_s"]),
            "probe_split": spec["split"],
            "action_target": from_numpy(target.astype(np.float32)),
            "action_valid": from_numpy(validity.astype(bool)),
            "action_steps": second - first,
            "action_representation": self.manifest["action_target_contract"],
        }
