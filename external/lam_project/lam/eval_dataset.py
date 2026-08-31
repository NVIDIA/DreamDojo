"""Deterministic, accepted-only Lance benchmark data for LAM evaluation."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from einops import rearrange
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from lam.eval_actions import (
    ACTION_DIMENSION_NAMES,
    ACTION_GROUPS,
    aggregate_robot_action,
)
from lam.lance_dataset import (
    ACCEPTED_FILTER,
    CAMERA_VALIDITY_FIELDS,
    _decode_window,
    _letterbox,
    deterministic_episode_split,
    valid_window_starts,
)
from lam.manifest import validate_time_gaps


EVAL_MANIFEST_SCHEMA = "dreamdojo-lam-fixed-lance-eval-v1"


def _stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _open_lance(dataset_path: str, dataset_version: int) -> Any:
    try:
        import lance
    except ImportError as exc:
        raise ImportError("LAM Lance evaluation requires pylance==10.0.0") from exc
    return lance.dataset(dataset_path, version=int(dataset_version))


def _manifest_contract(
    dataset_path: str,
    dataset_version: int,
    num_samples: int,
    camera_allowlist: Sequence[str],
    time_gaps_s: Sequence[float],
    eval_fraction: float,
    split_seed: str,
) -> dict[str, Any]:
    return {
        "schema": EVAL_MANIFEST_SCHEMA,
        "dataset_path": dataset_path,
        "dataset_version": int(dataset_version),
        "accepted_filter": ACCEPTED_FILTER,
        "episode_split": "eval",
        "eval_fraction": float(eval_fraction),
        "split_seed": split_seed,
        "num_frames": 2,
        "num_samples": int(num_samples),
        "camera_allowlist": list(camera_allowlist),
        "time_gaps_s": list(validate_time_gaps(time_gaps_s)),
        "source_sampling": "uniform_round_robin",
        "action_target_contract": "integrated canonical Cartesian robot delta",
        "action_dimension_names": list(ACTION_DIMENSION_NAMES),
        "action_groups": {
            key: list(indices) for key, indices in ACTION_GROUPS.items()
        },
    }


def _clip_spec(
    dataset: Any,
    source: str,
    row_id: int,
    variant: int,
    camera_allowlist: Sequence[str],
    time_gaps_s: Sequence[float],
    split_seed: str,
) -> dict[str, Any]:
    rows = dataset._take_rows(
        [row_id],
        columns=[
            "episode_id",
            "task",
            "quality",
            "cameras",
            *CAMERA_VALIDITY_FIELDS.values(),
        ],
    ).to_pylist()
    if len(rows) != 1:
        raise RuntimeError(f"Lance row-ID lookup returned {len(rows)} rows")
    row = rows[0]
    quality = row["quality"]
    if quality["capture_status"] != "accepted" or not quality["capture_valid"]:
        raise RuntimeError("accepted-only query returned an ineligible episode")

    choices: list[tuple[str, float, int, list[int]]] = []
    for camera in sorted(row["cameras"], key=lambda value: value["name"]):
        camera_name = str(camera["name"])
        if camera_name not in camera_allowlist or not camera.get("video"):
            continue
        fps = float(camera["fps"])
        if not math.isfinite(fps) or fps <= 0:
            continue
        validity = row[CAMERA_VALIDITY_FIELDS[camera_name]] or []
        frame_count = min(int(camera["num_frames"]), len(validity))
        for gap_s in time_gaps_s:
            frame_stride = max(1, int(round(fps * gap_s)))
            starts = valid_window_starts(
                [bool(value) for value in validity[:frame_count]],
                num_frames=2,
                frame_stride=frame_stride,
            )
            if starts:
                choices.append((camera_name, float(gap_s), frame_stride, starts))
    if not choices:
        raise ValueError("episode has no valid fixed LAM evaluation window")

    choice_index = _stable_int(split_seed, source, row_id, variant, "choice") % len(choices)
    camera_name, gap_s, frame_stride, starts = choices[choice_index]
    start = starts[
        _stable_int(split_seed, source, row_id, variant, "start") % len(starts)
    ]
    indices = [start, start + frame_stride]
    identity = {
        "source": source,
        "row_id": int(row_id),
        "episode_id": str(row["episode_id"]),
        "task": str(row["task"]),
        "camera": camera_name,
        "time_gap_s": gap_s,
        "frame_stride": frame_stride,
        "frame_indices": indices,
    }
    identity["sample_id"] = sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return identity


def build_fixed_eval_manifest(
    manifest_path: str | Path,
    dataset_path: str,
    dataset_version: int,
    num_samples: int = 256,
    camera_allowlist: Sequence[str] = ("left_wrist", "right_wrist", "overhead"),
    time_gaps_s: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    eval_fraction: float = 0.01,
    split_seed: str = "dreamdojo-lam-eval-v1",
) -> dict[str, Any]:
    """Create or validate a fixed, source-balanced eval manifest."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    unknown = set(camera_allowlist) - set(CAMERA_VALIDITY_FIELDS)
    if unknown:
        raise ValueError("unsupported cameras: " + ", ".join(sorted(unknown)))
    contract = _manifest_contract(
        dataset_path,
        dataset_version,
        num_samples,
        camera_allowlist,
        time_gaps_s,
        eval_fraction,
        split_seed,
    )
    target = Path(manifest_path)
    if target.exists():
        manifest = json.loads(target.read_text())
        for key, value in contract.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"existing eval manifest contract mismatch for {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        if len(manifest.get("samples", [])) != num_samples:
            raise ValueError("existing eval manifest has the wrong sample count")
        return manifest

    dataset = _open_lance(dataset_path, dataset_version)
    metadata = dataset.to_table(
        columns=["_rowid", "source_dataset"],
        filter=ACCEPTED_FILTER,
    ).to_pylist()
    grouped: dict[str, list[int]] = {}
    for row in metadata:
        grouped.setdefault(str(row["source_dataset"]), []).append(int(row["_rowid"]))
    eval_rows = deterministic_episode_split(
        grouped,
        split="eval",
        eval_fraction=eval_fraction,
        split_seed=split_seed,
    )
    if not eval_rows:
        raise ValueError("deterministic accepted-only eval split is empty")
    ranked = {
        source: sorted(
            rows,
            key=lambda row_id: _stable_int(split_seed, source, row_id, "rank"),
        )
        for source, rows in eval_rows.items()
    }
    sources = sorted(ranked)

    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(10_000, num_samples * 100)
    while len(samples) < num_samples and attempts < max_attempts:
        source = sources[attempts % len(sources)]
        source_round = attempts // len(sources)
        rows = ranked[source]
        row_id = rows[source_round % len(rows)]
        variant = source_round // len(rows)
        attempts += 1
        try:
            spec = _clip_spec(
                dataset,
                source,
                row_id,
                variant,
                camera_allowlist,
                time_gaps_s,
                split_seed,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
        if spec["sample_id"] in seen:
            continue
        seen.add(spec["sample_id"])
        samples.append(spec)
    if len(samples) != num_samples:
        raise RuntimeError(
            f"could only build {len(samples)}/{num_samples} unique eval clips"
        )

    source_episode_counts = {
        source: len(rows) for source, rows in sorted(eval_rows.items())
    }
    source_sample_counts = {
        source: sum(sample["source"] == source for sample in samples)
        for source in sources
    }
    manifest = {
        **contract,
        "accepted_episode_count": len(metadata),
        "eval_episode_count": sum(source_episode_counts.values()),
        "eval_episodes_by_source": source_episode_counts,
        "samples_by_source": source_sample_counts,
        "samples": samples,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return manifest


class FixedLanceEvalDataset(Dataset):
    """Decode the exact clip windows stored in a fixed eval manifest."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = str(manifest_path)
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get("schema") != EVAL_MANIFEST_SCHEMA:
            raise ValueError("unsupported LAM eval manifest schema")
        if self.manifest.get("accepted_filter") != ACCEPTED_FILTER:
            raise ValueError("eval manifest is not pinned to the accepted-only policy")
        self.samples = list(self.manifest["samples"])
        self._dataset: Any | None = None

    def _open_dataset(self) -> Any:
        if self._dataset is None:
            self._dataset = _open_lance(
                self.manifest["dataset_path"], self.manifest["dataset_version"]
            )
        return self._dataset

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_dataset"] = None
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int | float]:
        spec = self.samples[index]
        rows = self._open_dataset()._take_rows(
            [int(spec["row_id"])],
            columns=["episode_id", "task", "quality", "robot", "cameras"],
        ).to_pylist()
        if len(rows) != 1:
            raise RuntimeError(f"Lance row-ID lookup returned {len(rows)} rows")
        row = rows[0]
        quality = row["quality"]
        if quality["capture_status"] != "accepted" or not quality["capture_valid"]:
            raise RuntimeError("fixed eval sample is no longer accepted")
        cameras = {camera["name"]: camera for camera in row["cameras"]}
        camera = cameras[spec["camera"]]
        action_target, action_valid, action_steps, action_representation = (
            aggregate_robot_action(row.get("robot"), camera, spec["frame_indices"])
        )
        video = _letterbox(
            _decode_window(bytes(camera["video"]), spec["frame_indices"])
        )
        return {
            "videos": rearrange(video, "t c h w -> t h w c"),
            "sample_id": spec["sample_id"],
            "source": spec["source"],
            "row_id": int(spec["row_id"]),
            "episode_id": str(row["episode_id"]),
            "task": str(row["task"]),
            "camera": spec["camera"],
            "time_gap_s": float(spec["time_gap_s"]),
            "action_target": from_numpy(action_target),
            "action_valid": from_numpy(action_valid),
            "action_steps": action_steps,
            "action_representation": action_representation,
        }
