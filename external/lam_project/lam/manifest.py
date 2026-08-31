"""Validation and sampling helpers for manifest-backed LAM video data."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class VideoManifestEntry:
    """One camera stream and its valid sampling interval."""

    video_path: Path
    source: str
    camera: str
    fps: float
    weight: float = 1.0
    start_time_s: float = 0.0
    end_time_s: float | None = None


def _positive_number(value: object, field: str, location: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: {field} must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{location}: {field} must be finite and positive")
    return number


def _nonnegative_number(value: object, field: str, location: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: {field} must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{location}: {field} must be finite and nonnegative")
    return number


def load_video_manifests(
    manifest_paths: Sequence[str | Path],
    camera_allowlist: Sequence[str] | None = None,
    require_files: bool = True,
) -> list[VideoManifestEntry]:
    """Load JSONL manifests, resolving relative videos beside each manifest."""

    allowed_cameras = set(camera_allowlist) if camera_allowlist else None
    entries: list[VideoManifestEntry] = []
    for raw_manifest_path in manifest_paths:
        manifest_path = Path(raw_manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise ValueError(f"Manifest does not exist: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                location = f"{manifest_path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{location}: each line must contain a JSON object")

                missing = [field for field in ("video_path", "source", "camera", "fps") if field not in record]
                if missing:
                    raise ValueError(f"{location}: missing required fields: {', '.join(missing)}")

                source = str(record["source"]).strip()
                camera = str(record["camera"]).strip()
                if not source or not camera:
                    raise ValueError(f"{location}: source and camera must be non-empty")
                if allowed_cameras is not None and camera not in allowed_cameras:
                    continue

                video_path = Path(str(record["video_path"])).expanduser()
                if not video_path.is_absolute():
                    video_path = manifest_path.parent / video_path
                video_path = video_path.resolve()
                if require_files and not video_path.is_file():
                    raise ValueError(f"{location}: video does not exist: {video_path}")

                start_time_s = _nonnegative_number(record.get("start_time_s", 0.0), "start_time_s", location)
                end_value = record.get("end_time_s")
                end_time_s = None if end_value is None else _positive_number(end_value, "end_time_s", location)
                if end_time_s is not None and end_time_s <= start_time_s:
                    raise ValueError(f"{location}: end_time_s must be greater than start_time_s")

                entries.append(
                    VideoManifestEntry(
                        video_path=video_path,
                        source=source,
                        camera=camera,
                        fps=_positive_number(record["fps"], "fps", location),
                        weight=_positive_number(record.get("weight", 1.0), "weight", location),
                        start_time_s=start_time_s,
                        end_time_s=end_time_s,
                    )
                )

    if not entries:
        detail = " after applying camera_allowlist" if allowed_cameras is not None else ""
        raise ValueError(f"No video entries were loaded{detail}")
    return entries


def validate_time_gaps(time_gaps_s: Iterable[float]) -> tuple[float, ...]:
    gaps = tuple(_positive_number(gap, "time_gaps_s", "configuration") for gap in time_gaps_s)
    if not gaps:
        raise ValueError("configuration: time_gaps_s must contain at least one value")
    return gaps


def frame_stride_for_gap(fps: float, gap_s: float) -> int:
    """Convert a physical time gap to the nearest positive frame stride."""

    return max(1, int(round(fps * gap_s)))


def group_entries_by_source(
    entries: Sequence[VideoManifestEntry],
) -> dict[str, list[VideoManifestEntry]]:
    grouped: dict[str, list[VideoManifestEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.source, []).append(entry)
    return grouped


def normalized_source_weights(
    grouped_entries: Mapping[str, Sequence[VideoManifestEntry]],
    source_weights: Mapping[str, float] | None,
) -> tuple[list[str], list[float]]:
    """Return stable source names and normalized sampling probabilities."""

    source_names = sorted(grouped_entries)
    if source_weights is None:
        raw_weights = [sum(entry.weight for entry in grouped_entries[source]) for source in source_names]
    else:
        missing = sorted(set(source_names) - set(source_weights))
        unknown = sorted(set(source_weights) - set(source_names))
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"missing sources: {', '.join(missing)}")
            if unknown:
                parts.append(f"unknown sources: {', '.join(unknown)}")
            raise ValueError("source_weights do not match manifest sources (" + "; ".join(parts) + ")")
        raw_weights = [
            _positive_number(source_weights[source], f"source_weights.{source}", "configuration")
            for source in source_names
        ]

    total = sum(raw_weights)
    return source_names, [weight / total for weight in raw_weights]


def choose_entry(
    grouped_entries: Mapping[str, Sequence[VideoManifestEntry]],
    source_names: Sequence[str],
    source_probabilities: Sequence[float],
    rng: Random,
) -> VideoManifestEntry:
    """Choose a source first, then a weighted camera stream within it."""

    source = rng.choices(source_names, weights=source_probabilities, k=1)[0]
    candidates = grouped_entries[source]
    return rng.choices(candidates, weights=[entry.weight for entry in candidates], k=1)[0]
