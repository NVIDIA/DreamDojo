"""Align canonical robot actions with fixed LAM frame pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ACTION_DIM = 14
ACTION_DIMENSION_NAMES = (
    "left_translation_x",
    "left_translation_y",
    "left_translation_z",
    "left_rotation_x",
    "left_rotation_y",
    "left_rotation_z",
    "left_gripper",
    "right_translation_x",
    "right_translation_y",
    "right_translation_z",
    "right_rotation_x",
    "right_rotation_y",
    "right_rotation_z",
    "right_gripper",
)
ACTION_GROUPS = {
    "left_translation": (0, 1, 2),
    "left_rotation": (3, 4, 5),
    "left_gripper": (6,),
    "right_translation": (7, 8, 9),
    "right_rotation": (10, 11, 12),
    "right_gripper": (13,),
}


def _matrix(value: Any, *, dtype: Any) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 2 or result.shape[1] != ACTION_DIM:
        return None
    return result


def aggregate_robot_action(
    robot: Mapping[str, Any] | None,
    camera: Mapping[str, Any],
    frame_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Integrate canonical robot deltas between two camera frames.

    Action timestamps are interval midpoints.  For frames at ``t0`` and ``t1``,
    actions in ``(t0, t1]`` are summed.  A target dimension is valid only when
    every contributing step for that dimension is finite and marked valid.
    Smoothed global deltas are preferred because they form the most consistent
    cross-robot target; progressively weaker canonical fallbacks are explicit
    in the returned representation label.
    """

    empty_target = np.zeros(ACTION_DIM, dtype=np.float32)
    empty_valid = np.zeros(ACTION_DIM, dtype=bool)
    if robot is None or len(frame_indices) != 2:
        return empty_target, empty_valid, 0, "unavailable"

    frame_timestamps = np.asarray(camera.get("frame_timestamps_ns") or [], dtype=np.int64)
    first, second = (int(frame_indices[0]), int(frame_indices[1]))
    if (
        first < 0
        or second < 0
        or first >= len(frame_timestamps)
        or second >= len(frame_timestamps)
        or second <= first
    ):
        return empty_target, empty_valid, 0, "unavailable"
    start_ns = int(frame_timestamps[first])
    end_ns = int(frame_timestamps[second])
    if end_ns <= start_ns:
        return empty_target, empty_valid, 0, "unavailable"

    timestamps = np.asarray(robot.get("action_timestamps_ns") or [], dtype=np.int64)
    candidates = (
        ("smoothed_actions_global", "action_valid_global"),
        ("actions_global", "action_valid_global"),
        ("smoothed_actions_local", "action_valid_local"),
        ("actions_local", "action_valid_local"),
    )
    for action_field, validity_field in candidates:
        actions = _matrix(robot.get(action_field), dtype=np.float64)
        validity = _matrix(robot.get(validity_field), dtype=bool)
        if actions is None or validity is None:
            continue
        count = min(len(timestamps), len(actions), len(validity))
        if count < 1:
            continue
        selected = (timestamps[:count] > start_ns) & (timestamps[:count] <= end_ns)
        if not selected.any():
            continue
        selected_actions = actions[:count][selected]
        selected_validity = validity[:count][selected] & np.isfinite(selected_actions)
        dimension_validity = selected_validity.all(axis=0)
        integrated = np.where(selected_validity, selected_actions, 0.0).sum(axis=0)
        integrated[~dimension_validity] = 0.0
        return (
            integrated.astype(np.float32),
            dimension_validity.astype(bool),
            int(selected.sum()),
            action_field,
        )

    return empty_target, empty_valid, 0, "unavailable"
