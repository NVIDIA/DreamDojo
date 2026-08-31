"""Filesystem helpers for bounded, durable LAM checkpoint retention."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Collection


def milestone_path_for_step(
    directory: Path,
    step: int,
    *,
    milestone_every_n_train_steps: int,
    preserve_steps: Collection[int],
) -> Path | None:
    """Return the immutable checkpoint path for a step, if it should persist."""
    if step in preserve_steps or step % milestone_every_n_train_steps == 0:
        return directory / f"step-{step:09d}.ckpt"
    return None


def promote_rolling_checkpoint(
    temporary_path: Path,
    rolling_path: Path,
    *,
    milestone_path: Path | None = None,
) -> str | None:
    """Atomically replace the rolling checkpoint and optionally retain its inode.

    A hard link makes milestone creation effectively free and keeps the milestone
    immutable when the rolling path is replaced later. Filesystems without hard
    link support fall back to an atomic copy.
    """
    if not temporary_path.is_file():
        raise FileNotFoundError(temporary_path)

    rolling_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_path, rolling_path)

    if milestone_path is None or milestone_path.exists():
        return None

    milestone_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(rolling_path, milestone_path)
        return "hardlink"
    except OSError:
        copy_path = milestone_path.with_name(f".{milestone_path.name}.tmp")
        copy_path.unlink(missing_ok=True)
        try:
            shutil.copy2(rolling_path, copy_path)
            os.replace(copy_path, milestone_path)
        finally:
            copy_path.unlink(missing_ok=True)
        return "copy"
