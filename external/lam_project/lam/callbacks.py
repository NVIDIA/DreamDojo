"""Training callbacks for LAM continued pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from lightning.pytorch import Callback, LightningModule, Trainer

from lam.checkpointing import milestone_path_for_step, promote_rolling_checkpoint


class RollingCheckpoint(Callback):
    """Keep one rolling checkpoint plus sparse immutable milestones.

    Checkpoint serialization happens once per rolling interval. The completed
    temporary file atomically replaces ``last.ckpt``. At configured milestone
    steps, the same checkpoint is retained with a step-numbered hard link (or an
    atomic copy when the filesystem does not support hard links).
    """

    def __init__(
        self,
        dirpath: str,
        every_n_train_steps: int = 500,
        milestone_every_n_train_steps: int = 10_000,
        preserve_steps: Sequence[int] = (500,),
        rolling_filename: str = "last.ckpt",
    ) -> None:
        super().__init__()
        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive")
        if milestone_every_n_train_steps <= 0:
            raise ValueError("milestone_every_n_train_steps must be positive")
        if milestone_every_n_train_steps % every_n_train_steps != 0:
            raise ValueError(
                "milestone_every_n_train_steps must be a multiple of "
                "every_n_train_steps"
            )
        if Path(rolling_filename).name != rolling_filename:
            raise ValueError("rolling_filename must be a filename, not a path")

        self.dirpath = Path(dirpath)
        self.every_n_train_steps = int(every_n_train_steps)
        self.milestone_every_n_train_steps = int(
            milestone_every_n_train_steps
        )
        self.preserve_steps = frozenset(int(step) for step in preserve_steps)
        if any(step <= 0 or step % self.every_n_train_steps != 0 for step in self.preserve_steps):
            raise ValueError(
                "preserve_steps must contain positive rolling-checkpoint steps"
            )
        self.rolling_filename = rolling_filename
        self._last_global_step_saved = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        step = int(trainer.global_step)
        if (
            step <= 0
            or step % self.every_n_train_steps != 0
            or step == self._last_global_step_saved
        ):
            return

        rolling_path = self.dirpath / self.rolling_filename
        temporary_path = self.dirpath / f".{self.rolling_filename}.tmp"
        milestone_path = milestone_path_for_step(
            self.dirpath,
            step,
            milestone_every_n_train_steps=self.milestone_every_n_train_steps,
            preserve_steps=self.preserve_steps,
        )

        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            temporary_path.unlink(missing_ok=True)
        trainer.strategy.barrier()

        # All ranks call save_checkpoint so distributed strategies can
        # participate; the strategy limits physical I/O to global rank zero.
        trainer.save_checkpoint(str(temporary_path), weights_only=False)
        trainer.strategy.barrier()

        retention_method = None
        if trainer.is_global_zero:
            retention_method = promote_rolling_checkpoint(
                temporary_path,
                rolling_path,
                milestone_path=milestone_path,
            )
            milestone_text = str(milestone_path) if milestone_path else "none"
            print(
                "LAM_CHECKPOINT_SAVED "
                f"step={step} rolling={rolling_path} "
                f"milestone={milestone_text} retention={retention_method or 'none'}",
                flush=True,
            )
        trainer.strategy.barrier()
        self._last_global_step_saved = step
