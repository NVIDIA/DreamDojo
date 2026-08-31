import importlib.util
import os
from pathlib import Path
import sys

import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "lam" / "checkpointing.py"
SPEC = importlib.util.spec_from_file_location("checkpointing_under_test", MODULE_PATH)
checkpointing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpointing
SPEC.loader.exec_module(checkpointing)

milestone_path_for_step = checkpointing.milestone_path_for_step
promote_rolling_checkpoint = checkpointing.promote_rolling_checkpoint


def test_milestone_policy_preserves_step_500_and_every_10000(tmp_path: Path) -> None:
    kwargs = {
        "milestone_every_n_train_steps": 10_000,
        "preserve_steps": {500},
    }
    assert milestone_path_for_step(tmp_path, 500, **kwargs) == (
        tmp_path / "step-000000500.ckpt"
    )
    assert milestone_path_for_step(tmp_path, 1_000, **kwargs) is None
    assert milestone_path_for_step(tmp_path, 10_000, **kwargs) == (
        tmp_path / "step-000010000.ckpt"
    )


def test_rolling_checkpoint_replaces_last_but_keeps_milestone(tmp_path: Path) -> None:
    temporary = tmp_path / ".last.ckpt.tmp"
    rolling = tmp_path / "last.ckpt"
    step_500 = tmp_path / "step-000000500.ckpt"

    temporary.write_bytes(b"step 500")
    method = promote_rolling_checkpoint(
        temporary,
        rolling,
        milestone_path=step_500,
    )
    assert method in {"hardlink", "copy"}
    assert rolling.read_bytes() == b"step 500"
    assert step_500.read_bytes() == b"step 500"

    temporary.write_bytes(b"step 1000")
    assert promote_rolling_checkpoint(temporary, rolling) is None
    assert rolling.read_bytes() == b"step 1000"
    assert step_500.read_bytes() == b"step 500"
    assert set(os.listdir(tmp_path)) == {"last.ckpt", "step-000000500.ckpt"}


def test_training_config_disables_lightning_default_checkpointing() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lam_umi_accepted_continued_700m_b160.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trainer = config["trainer"]
    assert trainer["enable_checkpointing"] is False
    callback = trainer["callbacks"][0]
    assert callback["class_path"] == "lam.callbacks.RollingCheckpoint"
    assert callback["init_args"]["every_n_train_steps"] == 500
    assert callback["init_args"]["milestone_every_n_train_steps"] == 10_000
    assert callback["init_args"]["preserve_steps"] == [500]
