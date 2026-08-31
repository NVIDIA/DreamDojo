#!/usr/bin/env python3
"""Evaluate a DreamDojo LAM checkpoint on a fixed accepted-only benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import numpy as np
import piq
import torch
from PIL import Image, ImageDraw
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from lam.eval_actions import ACTION_DIMENSION_NAMES, ACTION_GROUPS
from lam.eval_dataset import FixedLanceEvalDataset, build_fixed_eval_manifest
from lam.eval_metrics import (
    deterministic_test_mask,
    leave_one_category_out_ridge,
    masked_ridge_probe,
    nearest_centroid_probe,
    stratified_episode_bootstrap,
)
from lam.model import LAM
from lam.modules.blocks import unpatchify
from lam.teleop_eval_dataset import (
    FixedTeleopEvalDataset,
    build_fixed_teleop_eval_manifest,
)


MODEL_ARGS = {
    "image_channels": 3,
    "lam_model_dim": 1024,
    "lam_latent_dim": 32,
    "lam_patch_size": 16,
    "lam_enc_blocks": 24,
    "lam_dec_blocks": 24,
    "lam_num_heads": 16,
    "beta": 0.000001,
}
METRIC_FIELDS = (
    "mse",
    "psnr",
    "ssim",
    "lpips",
    "zero_latent_mse",
    "zero_latent_psnr",
    "shuffled_latent_mse",
    "shuffled_latent_psnr",
)


def _checkpoint_label(path: Path, explicit: str | None) -> str:
    if explicit:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", explicit).strip("-")
    match = re.search(r"step[-_=](\d+)", path.name)
    if match:
        return f"continued-step-{int(match.group(1)):09d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[LAM, int | None]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    global_step = checkpoint.get("global_step") if isinstance(checkpoint, dict) else None
    model = LAM(**MODEL_ARGS)
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    model.eval().to(device)
    return model, int(global_step) if global_step is not None else None


def _decode_latent(
    model: LAM,
    patches: Tensor,
    latent: Tensor,
    height: int,
    width: int,
) -> Tensor:
    video_patches = model.lam.patch_up(patches[:, :-1])
    action_patches = model.lam.action_up(latent)
    reconstruction = torch.sigmoid(model.lam.decoder(video_patches + action_patches))
    return unpatchify(reconstruction, model.lam.patch_size, height, width)


def _per_sample_image_metrics(
    target: Tensor,
    prediction: Tensor,
    lpips_metric: piq.LPIPS,
) -> dict[str, np.ndarray]:
    target = target.float().clamp(0, 1)
    prediction = prediction.float().clamp(0, 1)
    mse = (target - prediction).square().flatten(1).mean(1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    ssim = piq.ssim(target, prediction, data_range=1.0, reduction="none")
    lpips = lpips_metric(target, prediction)
    return {
        "mse": mse.float().cpu().numpy(),
        "psnr": psnr.float().cpu().numpy(),
        "ssim": ssim.reshape(-1).float().cpu().numpy(),
        "lpips": lpips.reshape(-1).float().cpu().numpy(),
    }


def _variant_metrics(target: Tensor, prediction: Tensor, prefix: str) -> dict[str, np.ndarray]:
    mse = (target.clamp(0, 1) - prediction.clamp(0, 1)).square().flatten(1).mean(1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {
        f"{prefix}_mse": mse.float().cpu().numpy(),
        f"{prefix}_psnr": psnr.float().cpu().numpy(),
    }


def _flow_target(video: Tensor) -> list[float]:
    frames = (video.clamp(0, 1).numpy() * 255).astype(np.uint8)
    first = cv.resize(frames[0], (160, 120), interpolation=cv.INTER_AREA)
    second = cv.resize(frames[1], (160, 120), interpolation=cv.INTER_AREA)
    first_gray = cv.cvtColor(first, cv.COLOR_RGB2GRAY)
    second_gray = cv.cvtColor(second, cv.COLOR_RGB2GRAY)
    flow = cv.calcOpticalFlowFarneback(
        first_gray,
        second_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    magnitude = np.linalg.norm(flow, axis=-1)
    return [
        float(flow[..., 0].mean()),
        float(flow[..., 1].mean()),
        float(magnitude.mean()),
        float(np.quantile(magnitude, 0.9)),
    ]


def _split_mask(sample_ids: Iterable[str]) -> np.ndarray:
    return np.asarray(
        [int(sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0 for value in sample_ids],
        dtype=bool,
    )


def _ridge_probe(
    latents: np.ndarray,
    targets: np.ndarray,
    test_mask: np.ndarray,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    train_mask = ~test_mask
    if train_mask.sum() <= latents.shape[1] or test_mask.sum() < 4:
        return {"available": False, "reason": "insufficient deterministic probe split"}
    x_train, x_test = latents[train_mask], latents[test_mask]
    y_train, y_test = targets[train_mask], targets[test_mask]
    x_mean, x_std = x_train.mean(0), x_train.std(0).clip(min=1e-6)
    y_mean, y_std = y_train.mean(0), y_train.std(0).clip(min=1e-6)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train_norm = (y_train - y_mean) / y_std
    x_train = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)
    regularizer = np.eye(x_train.shape[1]) * ridge
    regularizer[-1, -1] = 0
    weights = np.linalg.solve(x_train.T @ x_train + regularizer, x_train.T @ y_train_norm)
    prediction = (x_test @ weights) * y_std + y_mean
    residual = ((y_test - prediction) ** 2).sum(0)
    total = ((y_test - y_test.mean(0)) ** 2).sum(0).clip(min=1e-12)
    r2 = 1.0 - residual / total
    baseline_mae = np.abs(y_test - y_mean).mean(0).clip(min=1e-12)
    normalized_mae = np.abs(y_test - prediction).mean(0) / baseline_mae
    names = ("mean_flow_x", "mean_flow_y", "mean_flow_magnitude", "p90_flow_magnitude")
    return {
        "available": True,
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "r2": {name: float(value) for name, value in zip(names, r2)},
        "mean_r2": float(r2.mean()),
        "mae_relative_to_mean_baseline": {
            name: float(value) for name, value in zip(names, normalized_mae)
        },
    }


def _source_probe(
    latents: np.ndarray,
    sources: list[str],
    test_mask: np.ndarray,
) -> dict[str, Any]:
    train_mask = ~test_mask
    unique_sources = sorted(set(sources))
    x_mean = latents[train_mask].mean(0)
    x_std = latents[train_mask].std(0).clip(min=1e-6)
    normalized = (latents - x_mean) / x_std
    centroids = {}
    for source in unique_sources:
        mask = np.asarray([value == source for value in sources]) & train_mask
        if mask.any():
            centroids[source] = normalized[mask].mean(0)
    test_indices = np.flatnonzero(test_mask)
    if not centroids or len(test_indices) < 4:
        return {"available": False, "reason": "insufficient deterministic probe split"}
    predictions = []
    for index in test_indices:
        prediction = min(
            centroids,
            key=lambda source: float(((normalized[index] - centroids[source]) ** 2).sum()),
        )
        predictions.append(prediction)
    labels = [sources[index] for index in test_indices]
    return {
        "available": True,
        "accuracy": float(np.mean([a == b for a, b in zip(labels, predictions)])),
        "chance_accuracy_uniform": 1.0 / len(unique_sources),
        "classes": len(unique_sources),
        "interpretation": "Lower is better; high accuracy indicates source-specific shortcuts in the latent.",
    }


def _latent_health(latents: np.ndarray, log_variances: np.ndarray) -> dict[str, Any]:
    covariance = np.cov(latents, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, 1e-12)
    positive = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    participation_ratio = float(total**2 / max(float((eigenvalues**2).sum()), 1e-12))
    variance = latents.var(0)
    return {
        "latent_dimensions": int(latents.shape[1]),
        "mean_l2_norm": float(np.linalg.norm(latents, axis=1).mean()),
        "mean_dimension_variance": float(variance.mean()),
        "min_dimension_variance": float(variance.min()),
        "max_dimension_variance": float(variance.max()),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "mean_log_variance": float(log_variances.mean()),
        "mean_posterior_std": float(np.exp(0.5 * log_variances).mean()),
    }


def _aggregate(records: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    micro = {field: float(np.mean([row[field] for row in records])) for field in METRIC_FIELDS}
    by_source: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["source"]].append(row)
    for source, rows in sorted(grouped.items()):
        by_source[source] = {
            "samples": len(rows),
            **{field: float(np.mean([row[field] for row in rows])) for field in METRIC_FIELDS},
        }
    macro = {
        field: float(np.mean([metrics[field] for metrics in by_source.values()]))
        for field in METRIC_FIELDS
    }
    return micro, {"macro_average": macro, "by_source": by_source}


def _save_montage(items: list[tuple[Tensor, Tensor]], path: Path) -> None:
    if not items:
        return
    rows = []
    for video, reconstruction in items:
        first = (video[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        target = (video[1].clamp(0, 1).numpy() * 255).astype(np.uint8)
        predicted = (reconstruction.clamp(0, 1).numpy() * 255).astype(np.uint8)
        error = np.abs(target.astype(np.int16) - predicted.astype(np.int16)).clip(0, 255).astype(np.uint8)
        rows.append(np.concatenate([first, target, predicted, error], axis=1))
    canvas = np.concatenate(rows, axis=0)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    labels = ("condition", "target", "reconstruction", "absolute error")
    width = items[0][0].shape[2]
    for index, label in enumerate(labels):
        draw.rectangle((index * width, 0, index * width + 145, 22), fill=(0, 0, 0))
        draw.text((index * width + 4, 4), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _update_summary(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "checkpoint_label",
        "checkpoint_global_step",
        "samples",
        "mse",
        "psnr",
        "ssim",
        "lpips",
        "zero_latent_mse",
        "shuffled_latent_mse",
        "latent_effective_rank",
        "motion_probe_mean_r2",
        "action_probe_mean_r2",
        "action_cross_source_macro_mean_r2",
        "source_probe_accuracy",
        "camera_probe_accuracy",
    ]
    existing = []
    if path.exists():
        with path.open(newline="") as source:
            existing = list(csv.DictReader(source))
    existing = [value for value in existing if value["checkpoint_label"] != row["checkpoint_label"]]
    existing.append({key: row.get(key, "") for key in fieldnames})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(arguments: argparse.Namespace) -> Path:
    if arguments.dataset_kind == "lance":
        if arguments.dataset_version is None:
            raise ValueError("--dataset-version is required for a Lance benchmark")
        manifest = build_fixed_eval_manifest(
            arguments.manifest,
            arguments.dataset_path,
            arguments.dataset_version,
            num_samples=arguments.num_samples,
            camera_allowlist=arguments.camera,
            time_gaps_s=arguments.time_gap,
            eval_fraction=arguments.eval_fraction,
            split_seed=arguments.split_seed,
        )
        dataset: FixedLanceEvalDataset | FixedTeleopEvalDataset = FixedLanceEvalDataset(
            arguments.manifest
        )
    else:
        manifest = build_fixed_teleop_eval_manifest(
            arguments.manifest,
            arguments.dataset_path,
            time_gaps_s=arguments.time_gap,
            train_samples_per_episode=arguments.train_samples_per_episode,
            validation_samples_per_episode=arguments.validation_samples_per_episode,
            split_seed=arguments.split_seed,
        )
        dataset = FixedTeleopEvalDataset(arguments.manifest)
    action_dimension_names = tuple(
        manifest.get("action_dimension_names", ACTION_DIMENSION_NAMES)
    )
    action_groups = {
        key: tuple(indices)
        for key, indices in manifest.get("action_groups", ACTION_GROUPS).items()
    }
    checkpoint_path = Path(arguments.checkpoint)
    label = _checkpoint_label(checkpoint_path, arguments.checkpoint_label)
    output_root = Path(arguments.output_dir)
    checkpoint_output = output_root / label
    result_path = checkpoint_output / "result.json"
    if result_path.exists() and not arguments.overwrite:
        print(f"LAM_EVAL_ALREADY_COMPLETE result={result_path}")
        return result_path

    device = torch.device(arguments.device)
    model, checkpoint_global_step = _load_model(checkpoint_path, device)
    lpips_metric = piq.LPIPS(reduction="none").eval().to(device)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": arguments.batch_size,
        "shuffle": False,
        "num_workers": arguments.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if arguments.num_workers:
        loader_kwargs.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
    loader = DataLoader(**loader_kwargs)

    records: list[dict[str, Any]] = []
    latents: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    flow_targets: list[list[float]] = []
    action_targets: list[np.ndarray] = []
    action_validity: list[np.ndarray] = []
    action_representations: list[str] = []
    probe_splits: list[str] = []
    montage: list[tuple[Tensor, Tensor]] = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    for batch in tqdm(loader, desc=f"LAM eval {label}"):
        videos_cpu = batch["videos"]
        videos = videos_cpu.to(device, non_blocking=True)
        batch_size, _, height, width, _ = videos.shape
        with autocast:
            encoded = model.lam.encode(videos)
            latent = encoded["z_mu"].reshape(batch_size, 1, 1, -1)
            correct = _decode_latent(model, encoded["patches"], latent, height, width)
            zero = _decode_latent(model, encoded["patches"], torch.zeros_like(latent), height, width)
            shuffled = _decode_latent(model, encoded["patches"], latent.roll(1, 0), height, width)
        model.lam.mu_record = None
        target = rearrange(videos[:, 1:], "b t h w c -> (b t) c h w")
        correct_images = rearrange(correct, "b t h w c -> (b t) c h w")
        zero_images = rearrange(zero, "b t h w c -> (b t) c h w")
        shuffled_images = rearrange(shuffled, "b t h w c -> (b t) c h w")
        values = _per_sample_image_metrics(target, correct_images, lpips_metric)
        values.update(_variant_metrics(target, zero_images, "zero_latent"))
        values.update(_variant_metrics(target, shuffled_images, "shuffled_latent"))

        latent_array = encoded["z_mu"].float().cpu().numpy()
        log_variance_array = encoded["z_var"].float().cpu().numpy()
        latents.append(latent_array)
        log_variances.append(log_variance_array)
        action_targets.append(batch["action_target"].float().cpu().numpy())
        action_validity.append(batch["action_valid"].bool().cpu().numpy())
        action_representations.extend(batch["action_representation"])
        batch_probe_splits = list(batch.get("probe_split", ["deterministic"] * batch_size))
        probe_splits.extend(batch_probe_splits)
        for index in range(batch_size):
            row = {
                "sample_id": batch["sample_id"][index],
                "source": batch["source"][index],
                "row_id": int(batch["row_id"][index]),
                "episode_id": batch["episode_id"][index],
                "task": batch["task"][index],
                "camera": batch["camera"][index],
                "time_gap_s": float(batch["time_gap_s"][index]),
                "action_target": batch["action_target"][index].float().tolist(),
                "action_valid": batch["action_valid"][index].bool().tolist(),
                "action_steps": int(batch["action_steps"][index]),
                "action_representation": batch["action_representation"][index],
                "probe_split": batch_probe_splits[index],
            }
            row.update({field: float(values[field][index]) for field in METRIC_FIELDS})
            records.append(row)
            flow_targets.append(_flow_target(videos_cpu[index]))
            metric_split = manifest.get("metric_split")
            if (
                len(montage) < arguments.montage_samples
                and (metric_split is None or batch_probe_splits[index] == metric_split)
            ):
                montage.append((videos_cpu[index], correct[index, 0].float().cpu()))

    latent_array = np.concatenate(latents, axis=0)
    log_variance_array = np.concatenate(log_variances, axis=0)
    flow_array = np.asarray(flow_targets, dtype=np.float64)
    action_target_array = np.concatenate(action_targets, axis=0)
    action_validity_array = np.concatenate(action_validity, axis=0)
    episode_group_ids = [f'{row["source"]}\0{row["row_id"]}' for row in records]
    sources = [row["source"] for row in records]
    cameras = [row["camera"] for row in records]
    if manifest.get("probe_split_field"):
        metric_split = str(manifest["metric_split"])
        test_mask = np.asarray(
            [probe_split == metric_split for probe_split in probe_splits], dtype=bool
        )
    else:
        metric_split = None
        test_mask = deterministic_test_mask(episode_group_ids)
    metric_records = (
        [record for record, selected in zip(records, test_mask) if selected]
        if metric_split is not None
        else records
    )
    if not metric_records:
        raise RuntimeError("fixed benchmark metric split is empty")
    micro, source_metrics = _aggregate(metric_records)
    latent_health = _latent_health(latent_array, log_variance_array)
    motion_probe = _ridge_probe(latent_array, flow_array, test_mask)
    action_probe = masked_ridge_probe(
        latent_array,
        action_target_array,
        action_validity_array,
        test_mask,
        action_dimension_names,
        action_groups,
    )
    cross_source_action_probe = leave_one_category_out_ridge(
        latent_array,
        action_target_array,
        action_validity_array,
        sources,
        action_dimension_names,
        action_groups,
    )
    source_probe = nearest_centroid_probe(latent_array, sources, test_mask)
    camera_probe = nearest_centroid_probe(latent_array, cameras, test_mask)
    bootstrap = stratified_episode_bootstrap(
        metric_records,
        METRIC_FIELDS,
        arguments.bootstrap_replicates,
        arguments.bootstrap_seed,
    )
    usage = {
        "mse_gain_vs_zero_latent": micro["zero_latent_mse"] - micro["mse"],
        "mse_gain_vs_shuffled_latent": micro["shuffled_latent_mse"] - micro["mse"],
        "psnr_gain_vs_zero_latent": micro["psnr"] - micro["zero_latent_psnr"],
        "psnr_gain_vs_shuffled_latent": micro["psnr"] - micro["shuffled_latent_psnr"],
    }
    result = {
        "schema": "dreamdojo-lam-checkpoint-eval-v2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_label": label,
        "checkpoint_global_step": checkpoint_global_step,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "eval_manifest": str(arguments.manifest),
        "eval_manifest_sha256": sha256(Path(arguments.manifest).read_bytes()).hexdigest(),
        "dataset_kind": arguments.dataset_kind,
        "dataset_version": manifest.get("dataset_version"),
        "accepted_filter": manifest["accepted_filter"],
        "samples": len(metric_records),
        "probe_samples": len(records),
        "probe_train_samples": int((~test_mask).sum()),
        "probe_test_samples": int(test_mask.sum()),
        "metric_split": metric_split,
        "leakage_status": arguments.leakage_status,
        "micro_average": micro,
        **source_metrics,
        "latent_action_utilization": usage,
        "latent_health": latent_health,
        "optical_flow_linear_probe": motion_probe,
        "canonical_robot_action_probe": action_probe,
        "canonical_robot_action_cross_source_probe": cross_source_action_probe,
        "canonical_robot_action_coverage": {
            "samples_with_any_valid_dimension": int(action_validity_array.any(1).sum()),
            "valid_fraction_by_dimension": {
                name: float(action_validity_array[:, index].mean())
                for index, name in enumerate(action_dimension_names)
            },
            "representations": dict(sorted(Counter(action_representations).items())),
        },
        "source_leakage_nearest_centroid_probe": source_probe,
        "camera_leakage_nearest_centroid_probe": camera_probe,
        "episode_bootstrap_95ci": bootstrap,
    }
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    with (checkpoint_output / "samples.jsonl").open("w") as destination:
        for row in records:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
    _save_montage(montage, checkpoint_output / "reconstructions.png")
    _update_summary(
        output_root / "summary.csv",
        {
            "checkpoint_label": label,
            "checkpoint_global_step": checkpoint_global_step,
            "samples": len(metric_records),
            **micro,
            "latent_effective_rank": latent_health["effective_rank"],
            "motion_probe_mean_r2": motion_probe.get("mean_r2", ""),
            "action_probe_mean_r2": action_probe.get("mean_r2", ""),
            "action_cross_source_macro_mean_r2": cross_source_action_probe.get(
                "macro_mean_r2", ""
            ),
            "source_probe_accuracy": source_probe.get("accuracy", ""),
            "camera_probe_accuracy": camera_probe.get("accuracy", ""),
        },
    )
    print(
        "LAM_EVAL_OK "
        f"checkpoint={label} samples={len(metric_records)} psnr={micro['psnr']:.4f} "
        f"ssim={micro['ssim']:.4f} lpips={micro['lpips']:.4f} "
        f"action_r2={action_probe.get('mean_r2', float('nan')):.4f} "
        f"result={result_path}"
    )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-label")
    parser.add_argument(
        "--dataset-kind", choices=("lance", "pantheon-teleop"), default="lance"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-fraction", type=float, default=0.01)
    parser.add_argument("--train-samples-per-episode", type=int, default=4)
    parser.add_argument("--validation-samples-per-episode", type=int, default=64)
    parser.add_argument("--split-seed", default="dreamdojo-lam-eval-v1")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        choices=("left_wrist", "right_wrist", "overhead"),
    )
    parser.add_argument("--time-gap", action="append", type=float, default=None)
    parser.add_argument("--montage-samples", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", default="dreamdojo-lam-bootstrap-v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--leakage-status",
        default="fixed benchmark; current live run sampled the full accepted pool",
    )
    arguments = parser.parse_args()
    arguments.camera = arguments.camera or ["left_wrist", "right_wrist", "overhead"]
    arguments.time_gap = arguments.time_gap or [0.1, 0.2, 0.3, 0.4]
    return arguments


if __name__ == "__main__":
    evaluate(parse_args())
