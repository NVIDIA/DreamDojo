"""Pure NumPy statistics and probes for LAM checkpoint evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np


def deterministic_test_mask(group_ids: Sequence[str], denominator: int = 5) -> np.ndarray:
    """Assign entire episode groups to a stable probe test split."""

    if denominator < 2:
        raise ValueError("denominator must be at least two")
    return np.asarray(
        [
            int.from_bytes(sha256(value.encode("utf-8")).digest()[:8], "big")
            % denominator
            == 0
            for value in group_ids
        ],
        dtype=bool,
    )


def masked_ridge_probe(
    latents: np.ndarray,
    targets: np.ndarray,
    validity: np.ndarray,
    test_mask: np.ndarray,
    dimension_names: Sequence[str],
    dimension_groups: Mapping[str, Sequence[int]] | None = None,
    ridge: float = 1e-2,
    min_test_samples: int = 8,
) -> dict[str, Any]:
    """Fit independent deterministic ridge probes with per-target validity."""

    latents = np.asarray(latents, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    validity = np.asarray(validity, dtype=bool) & np.isfinite(targets)
    test_mask = np.asarray(test_mask, dtype=bool)
    if latents.ndim != 2 or targets.ndim != 2 or validity.shape != targets.shape:
        raise ValueError("latents, targets, and validity must be compatible matrices")
    if len(latents) != len(targets) or len(test_mask) != len(latents):
        raise ValueError("probe arrays must contain the same number of samples")
    if len(dimension_names) != targets.shape[1]:
        raise ValueError("dimension_names must match the target width")

    dimensions: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(dimension_names):
        train = (~test_mask) & validity[:, index]
        test = test_mask & validity[:, index]
        if train.sum() <= latents.shape[1] or test.sum() < min_test_samples:
            dimensions[name] = {
                "available": False,
                "train_samples": int(train.sum()),
                "test_samples": int(test.sum()),
                "reason": "insufficient valid episode-level probe split",
            }
            continue

        x_train, x_test = latents[train], latents[test]
        y_train, y_test = targets[train, index], targets[test, index]
        x_mean = x_train.mean(0)
        x_std = x_train.std(0).clip(min=1e-6)
        y_mean = float(y_train.mean())
        y_std = float(max(y_train.std(), 1e-6))
        x_train = (x_train - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std
        x_train = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
        x_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)
        regularizer = np.eye(x_train.shape[1]) * ridge
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(
            x_train.T @ x_train + regularizer,
            x_train.T @ ((y_train - y_mean) / y_std),
        )
        prediction = (x_test @ weights) * y_std + y_mean
        residual = float(np.square(y_test - prediction).sum())
        total = float(np.square(y_test - y_test.mean()).sum())
        r2 = 1.0 - residual / max(total, 1e-12)
        baseline_mae = float(np.abs(y_test - y_mean).mean())
        relative_mae = float(np.abs(y_test - prediction).mean()) / max(
            baseline_mae, 1e-12
        )
        dimensions[name] = {
            "available": True,
            "train_samples": int(train.sum()),
            "test_samples": int(test.sum()),
            "r2": float(r2),
            "mae_relative_to_train_mean_baseline": float(relative_mae),
            "target_train_std": y_std,
        }

    available = [value for value in dimensions.values() if value["available"]]
    result: dict[str, Any] = {
        "available": bool(available),
        "ridge": float(ridge),
        "split_unit": "episode",
        "available_dimensions": len(available),
        "dimensions": dimensions,
    }
    if available:
        result.update(
            mean_r2=float(np.mean([value["r2"] for value in available])),
            median_r2=float(np.median([value["r2"] for value in available])),
            mean_relative_mae=float(
                np.mean(
                    [
                        value["mae_relative_to_train_mean_baseline"]
                        for value in available
                    ]
                )
            ),
        )
    if dimension_groups:
        grouped = {}
        for group, indices in dimension_groups.items():
            values = [dimensions[dimension_names[index]] for index in indices]
            values = [value for value in values if value["available"]]
            grouped[group] = {
                "available_dimensions": len(values),
                "mean_r2": (
                    float(np.mean([value["r2"] for value in values]))
                    if values
                    else None
                ),
                "mean_relative_mae": (
                    float(
                        np.mean(
                            [
                                value["mae_relative_to_train_mean_baseline"]
                                for value in values
                            ]
                        )
                    )
                    if values
                    else None
                ),
            }
        result["groups"] = grouped
    return result


def leave_one_category_out_ridge(
    latents: np.ndarray,
    targets: np.ndarray,
    validity: np.ndarray,
    categories: Sequence[str],
    dimension_names: Sequence[str],
    dimension_groups: Mapping[str, Sequence[int]] | None = None,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Measure canonical action transfer to each unseen source category."""

    categories_array = np.asarray(categories)
    by_category = {}
    for category in sorted(set(categories)):
        by_category[category] = masked_ridge_probe(
            latents,
            targets,
            validity,
            categories_array == category,
            dimension_names,
            dimension_groups,
            ridge=ridge,
        )
    available = [
        value["mean_r2"]
        for value in by_category.values()
        if value.get("available") and "mean_r2" in value
    ]
    return {
        "available": bool(available),
        "held_out_unit": "source_dataset",
        "macro_mean_r2": float(np.mean(available)) if available else None,
        "by_source": by_category,
    }


def nearest_centroid_probe(
    latents: np.ndarray,
    labels: Sequence[str],
    test_mask: np.ndarray,
) -> dict[str, Any]:
    """Probe nuisance-label leakage with a deterministic nearest centroid."""

    labels_array = np.asarray(labels)
    train_mask = ~np.asarray(test_mask, dtype=bool)
    unique = sorted(set(labels))
    if len(unique) < 2:
        return {"available": False, "reason": "fewer than two classes"}
    x_mean = latents[train_mask].mean(0)
    x_std = latents[train_mask].std(0).clip(min=1e-6)
    normalized = (latents - x_mean) / x_std
    centroids = {
        label: normalized[(labels_array == label) & train_mask].mean(0)
        for label in unique
        if ((labels_array == label) & train_mask).any()
    }
    test_indices = np.asarray(
        [
            index
            for index in np.flatnonzero(test_mask)
            if labels_array[index] in centroids
        ],
        dtype=np.int64,
    )
    if len(centroids) < 2 or len(test_indices) < 4:
        return {"available": False, "reason": "insufficient episode-level split"}
    predictions = [
        min(
            centroids,
            key=lambda label: float(
                np.square(normalized[index] - centroids[label]).sum()
            ),
        )
        for index in test_indices
    ]
    accuracy = float(
        np.mean([prediction == labels_array[index] for prediction, index in zip(predictions, test_indices)])
    )
    return {
        "available": True,
        "accuracy": accuracy,
        "chance_accuracy_uniform": 1.0 / len(centroids),
        "classes": len(centroids),
        "total_classes": len(unique),
        "classes_without_probe_train_examples": sorted(set(unique) - set(centroids)),
        "train_samples": int(train_mask.sum()),
        "test_samples": len(test_indices),
        "split_unit": "episode",
        "interpretation": "Lower is better; high accuracy indicates nuisance-specific shortcuts.",
    }


def stratified_episode_bootstrap(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    replicates: int,
    seed: str,
) -> dict[str, Any]:
    """Return source-stratified, episode-clustered percentile intervals."""

    if replicates < 1:
        return {"available": False, "replicates": int(replicates)}
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        source = str(record["source"])
        episode = str(record.get("episode_id") or record["row_id"])
        grouped[source][episode].append(index)
    if not grouped:
        return {"available": False, "replicates": int(replicates)}
    arrays = {
        field: np.asarray([float(record[field]) for record in records], dtype=np.float64)
        for field in fields
    }
    seed_value = int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng(seed_value)
    estimates = {field: np.empty(replicates, dtype=np.float64) for field in fields}
    source_groups = [list(episodes.values()) for episodes in grouped.values()]
    for replicate in range(replicates):
        sampled_indices = []
        for episode_groups in source_groups:
            choices = generator.integers(0, len(episode_groups), size=len(episode_groups))
            sampled_indices.extend(
                index
                for choice in choices
                for index in episode_groups[int(choice)]
            )
        indices = np.asarray(sampled_indices, dtype=np.int64)
        for field, values in arrays.items():
            estimates[field][replicate] = float(values[indices].mean())
    intervals = {
        field: {
            "lower": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for field, values in estimates.items()
    }
    return {
        "available": True,
        "confidence": 0.95,
        "method": "percentile bootstrap",
        "sampling_unit": "episode",
        "stratified_by": "source_dataset",
        "replicates": int(replicates),
        "seed": seed,
        "unique_episodes": sum(len(episodes) for episodes in grouped.values()),
        "unique_episodes_by_source": {
            source: len(episodes) for source, episodes in sorted(grouped.items())
        },
        "intervals": intervals,
    }
