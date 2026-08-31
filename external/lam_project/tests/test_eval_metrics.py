import unittest

import numpy as np

from lam.eval_metrics import (
    deterministic_test_mask,
    masked_ridge_probe,
    nearest_centroid_probe,
    stratified_episode_bootstrap,
)


class EvalMetricsTest(unittest.TestCase):
    def test_probe_split_keeps_episode_variants_together(self):
        mask = deterministic_test_mask(["episode-a", "episode-a", "episode-b"])
        self.assertEqual(bool(mask[0]), bool(mask[1]))

    def test_masked_ridge_recovers_linear_targets(self):
        generator = np.random.default_rng(7)
        latents = generator.normal(size=(400, 4))
        targets = np.stack(
            [2.0 * latents[:, 0] - latents[:, 1], latents[:, 2] + 0.5], axis=1
        )
        validity = np.ones_like(targets, dtype=bool)
        validity[::7, 1] = False
        test_mask = np.zeros(len(latents), dtype=bool)
        test_mask[::5] = True
        result = masked_ridge_probe(
            latents,
            targets,
            validity,
            test_mask,
            ("first", "second"),
            {"all": (0, 1)},
        )
        self.assertTrue(result["available"])
        self.assertGreater(result["mean_r2"], 0.999)
        self.assertEqual(result["available_dimensions"], 2)
        self.assertGreater(result["groups"]["all"]["mean_r2"], 0.999)

    def test_nearest_centroid_detects_separated_nuisance_classes(self):
        generator = np.random.default_rng(11)
        latents = np.concatenate(
            [generator.normal(-3, 0.1, (50, 3)), generator.normal(3, 0.1, (50, 3))]
        )
        labels = ["a"] * 50 + ["b"] * 50
        test_mask = np.zeros(100, dtype=bool)
        test_mask[::5] = True
        result = nearest_centroid_probe(latents, labels, test_mask)
        self.assertTrue(result["available"])
        self.assertEqual(result["accuracy"], 1.0)

    def test_episode_bootstrap_is_deterministic_and_clustered(self):
        records = [
            {"source": "a", "episode_id": "1", "row_id": 1, "mse": 1.0},
            {"source": "a", "episode_id": "1", "row_id": 1, "mse": 1.0},
            {"source": "a", "episode_id": "2", "row_id": 2, "mse": 3.0},
            {"source": "b", "episode_id": "3", "row_id": 3, "mse": 5.0},
            {"source": "b", "episode_id": "4", "row_id": 4, "mse": 7.0},
        ]
        first = stratified_episode_bootstrap(records, ("mse",), 100, "fixed")
        second = stratified_episode_bootstrap(records, ("mse",), 100, "fixed")
        self.assertEqual(first, second)
        self.assertEqual(first["unique_episodes"], 4)
        self.assertLessEqual(
            first["intervals"]["mse"]["lower"],
            first["intervals"]["mse"]["upper"],
        )


if __name__ == "__main__":
    unittest.main()
