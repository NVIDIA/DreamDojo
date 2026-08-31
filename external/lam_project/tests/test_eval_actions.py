import unittest

import numpy as np

from lam.eval_actions import ACTION_DIM, aggregate_robot_action


class EvalActionsTest(unittest.TestCase):
    def test_integrates_midpoint_actions_and_requires_complete_validity(self):
        first = np.arange(ACTION_DIM, dtype=np.float32)
        second = first + 1.0
        validity = np.ones((2, ACTION_DIM), dtype=bool)
        validity[1, 3] = False
        robot = {
            "action_timestamps_ns": [25, 75],
            "smoothed_actions_global": [first.tolist(), second.tolist()],
            "action_valid_global": validity.tolist(),
        }
        camera = {"frame_timestamps_ns": [0, 50, 100]}

        target, valid, steps, representation = aggregate_robot_action(
            robot, camera, [0, 2]
        )

        self.assertEqual(steps, 2)
        self.assertEqual(representation, "smoothed_actions_global")
        np.testing.assert_allclose(target[valid], (first + second)[valid])
        self.assertFalse(valid[3])
        self.assertEqual(target[3], 0.0)

    def test_prefers_global_and_falls_back_to_local(self):
        action = np.ones((1, ACTION_DIM), dtype=np.float32)
        validity = np.ones_like(action, dtype=bool)
        robot = {
            "action_timestamps_ns": [25],
            "actions_local": action.tolist(),
            "action_valid_local": validity.tolist(),
        }
        target, valid, steps, representation = aggregate_robot_action(
            robot, {"frame_timestamps_ns": [0, 50]}, [0, 1]
        )
        np.testing.assert_allclose(target, 1.0)
        self.assertTrue(valid.all())
        self.assertEqual(steps, 1)
        self.assertEqual(representation, "actions_local")

    def test_returns_masked_zero_when_interval_has_no_action(self):
        target, valid, steps, representation = aggregate_robot_action(
            {
                "action_timestamps_ns": [200],
                "actions_global": [[1.0] * ACTION_DIM],
                "action_valid_global": [[True] * ACTION_DIM],
            },
            {"frame_timestamps_ns": [0, 50]},
            [0, 1],
        )
        np.testing.assert_allclose(target, 0.0)
        self.assertFalse(valid.any())
        self.assertEqual(steps, 0)
        self.assertEqual(representation, "unavailable")


if __name__ == "__main__":
    unittest.main()
