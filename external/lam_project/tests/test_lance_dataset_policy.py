import importlib.util
from pathlib import Path
import sys
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The policy helper is pure Python, but the production module also owns the
# optional CV/Torch decoder. Supply import-time stubs so this unit test remains
# runnable in a lightweight checkout.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
torch_stub = types.ModuleType("torch")
torch_stub.Tensor = object
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules.setdefault("torch.nn.functional", types.ModuleType("torch.nn.functional"))
einops_stub = types.ModuleType("einops")
einops_stub.rearrange = lambda value, *_args, **_kwargs: value
sys.modules.setdefault("einops", einops_stub)
torch_utils_stub = types.ModuleType("torch.utils")
torch_data_stub = types.ModuleType("torch.utils.data")
torch_data_stub.Dataset = object
sys.modules.setdefault("torch.utils", torch_utils_stub)
sys.modules.setdefault("torch.utils.data", torch_data_stub)

lam_package = types.ModuleType("lam")
lam_package.__path__ = [str(PROJECT_ROOT / "lam")]
sys.modules.setdefault("lam", lam_package)
load_module("lam.manifest", "lam/manifest.py")
lance_dataset = load_module("lam.lance_dataset", "lam/lance_dataset.py")


class AcceptedLancePolicyTest(unittest.TestCase):
    def test_filter_is_exactly_accepted_and_valid(self):
        self.assertEqual(
            lance_dataset.ACCEPTED_FILTER,
            "quality.capture_status = 'accepted' AND quality.capture_valid = true",
        )
        self.assertNotIn("accepted_with_issue", lance_dataset.ACCEPTED_FILTER)

    def test_valid_windows_enforce_every_sampled_frame(self):
        validity = [True, True, False, True, True, True]
        self.assertEqual(lance_dataset.valid_window_starts(validity, 2, 2), [1, 3])

    def test_invalid_window_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            lance_dataset.valid_window_starts([True], 0, 1)

    def test_source_weights_default_to_episode_proportions(self):
        names, weights = lance_dataset.normalized_row_source_weights(
            {"umi": [1, 2, 3], "teleop": [4]},
            None,
        )
        self.assertEqual(names, ["teleop", "umi"])
        self.assertEqual(weights, [0.25, 0.75])

    def test_episode_holdout_is_deterministic_disjoint_and_per_source(self):
        grouped = {
            "small": list(range(10)),
            "large": list(range(100, 300)),
        }
        train = lance_dataset.deterministic_episode_split(
            grouped, "train", eval_fraction=0.1, split_seed="fixed"
        )
        evaluation = lance_dataset.deterministic_episode_split(
            grouped, "eval", eval_fraction=0.1, split_seed="fixed"
        )
        repeated = lance_dataset.deterministic_episode_split(
            grouped, "eval", eval_fraction=0.1, split_seed="fixed"
        )
        self.assertEqual(evaluation, repeated)
        for source, rows in grouped.items():
            self.assertTrue(evaluation[source])
            self.assertFalse(set(train[source]) & set(evaluation[source]))
            self.assertEqual(set(train[source]) | set(evaluation[source]), set(rows))

    def test_episode_split_rejects_invalid_contract(self):
        with self.assertRaises(ValueError):
            lance_dataset.deterministic_episode_split({"x": [1, 2]}, "holdout")
        with self.assertRaises(ValueError):
            lance_dataset.deterministic_episode_split(
                {"x": [1, 2]}, "eval", eval_fraction=1.0
            )


if __name__ == "__main__":
    unittest.main()
