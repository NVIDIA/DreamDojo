from pathlib import Path


def _load_dataset_paths(config_path):
    """Extract dataset_path entries from a config YAML using only stdlib."""
    paths = []
    in_dataset_path = False
    with open(config_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            if stripped.startswith("dataset_path:"):
                in_dataset_path = True
                continue
            if in_dataset_path:
                if stripped.startswith("- "):
                    if indent > 4:
                        paths.append(stripped[2:].strip())
                    else:
                        in_dataset_path = False
                elif indent <= 4:
                    in_dataset_path = False
    return paths


def test_no_test_splits_or_parent_overlap_in_configs():
    config_dir = Path(__file__).parent.parent / "configs"
    for config_path in config_dir.glob("*.yaml"):
        basenames = [p.rsplit("/", 1)[-1] for p in _load_dataset_paths(config_path)]
        base_set = set(basenames)

        test_splits = [b for b in basenames if b.endswith("_test")]
        assert not test_splits, (
            f"{config_path}: test splits cannot be used for training: {test_splits}"
        )

        for b in basenames:
            if b.endswith("_train") or b.endswith("_test"):
                base = b.rsplit("_", 1)[0]
                assert base not in base_set, (
                    f"{config_path}: base dataset {base} overlaps with its split {b}"
                )

