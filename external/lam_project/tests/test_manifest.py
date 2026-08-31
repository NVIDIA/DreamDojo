import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "lam" / "manifest.py"
SPEC = importlib.util.spec_from_file_location("manifest_under_test", MODULE_PATH)
manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest
SPEC.loader.exec_module(manifest)


class ManifestTest(unittest.TestCase):
    def test_loads_relative_video_and_filters_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "video_path": "left.mp4",
                    "source": "umi",
                    "camera": "left_wrist",
                    "fps": 20,
                },
                {
                    "video_path": "overhead.mp4",
                    "source": "umi",
                    "camera": "overhead",
                    "fps": 30,
                },
            ]
            manifest_path = root / "videos.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            entries = manifest.load_video_manifests(
                [manifest_path],
                camera_allowlist=["left_wrist"],
                require_files=False,
            )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].camera, "left_wrist")
        self.assertEqual(entries[0].video_path, (root / "left.mp4").resolve())

    def test_time_gap_uses_each_videos_fps(self):
        self.assertEqual(manifest.frame_stride_for_gap(20, 0.2), 4)
        self.assertEqual(manifest.frame_stride_for_gap(30, 0.2), 6)

    def test_source_weights_must_cover_loaded_sources(self):
        entries = [
            manifest.VideoManifestEntry(Path("a.mp4"), "umi", "left_wrist", 20),
            manifest.VideoManifestEntry(Path("b.mp4"), "teleop", "overhead", 30),
        ]
        grouped = manifest.group_entries_by_source(entries)
        with self.assertRaises(ValueError):
            manifest.normalized_source_weights(grouped, {"umi": 1.0})


if __name__ == "__main__":
    unittest.main()
