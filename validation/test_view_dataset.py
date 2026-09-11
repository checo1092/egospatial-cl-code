"""Focused viewer contract tests using temporary synthetic episode files."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from view_dataset import Episode, Release, LocalViewer, export, within


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = {"town": "Town06", "split": "val", "episode": "episode_0008",
                       "domain_name": "clear__NPC00_none"}
        self.uid = "Town06/val/episode_0008"
        self.path = self.root / self.uid
        self.path.mkdir(parents=True)
        self.rows = []
        self.index = []
        for i, frame in enumerate((10, 20, 30)):
            self.rows.append(dict(frame=frame, sim_time=100 + i * .5, vel_x=i + 1,
                                  vel_y=0, vel_z=0, steer=i * .1, throttle=.5, brake=0))
            Image.new("RGB", (16, 12), (i * 70, 120, 90)).save(self.path / f"{frame}.png")
            np.save(self.path / f"{frame}.npy",
                    np.array([[1, 2, -1, .3], [2, 3, 0, .6]], dtype=np.float32))
            self.index.append(dict(frame=frame, rgb_path=f"{frame}.png",
                                   semseg_path=f"{frame}.png", lidar_path=f"{frame}.npy"))
        self.write("state.csv", self.rows)
        self.write("index.csv", self.index)
        manifest = self.root / "manifest.json"
        missing = dict(self.record, episode="episode_9999")
        manifest.write_text(json.dumps({"episodes": [self.record, missing]}))
        self.args = argparse.Namespace(dataset_root=self.root, manifest=manifest,
                                       town=None, split=None, domain_name=None, max_points=20000)

    def write(self, name, rows):
        with (self.path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_available_episodes_only(self):
        release = Release(self.args)
        self.assertEqual(list(release.records), [self.uid])
        self.args.town = "Town07"
        with self.assertRaisesRegex(ValueError, "No extracted"):
            Release(self.args)

    def test_sync_by_frame_not_csv_order(self):
        self.write("state.csv", self.rows[::-1])
        episode = Episode(self.root, self.record)
        self.assertEqual(episode.series["frame"], [10, 20, 30])
        np.testing.assert_allclose(episode.series["speed"], [3.6, 7.2, 10.8])
        self.assertEqual(episode.series["time"], [0, .5, 1])

    def test_duplicate_and_missing_frames_fail(self):
        self.write("state.csv", [self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate state"):
            Episode(self.root, self.record)
        self.write("state.csv", self.rows[:2])
        with self.assertRaisesRegex(ValueError, "frame sets differ"):
            Episode(self.root, self.record)

    def test_nonfinite_state_fails(self):
        self.rows[0]["vel_x"] = "nan"
        self.write("state.csv", self.rows)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            Episode(self.root, self.record)

    def test_point_cap_and_missing_modality(self):
        episode = Episode(self.root, self.record)
        images, points, total = episode.sample(0, 1)
        self.assertEqual(images[0].size, (16, 12))
        self.assertEqual(points.shape, (1, 4))
        self.assertEqual(total, 2)
        episode.index[0]["semseg_path"] = ""
        with self.assertRaisesRegex(ValueError, "Missing modality"):
            episode.sample(0, 10)

    def test_bad_cloud_and_sample_bounds(self):
        episode = Episode(self.root, self.record)
        np.save(self.path / "10.npy", np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "LiDAR"):
            episode.sample(0, 10)
        for index in (-1, 3):
            with self.assertRaisesRegex(ValueError, "out of range"):
                episode.sample(index, 10)

    def test_paths_cannot_escape(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            within(self.root, "../outside")
        (self.path / "outside.png").symlink_to("/etc/passwd")
        episode = Episode(self.root, self.record)
        episode.index[0]["rgb_path"] = "outside.png"
        with self.assertRaisesRegex(ValueError, "escapes"):
            episode.sample(0, 10)

    def test_exports_and_end_clipping(self):
        release = Release(self.args)
        data, indices = export(release, self.uid, 1, 3, 1, 6, "height", ".gif")
        self.assertEqual(indices, [1, 2])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.n_frames, 2)
            self.assertEqual(image.size, (1400, 800))
        data, indices = export(release, self.uid, 0, 1, 1, 6, "intensity", ".png")
        self.assertEqual(indices, [0])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.format, "PNG")
        with self.assertRaises(ValueError):
            export(release, self.uid, 0, 121, 1, 6, "height", ".gif")

    def test_navigation_playback_and_failed_frame(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        args = argparse.Namespace(sample=0, max_points=20000, color="height", fps=6)
        figure = Figure(figsize=(14, 9))
        FigureCanvasAgg(figure)
        viewer = LocalViewer(figure, Episode(self.root, self.record), args)
        viewer.step(1)
        self.assertEqual(viewer.panel.index, 1)
        np.testing.assert_allclose(viewer.panel.cursors[0].get_xdata(), [.5, .5])
        viewer.toggle()
        self.assertTrue(viewer.playing)
        viewer.tick()
        self.assertEqual(viewer.panel.index, 2)
        self.assertFalse(viewer.playing)
        viewer.step(1)
        self.assertEqual(viewer.panel.index, 2)
        viewer.step(-1)
        (self.path / "30.npy").unlink()
        viewer.slider.set_val(3)
        self.assertEqual(viewer.panel.index, 1)
        self.assertEqual(viewer.slider.val, 2)
        self.assertFalse(viewer.playing)
        viewer.key(argparse.Namespace(key="left"))
        self.assertEqual(viewer.panel.index, 0)
        viewer.key(argparse.Namespace(key=" "))
        self.assertTrue(viewer.playing)
        viewer.stop()


if __name__ == "__main__":
    unittest.main()
