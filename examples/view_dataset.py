#!/usr/bin/env python3
"""Browse extracted Hugging Face episodes and export synchronized PNG/GIF previews.

Python 3.10+. Install examples/requirements-viewer.txt.
Run --help for input filters, playback, and export options. No CARLA or torch needed.
Interactive mode uses a desktop Matplotlib window; --export works headlessly.
Only episodes present locally are offered; the manifest supplies release identities.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

COLORS = {"speed": "#117a65", "steer": "#3265b0", "throttle": "#16804a", "brake": "#c64751"}
REQUIRED = ("frame", "sim_time", "vel_x", "vel_y", "vel_z", "steer", "throttle", "brake")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes the selected dataset or episode directory")
    return path


class Episode:
    def __init__(self, root, record):
        self.record = record
        self.uid = "/".join(str(record[k]) for k in ("town", "split", "episode"))
        self.path = within(root, self.uid)
        self.index = read_rows(self.path / "index.csv")
        state = read_rows(self.path / "state.csv")
        if not self.index or not state:
            raise ValueError(f"{self.uid}: empty index or state table")
        by_frame = {}
        for row in state:
            if any(not row.get(k) for k in REQUIRED):
                raise ValueError(f"{self.uid}: missing required state fields")
            frame = int(row["frame"])
            if frame in by_frame:
                raise ValueError(f"{self.uid}: duplicate state frame {frame}")
            by_frame[frame] = row
        frames = [int(row["frame"]) for row in self.index]
        if len(frames) != len(set(frames)):
            raise ValueError(f"{self.uid}: duplicate index frames")
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError(f"{self.uid}: index frames are not increasing")
        if set(frames) != set(by_frame):
            raise ValueError(f"{self.uid}: index/state frame sets differ")
        self.state = [by_frame[f] for f in frames]
        self.series = {k: [float(row[k]) for row in self.state] for k in REQUIRED}
        if not all(np.isfinite(values).all() for values in self.series.values()):
            raise ValueError(f"{self.uid}: non-finite state values")
        times = np.asarray(self.series["sim_time"])
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"{self.uid}: simulation time is not increasing")
        self.series["time"] = (times - times[0]).tolist()
        self.series["speed"] = (3.6 * np.sqrt(sum(
            np.square(self.series[k]) for k in ("vel_x", "vel_y", "vel_z")
        ))).tolist()

    def sample(self, index, max_points):
        if not 0 <= index < len(self.index):
            raise ValueError("Sample index out of range")
        row = self.index[index]
        images = []
        for key in ("rgb_path", "semseg_path"):
            if not row.get(key):
                raise ValueError(f"Missing modality: {key}")
            with Image.open(within(self.path, row[key])) as image:
                images.append(image.convert("RGB").copy())
        if not row.get("lidar_path"):
            raise ValueError("Missing modality: lidar_path")
        points = np.load(within(self.path, row["lidar_path"]), allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError("LiDAR must be a finite (N, 4) array")
        total = len(points)
        if total > max_points:
            points = points[np.linspace(0, total - 1, max_points, dtype=int)]
        return images, points, total


class Release:
    def __init__(self, args):
        self.root = args.dataset_root.expanduser().resolve()
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        self.records = {}
        for record in manifest["episodes"]:
            if any(getattr(args, key) and record[field] != getattr(args, key)
                   for key, field in (("town", "town"), ("split", "split"), ("domain_name", "domain_name"))):
                continue
            uid = "/".join(str(record[k]) for k in ("town", "split", "episode"))
            if (within(self.root, uid) / "index.csv").is_file():
                if uid in self.records:
                    raise ValueError(f"Duplicate manifest identity: {uid}")
                self.records[uid] = record
        if not self.records:
            raise ValueError("No extracted HF episodes match the manifest and filters")
        self.args = args

    def episode(self, uid):
        if uid not in self.records:
            raise ValueError("Episode is not available")
        return Episode(self.root, self.records[uid])


class Panel:
    """Shared Matplotlib artists for the desktop viewer and headless exports."""

    def __init__(self, figure, episode, index, max_points, color_mode, interactive=False):
        self.figure = figure
        self.episode = episode
        self.max_points = max_points
        self.color_mode = color_mode
        self.index = index
        grid = figure.add_gridspec(
            2, 3, height_ratios=(1.7, 1), left=.055, right=.96,
            top=.85, bottom=.19 if interactive else .09, hspace=.45, wspace=.30)
        self.images = []
        for col, title in enumerate(("RGB", "Semantic segmentation")):
            ax = figure.add_subplot(grid[0, col])
            self.images.append(ax.imshow(np.zeros((600, 800, 3)), interpolation="nearest"))
            ax.set_title(title, loc="left", fontsize=11)
            ax.axis("off")
        ax = figure.add_subplot(grid[0, 2])
        self.cloud = ax.scatter([], [], c=[], s=1, cmap="viridis")
        ax.scatter([0], [0], marker="^", c="#d24b55", s=45)
        ax.set(xlim=(-50, 50), ylim=(-50, 50), xlabel="y / right (m)", ylabel="x / forward (m)")
        ax.set_aspect("equal")
        self.cloud_title = ax.set_title("LiDAR", loc="left", fontsize=11)
        label = "Height (m)" if color_mode == "height" else "Intensity"
        self.cloud.set_clim(*((-3, 5) if color_mode == "height" else (0, 1)))
        figure.colorbar(self.cloud, ax=ax, fraction=.046, pad=.04, label=label)
        series = episode.series
        self.cursors = []
        for col, keys, title in ((0, ("speed",), "Speed (km/h)"),
                                 (1, ("steer", "throttle", "brake"), "Applied controls")):
            ax = figure.add_subplot(grid[1, col])
            for key in keys:
                ax.plot(series["time"], series[key], color=COLORS[key], label=key, linewidth=1)
            self.cursors.append(ax.axvline(series["time"][index], color="#202624", linewidth=1))
            if col == 1:
                ax.set_ylim(-1.05, 1.05)
                ax.legend(loc="lower right", fontsize=8)
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xlabel("Episode time (s)")
            ax.grid(alpha=.18)
        ax = figure.add_subplot(grid[1, 2])
        ax.axis("off")
        self.state_text = ax.text(0, 1, "", va="top", fontsize=11, linespacing=1.7)
        figure.text(.055, .96, "EgoSpatial-CL", fontsize=22, weight="bold")
        figure.text(.055, .915, f"{episode.uid}  |  {episode.record.get('domain_name', '')}", fontsize=11)
        figure.text(.055, .02, "Playback is not real-time. LiDAR coordinates are in the sensor frame.", fontsize=9)
        self.update(index)

    def update(self, index):
        images, points, total = self.episode.sample(index, self.max_points)
        # Read all modalities successfully before changing any displayed artists.
        for artist, image in zip(self.images, images):
            artist.set_data(image)
            width, height = image.size
            artist.set_extent((-.5, width - .5, height - .5, -.5))
        self.cloud.set_offsets(points[:, [1, 0]])
        self.cloud.set_array(points[:, 2 if self.color_mode == "height" else 3])
        self.cloud_title.set_text(f"LiDAR - {len(points):,}/{total:,} points")
        series = self.episode.series
        for cursor in self.cursors:
            cursor.set_xdata([series["time"][index]] * 2)
        self.state_text.set_text(
            f"Sample {index + 1} / {len(self.episode.index)}\n"
            f"Frame {int(series['frame'][index])}\n"
            f"Simulation time  {series['sim_time'][index]:.2f} s\n"
            f"Speed  {series['speed'][index]:.2f} km/h\n"
            f"Steer  {series['steer'][index]:+.3f}\n"
            f"Throttle  {series['throttle'][index]:.3f}    Brake  {series['brake'][index]:.3f}")
        self.index = index


def export(release, uid, index, count, stride, fps, color, suffix):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if count < 1 or count > 120 or stride < 1 or not 1 <= fps <= 30:
        raise ValueError("Use 1-120 frames, stride >= 1, and 1-30 fps")
    if suffix not in (".png", ".gif") or color not in ("height", "intensity"):
        raise ValueError("Unsupported export format or LiDAR color")
    episode = release.episode(uid)
    if not 0 <= index < len(episode.index):
        raise ValueError("Sample index out of range")
    indices = [index] if suffix == ".png" else list(
        range(index, min(index + count * stride, len(episode.index)), stride))
    figure = Figure(figsize=(14, 8), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    panel = Panel(figure, episode, index, release.args.max_points, color)
    frames = []
    for i in indices:
        if i != index:
            panel.update(i)
        canvas.draw()
        frames.append(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB"))
    output = io.BytesIO()
    if suffix == ".png":
        frames[0].save(output, format="PNG")
    else:
        frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:],
                       duration=round(1000 / fps), loop=0, disposal=2)
    return output.getvalue(), indices


class LocalViewer:
    def __init__(self, figure, episode, args):
        from matplotlib.widgets import Button, Slider

        self.panel = Panel(figure, episode, args.sample, args.max_points, args.color, interactive=True)
        self.playing = False
        self.slider = Slider(figure.add_axes((.12, .10, .78, .025)), "Sample",
                             1, max(2, len(episode.index)), valinit=args.sample + 1,
                             valstep=1, valfmt="%0.0f")
        self.buttons = []
        for x, label, callback in ((.30, "Previous", lambda _: self.step(-1)),
                                   (.44, "Play", self.toggle),
                                   (.58, "Next", lambda _: self.step(1))):
            button = Button(figure.add_axes((x, .05, .12, .035)), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.timer = figure.canvas.new_timer(interval=round(1000 / args.fps))
        self.timer.add_callback(self.tick)
        self.slider.on_changed(self.seek)
        figure.canvas.mpl_connect("close_event", lambda _: self.stop())
        figure.canvas.mpl_connect("key_press_event", self.key)
        self.status = figure.text(.055, .155, "", fontsize=9, color="#a52031")

    def seek(self, value):
        index = min(int(value) - 1, len(self.panel.episode.index) - 1)
        try:
            self.panel.update(index)
            self.status.set_text("")
        except (ValueError, OSError) as error:
            self.stop()
            self.status.set_text("Unable to load sample; see terminal.")
            print(f"Sample error: {error}")
            self.slider.eventson = False
            self.slider.set_val(self.panel.index + 1)
            self.slider.eventson = True
        self.panel.figure.canvas.draw_idle()

    def step(self, delta):
        self.stop()
        self.slider.set_val(max(1, min(len(self.panel.episode.index), self.panel.index + 1 + delta)))

    def toggle(self, _=None):
        if self.playing:
            self.stop()
        elif self.panel.index < len(self.panel.episode.index) - 1:
            self.playing = True
            self.buttons[1].label.set_text("Pause")
            self.timer.start()
        self.panel.figure.canvas.draw_idle()

    def stop(self):
        self.playing = False
        self.timer.stop()
        self.buttons[1].label.set_text("Play")

    def tick(self):
        if not self.playing:
            return
        if self.panel.index >= len(self.panel.episode.index) - 1:
            self.stop()
        else:
            self.slider.set_val(self.panel.index + 2)
            if self.panel.index == len(self.panel.episode.index) - 1:
                self.stop()
        self.panel.figure.canvas.draw_idle()

    def key(self, event):
        if event.key in ("left", "right"):
            self.step(-1 if event.key == "left" else 1)
        elif event.key == " ":
            self.toggle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extraction root containing Town/split/episode directories")
    parser.add_argument("--manifest", type=Path, required=True, help="Public benchmark_manifest.json")
    parser.add_argument("--town")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--domain-name")
    parser.add_argument("--list-episodes", action="store_true", help="List locally available episode UIDs and exit")
    parser.add_argument("--max-points", type=int, default=20000, help="Deterministic display cap; source clouds are not modified")
    parser.add_argument("--export", type=Path, help="Export .png or .gif instead of opening the desktop window")
    parser.add_argument("--episode", help="Episode UID; defaults to first locally available episode")
    parser.add_argument("--sample", type=int, default=0, help="Zero-based initial saved-sample index")
    parser.add_argument("--count", type=int, default=24, help="GIF frame count (1-120), clipped at episode end")
    parser.add_argument("--stride", type=int, default=1, help="Saved-sample stride for GIF")
    parser.add_argument("--fps", type=int, default=6, help="Playback rate; not real-time (1-30)")
    parser.add_argument("--color", choices=("height", "intensity"), default="height")
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    if not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")
    try:
        release = Release(args)
        if args.list_episodes:
            for uid, record in release.records.items():
                print(f"{uid}  {record.get('domain_name', '')}")
            return
        uid = args.episode or next(iter(release.records))
        if args.export:
            suffix = args.export.suffix.lower()
            if suffix not in (".png", ".gif"):
                parser.error("--export must end in .png or .gif")
            destination = args.export.expanduser().resolve()
            if destination.is_relative_to(release.root):
                parser.error("Choose an export path outside the extracted dataset")
            if destination.exists() or destination.with_suffix(destination.suffix + ".json").exists():
                parser.error("Export or provenance already exists; choose a new filename")
            content, indices = export(release, uid, args.sample, args.count, args.stride, args.fps, args.color, suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            episode = release.episode(uid)
            provenance = {"episode_uid": uid, "domain_name": episode.record.get("domain_name"),
                          "sample_indices": indices,
                          "frame_ids": [int(episode.series["frame"][i]) for i in indices],
                          "sim_times": [episode.series["sim_time"][i] for i in indices],
                          "playback": "not real-time", "fps": args.fps if suffix == ".gif" else None,
                          "lidar_color": args.color, "max_points": args.max_points}
            destination.with_suffix(destination.suffix + ".json").write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
            print(f"Exported {destination} ({len(indices)} samples)")
        else:
            import matplotlib
            import matplotlib.pyplot as plt

            if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template", "pgf") or "inline" in matplotlib.get_backend().lower():
                parser.error("Interactive mode needs a desktop backend (e.g. TkAgg). Use --export on a headless system.")
            episode = release.episode(uid)
            if not 0 <= args.sample < len(episode.index):
                parser.error("--sample is outside the selected episode")
            figure = plt.figure(figsize=(14, 9), dpi=100, facecolor="white")
            figure.canvas.manager.set_window_title("EgoSpatial-CL")
            viewer = LocalViewer(figure, episode, args)
            print(f"Viewing {uid} ({len(episode.index)} samples)")
            plt.show()
            viewer.stop()
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
