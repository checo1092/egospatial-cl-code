
#!/usr/bin/env python3
"""PyTorch frame-level loader for an extracted EgoSpatial-CL dataset copy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_TARGET_COLUMNS = ("steer", "throttle", "brake")


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_image(path: Path, image_size: Optional[Tuple[int, int]], nearest: bool = False) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        h, w = image_size
        resample = Image.NEAREST if nearest else Image.BILINEAR
        img = img.resize((w, h), resample=resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def _load_lidar(path: Path, max_points: Optional[int]) -> torch.Tensor:
    pts = np.load(path).astype(np.float32, copy=False)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Unexpected LiDAR shape for {path}: {pts.shape}")
    pts = pts[:, :4] if pts.shape[1] >= 4 else np.pad(pts[:, :3], ((0, 0), (0, 1)), constant_values=1.0)
    if max_points is not None and pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[idx]
    return torch.from_numpy(pts)


class EgoSpatialCLFrameDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        split: str = "train",
        towns: Optional[Sequence[str]] = None,
        domain_names: Optional[Sequence[str]] = None,
        modalities: Sequence[str] = ("rgb",),
        target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
        image_size: Optional[Tuple[int, int]] = (224, 224),
        frame_stride: int = 1,
        max_lidar_points: Optional[int] = None,
        return_meta: bool = True,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.split = split
        self.towns = set(towns) if towns is not None else None
        self.domain_names = set(domain_names) if domain_names is not None else None
        self.modalities = tuple(modalities)
        self.target_columns = tuple(target_columns)
        self.image_size = image_size
        self.frame_stride = max(1, int(frame_stride))
        self.max_lidar_points = max_lidar_points
        self.return_meta = return_meta
        unknown = set(self.modalities) - {"rgb", "semseg", "lidar"}
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("No samples matched the requested filters.")

    def _episode_dir(self, ep: Dict[str, Any]) -> Path:
        return self.dataset_root / ep["town"] / ep["split"] / ep["episode"]

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for ep in self.manifest["episodes"]:
            if ep["split"] != self.split:
                continue
            if self.towns is not None and ep["town"] not in self.towns:
                continue
            if self.domain_names is not None and ep["domain_name"] not in self.domain_names:
                continue
            ep_dir = self._episode_dir(ep)
            index_rows = _read_csv_dicts(ep_dir / "index.csv")
            state_rows = _read_csv_dicts(ep_dir / "state.csv")
            state_by_frame = {int(float(row["frame"])): row for row in state_rows}
            for frame_idx, idx_row in enumerate(index_rows):
                if frame_idx % self.frame_stride:
                    continue
                frame = int(float(idx_row["frame"]))
                samples.append({"episode": ep, "episode_dir": ep_dir, "index": idx_row, "state": state_by_frame[frame], "frame_idx": frame_idx})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        ep_dir = sample["episode_dir"]
        index = sample["index"]
        state = sample["state"]
        out: Dict[str, Any] = {}
        if "rgb" in self.modalities:
            out["rgb"] = _load_image(ep_dir / index["rgb_path"], self.image_size)
        if "semseg" in self.modalities:
            out["semseg"] = _load_image(ep_dir / index["semseg_path"], self.image_size, nearest=True)
        if "lidar" in self.modalities:
            out["lidar"] = _load_lidar(ep_dir / index["lidar_path"], self.max_lidar_points)
        out["target"] = torch.tensor([float(state[col]) for col in self.target_columns], dtype=torch.float32)
        if self.return_meta:
            ep = sample["episode"]
            out["meta"] = {
                "episode_uid": ep.get("episode_uid"),
                "town": ep["town"],
                "split": ep["split"],
                "domain_name": ep.get("domain_name"),
                "frame": int(float(index["frame"])),
                "frame_idx": sample["frame_idx"],
            }
        return out


def collate_egospatial_cl(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("rgb", "semseg", "target"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    if "lidar" in batch[0]:
        out["lidar"] = [b["lidar"] for b in batch]
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out
