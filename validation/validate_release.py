
#!/usr/bin/env python3
"""Validate public EgoSpatial-CL v1.0.0 manifest and shard metadata."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import Counter
from pathlib import Path


EXPECTED_TOWNS = {
    "Town01_Opt", "Town02_Opt", "Town04_Opt", "Town05_Opt",
    "Town06", "Town07", "Town10HD_Opt",
}
EXPECTED_DOMAINS = {
    "clear__NPC00_none", "night__NPC00_none", "fog__NPC00_none",
    "hardrain__NPC00_none", "clear__NPC11_veh_high",
    "hardrain__NPC11_veh_high", "clear__NPC31_both_high",
}
EXPECTED_SPLITS = {"train": 392, "val": 49, "test": 49}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest(path: Path) -> None:
    manifest = load_json(path)
    episodes = manifest["episodes"]
    assert len(episodes) == 490, f"expected 490 episodes, found {len(episodes)}"
    assert {e["town"] for e in episodes} == EXPECTED_TOWNS
    assert {e["domain_name"] for e in episodes} == EXPECTED_DOMAINS
    split_counts = Counter(e["split"] for e in episodes)
    assert dict(split_counts) == EXPECTED_SPLITS, split_counts
    global_ids = [e["global_episode_index"] for e in episodes]
    assert sorted(global_ids) == list(range(490))
    episode_uids = [e["episode_uid"] for e in episodes]
    assert len(set(episode_uids)) == len(episode_uids)
    samples = sum(int(e.get("num_samples", 1000)) for e in episodes)
    assert samples == 490000, f"expected 490000 samples, found {samples}"
    print("OK manifest")


def check_shards_manifest(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty shards manifest"
    total_episodes = sum(int(r["num_episodes"]) for r in rows)
    if "num_samples" in rows[0]:
        total_samples = sum(int(r["num_samples"]) for r in rows)
    else:
        total_samples = total_episodes * 1000
    assert total_episodes == 490, total_episodes
    assert total_samples == 490000, total_samples
    print(f"OK shards_manifest rows={len(rows)}")


def check_tar(path: Path) -> None:
    with tarfile.open(path, "r") as tf:
        names = tf.getnames()
    meta = [n for n in names if n.endswith("/meta.json")]
    state = [n for n in names if n.endswith("/state.csv")]
    index = [n for n in names if n.endswith("/index.csv")]
    images = [n for n in names if "/images/" in n and n.endswith(".png")]
    labels = [n for n in names if "/labels/" in n and n.endswith(".png")]
    lidar = [n for n in names if "/lidar/" in n and n.endswith(".npy")]
    assert meta and len(meta) == len(state) == len(index)
    assert len(images) == len(labels) == len(lidar)
    assert len(images) == 1000 * len(meta)
    print(f"OK tar path={path} episodes={len(meta)} samples={len(images)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--shards-manifest", type=Path)
    parser.add_argument("--tar", action="append", type=Path, default=[])
    args = parser.parse_args()

    check_manifest(args.manifest)
    if args.shards_manifest:
        check_shards_manifest(args.shards_manifest)
    for tar_path in args.tar:
        check_tar(tar_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
