
#!/usr/bin/env python3
"""Build a lightweight manifest from an extracted EgoSpatial-CL directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Extracted dataset root containing Town/split/episode dirs.")
    parser.add_argument("--reference-manifest", type=Path, help="Optional published benchmark_manifest.json.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference_by_uid = {}
    compact_domains = []
    if args.reference_manifest:
        ref = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
        compact_domains = ref.get("compact_domains", [])
        reference_by_uid = {e["episode_uid"]: e for e in ref["episodes"]}

    episodes = []
    for index_csv in sorted(args.root.glob("Town*/**/episode_*/index.csv")):
        ep_dir = index_csv.parent
        split = ep_dir.parent.name
        town = ep_dir.parent.parent.name
        episode = ep_dir.name
        uid = f"{town}/{split}/{episode}"
        meta_path = ep_dir / "meta.json"
        state_path = ep_dir / "state.csv"
        index_rows = read_rows(index_csv)
        state_rows = read_rows(state_path)
        entry = {
            "episode_uid": uid,
            "town": town,
            "split": split,
            "episode": episode,
            "num_samples": len(index_rows),
            "paths": {
                "episode_dir": str(ep_dir.relative_to(args.root)),
                "index_csv": str(index_csv.relative_to(args.root)),
                "state_csv": str(state_path.relative_to(args.root)),
                "meta_json": str(meta_path.relative_to(args.root)),
            },
        }
        entry.update({k: v for k, v in reference_by_uid.get(uid, {}).items() if k not in entry})
        if len(index_rows) != len(state_rows):
            raise ValueError(f"index/state row mismatch for {uid}: {len(index_rows)} != {len(state_rows)}")
        episodes.append(entry)

    manifest = {
        "benchmark_name": "egospatial-cl-v1-local-extracted",
        "schema_version": "1.0",
        "compact_domains": compact_domains,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output} episodes={len(episodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
