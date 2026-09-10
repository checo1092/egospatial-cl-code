#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_episodes.py — EgoSpatial-CL episode runner (CARLA 0.9.15)

Compatible with the collect_episode.py CLI:
  --host --port --cd-root --episode-id --map-name -N --fixed-dt --seed
  [--weather {current,clear,cloudy,wet,hardrain}] [--semseg] [--lidar]
  [--overwrite] [--park-server] + extras (warmup-ticks, sensor-timeout, etc.)

Collector output layout (fixed contract):
  episode_0000/
    images/   (RGB)
    labels/   (SemSeg)
    lidar/    (LiDAR)
    meta.json
    state.csv
    index.csv

This runner:
- Runs K episodes via subprocess
- Creates cd-root shim so collector writes into dataset root
- Creates normalized symlinks: episodes/ep_000000 -> ../episode_0000
- Verifies outputs + collects integrity (hashes, sizes, counts)
- Writes episodes_manifest.json atomically
- Builds dataset_index.jsonl (prefers index.csv per episode; fallback enumerates files)
- Writes dataset_report.json (summary + disk usage + splits + global index hash)

IMPORTANT:
- --collect-extra-args uses argparse.REMAINDER, so it MUST BE LAST in the command.
"""

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

MANIFEST_NAME = "episodes_manifest.json"
GLOBAL_INDEX_NAME = "dataset_index.jsonl"
REPORT_NAME = "dataset_report.json"

# Fixed contract for collector outputs
MODALITIES = {
    "rgb":    {"dir": "images", "patterns": ["*.png", "*.jpg", "*.jpeg"], "min_frames": 1},
    "semseg": {"dir": "labels", "patterns": ["*.png"],                   "min_frames": 1},
    "lidar":  {"dir": "lidar",  "patterns": ["*.npy", "*.pcd"],          "min_frames": 1},
}
REQUIRED_FILES = ["meta.json", "state.csv", "index.csv"]


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(tmp), str(path))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size)


def dir_size_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += int(p.stat().st_size)
    return total


def run_cmd(cmd: List[str], log_path: Optional[Path] = None, cwd=None, env=None, timeout=None) -> int:
    """
    Execute subprocess. If log_path, redirect stdout/stderr to file.
    """
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write("CMD: {}\n\n".format(" ".join(cmd)))
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd, env=env)
            try:
                return p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                return 124
    else:
        try:
            subprocess.check_call(cmd, cwd=cwd, env=env, timeout=timeout)
            return 0
        except subprocess.CalledProcessError as e:
            return e.returncode
        except subprocess.TimeoutExpired:
            return 124


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def wait_for_server(host: str, port: int, attempts: int = 30, sleep_s: float = 2.0) -> bool:
    for _ in range(attempts):
        if port_open(host, port, timeout=2.0):
            return True
        time.sleep(sleep_s)
    return False


def docker_restart_container(container_name: str) -> int:
    try:
        return subprocess.call(["docker", "restart", container_name])
    except Exception as e:
        print(f"[WARN] docker restart failed for {container_name}: {e}")
        return 1


def try_carla_health_check(host: str, port: int, timeout_s: float = 2.0):
    """
    Light health check. Returns None if carla import fails.
    """
    try:
        import carla  # noqa
    except Exception:
        return None

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout_s)
        world = client.get_world()
        settings = world.get_settings()
        return {
            "synchronous_mode": bool(settings.synchronous_mode),
            "fixed_delta_seconds": float(settings.fixed_delta_seconds) if settings.fixed_delta_seconds else None,
            "no_rendering_mode": bool(settings.no_rendering_mode),
            "map": world.get_map().name,
        }
    except Exception:
        return {"error": "carla_health_check_failed"}


def count_state_rows(state_csv_path: Path) -> int:
    with open(state_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0
    first = rows[0]
    has_alpha = any(any(c.isalpha() for c in cell) for cell in first)
    start = 1 if has_alpha else 0
    return max(0, len(rows) - start)


def glob_count(dirpath: Path, patterns: List[str]) -> int:
    total = 0
    for pat in patterns:
        total += len(glob.glob(str(Path(dirpath) / pat)))
    return total


def verify_episode_outputs(ep_dir: Path, expected_frames: int, enabled_modalities: List[str], strict: bool = True) -> Dict:
    """
    Fixed-contract verification:
      - required files exist
      - modality folders exist and counts == expected_frames (strict)
      - state_rows == expected_frames (strict)  [header excluded]
    """
    ep_dir = Path(ep_dir)
    report = {
        "ok": True,
        "missing": [],
        "counts": {},
        "state_rows": None,
        "notes": [],
    }

    for rf in REQUIRED_FILES:
        if not (ep_dir / rf).exists():
            report["ok"] = False
            report["missing"].append(rf)

    state_csv = ep_dir / "state.csv"
    if state_csv.exists():
        report["state_rows"] = count_state_rows(state_csv)
        if strict and expected_frames is not None and report["state_rows"] != int(expected_frames):
            report["ok"] = False
            report["notes"].append(f"state_rows({report['state_rows']}) != expected_frames({expected_frames})")

    for m in enabled_modalities:
        cfg = MODALITIES[m]
        d = ep_dir / cfg["dir"]
        if not d.exists():
            report["ok"] = False
            report["missing"].append(cfg["dir"])
            continue
        c = glob_count(d, cfg["patterns"])
        report["counts"][m] = c
        if c < cfg.get("min_frames", 0):
            report["ok"] = False
        if strict and expected_frames is not None and c != int(expected_frames):
            report["ok"] = False

    return report


def ensure_symlink(link_path: Path, target_dir: Path) -> None:
    """
    Creates/updates: link_path -> target_dir (relative when possible).
    """
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            raise RuntimeError(f"[FATAL] {link_path} exists and is a directory. Remove it manually.")

    rel = os.path.relpath(str(target_dir), str(link_path.parent))
    os.symlink(rel, str(link_path))


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {
            "schema_version": "1.3",
            "created_utc": utcnow_iso(),
            "updated_utc": utcnow_iso(),
            "dataset_root": None,
            "episodes": []
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_episode(manifest: Dict, entry: Dict) -> None:
    eid = entry.get("episode_id")
    eps = manifest.get("episodes", [])
    for i, e in enumerate(eps):
        if e.get("episode_id") == eid:
            eps[i] = entry
            manifest["episodes"] = eps
            return
    eps.append(entry)
    manifest["episodes"] = eps


def _rel_any(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except Exception:
        return str(path)


def index_csv_to_jsonl(ep_dir: Path, dataset_root: Path, episode_meta: Dict) -> List[Dict]:
    """
    Converts ep_dir/index.csv -> list of dict rows for dataset_index.jsonl.

    Row keys produced:
      episode_id, episode_relpath, frame_idx, frame_id (if present),
      rgb, semseg, lidar (if present),
      state_csv, meta_json, index_csv,
      seed, map, weather, fixed_dt
    """
    idx_path = ep_dir / "index.csv"
    if not idx_path.exists():
        return []

    with open(idx_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    cols = [c.strip() for c in (reader.fieldnames or [])]
    lower = [c.lower() for c in cols]

    def pick(*keys) -> Optional[str]:
        for k in keys:
            for c, cl in zip(cols, lower):
                if k in cl:
                    return c
        return None

    col_frame = pick("frame", "frame_id", "carla_frame")
    col_rgb   = pick("image", "rgb", "camera")
    col_lbl   = pick("label", "seg", "sem")
    col_lid   = pick("lidar", "point")

    def resolve_path(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        p = Path(value)
        if p.is_absolute():
            return _rel_any(p, dataset_root)
        return _rel_any((ep_dir / p), dataset_root)

    out = []
    for i, r in enumerate(rows):
        row = {
            "episode_id": int(episode_meta["episode_id"]),
            "episode_relpath": str(episode_meta["collector_output_relpath"]),
            "frame_idx": i,
            "state_csv": _rel_any(ep_dir / "state.csv", dataset_root),
            "meta_json": _rel_any(ep_dir / "meta.json", dataset_root),
            "index_csv": _rel_any(ep_dir / "index.csv", dataset_root),
            "seed": int(episode_meta.get("seed", 0)),
            "map": episode_meta.get("map"),
            "weather": episode_meta.get("weather"),
            "fixed_dt": episode_meta.get("fixed_dt"),
        }

        if col_frame and r.get(col_frame):
            try:
                row["frame_id"] = int(float(r[col_frame]))
            except Exception:
                row["frame_id"] = r[col_frame]

        if col_rgb:
            v = resolve_path(r.get(col_rgb))
            if v:
                row["rgb"] = v
        if col_lbl:
            v = resolve_path(r.get(col_lbl))
            if v:
                row["semseg"] = v
        if col_lid:
            v = resolve_path(r.get(col_lid))
            if v:
                row["lidar"] = v

        out.append(row)

    return out


def regenerate_global_index(dataset_root: Path, manifest: Dict, out_path: Path) -> None:
    """
    Writes dataset_index.jsonl global.

    Priority order per episode:
      1) ep_dir/index.jsonl (if present, concat)
      2) ep_dir/index.csv   (preferred; parsed into jsonl rows)
      3) fallback: enumerate files in images/labels/lidar
    """
    dataset_root = Path(dataset_root)
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    ok_eps = [e for e in manifest.get("episodes", []) if e.get("status") == "ok"]
    ok_eps = sorted(ok_eps, key=lambda x: x.get("episode_id", 0))

    with open(tmp, "w", encoding="utf-8") as wf:
        for e in ok_eps:
            collector_rel = e.get("collector_output_relpath")
            if not collector_rel:
                continue
            ep_dir = dataset_root / collector_rel
            if not ep_dir.exists():
                continue

            # 1) per-episode jsonl
            per_index = ep_dir / "index.jsonl"
            if per_index.exists():
                with open(per_index, "r", encoding="utf-8") as rf:
                    shutil.copyfileobj(rf, wf)
                continue

            # 2) index.csv preferred
            rows = index_csv_to_jsonl(ep_dir, dataset_root, e)
            if rows:
                for r in rows:
                    wf.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            # 3) fallback enumeration
            rgb_dir = ep_dir / MODALITIES["rgb"]["dir"]
            if not rgb_dir.exists():
                continue

            rgb_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                rgb_files.extend(sorted(rgb_dir.glob(ext)))
            if not rgb_files:
                continue

            sem_files = []
            lid_files = []
            sem_dir = ep_dir / MODALITIES["semseg"]["dir"]
            lid_dir = ep_dir / MODALITIES["lidar"]["dir"]

            if sem_dir.exists():
                sem_files = sorted(sem_dir.glob("*.png"))
            if lid_dir.exists():
                lid_files.extend(sorted(lid_dir.glob("*.npy")))
                lid_files.extend(sorted(lid_dir.glob("*.pcd")))

            for i, rgb in enumerate(rgb_files):
                row = {
                    "episode_id": int(e["episode_id"]),
                    "episode_relpath": str(e.get("collector_output_relpath")),
                    "frame_idx": i,
                    "rgb": _rel_any(rgb, dataset_root),
                    "state_csv": _rel_any(ep_dir / "state.csv", dataset_root),
                    "meta_json": _rel_any(ep_dir / "meta.json", dataset_root),
                    "index_csv": _rel_any(ep_dir / "index.csv", dataset_root),
                    "seed": int(e.get("seed", 0)),
                    "map": e.get("map"),
                    "weather": e.get("weather"),
                    "fixed_dt": e.get("fixed_dt"),
                }
                if i < len(sem_files):
                    row["semseg"] = _rel_any(sem_files[i], dataset_root)
                if i < len(lid_files):
                    row["lidar"] = _rel_any(lid_files[i], dataset_root)
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(str(tmp), str(out_path))


def make_episode_splits(dataset_root: Path, manifest: Dict, split_seed: int, train_p: float, val_p: float, test_p: float) -> Dict:
    import random

    dataset_root = Path(dataset_root)
    splits_dir = dataset_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    ok_eps = [e for e in manifest.get("episodes", []) if e.get("status") == "ok"]
    episode_ids = sorted([int(e["episode_id"]) for e in ok_eps])

    rng = random.Random(int(split_seed))
    shuffled = episode_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * float(train_p)))
    n_val = int(round(n * float(val_p)))
    n_test = max(0, n - n_train - n_val)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]

    def write_list(name: str, items: List[int]) -> None:
        p = splits_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for x in items:
                f.write(str(x) + "\n")

    write_list("train_episodes.txt", train)
    write_list("val_episodes.txt", val)
    write_list("test_episodes.txt", test)

    return {"train": train, "val": val, "test": test}


def read_episode_list(path: Path) -> List[int]:
    ids = []
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except Exception:
            pass
    return ids


def build_dataset_report(dataset_root: Path, manifest: Dict) -> Dict:
    """
    Build dataset_report.json: summary of episodes, counts, sizes, splits, global index hash, etc.
    """
    dataset_root = Path(dataset_root)

    episodes = manifest.get("episodes", [])
    ok_eps = [e for e in episodes if e.get("status") == "ok"]
    fail_eps = [e for e in episodes if e.get("status") != "ok"]

    ok_eps_sorted = sorted(ok_eps, key=lambda x: int(x.get("episode_id", 0)))
    ok_ids = [int(e["episode_id"]) for e in ok_eps_sorted]
    fail_ids = [int(e["episode_id"]) for e in sorted(fail_eps, key=lambda x: int(x.get("episode_id", 0)))]

    # Global index stats
    index_path = dataset_root / GLOBAL_INDEX_NAME
    index_lines = None
    index_sha256 = None
    index_bytes = None
    if index_path.exists():
        index_bytes = file_size_bytes(index_path)
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_lines = sum(1 for _ in f)
        except Exception:
            index_lines = None
        try:
            index_sha256 = sha256_file(index_path)
        except Exception:
            index_sha256 = None

    # Disk usage (best-effort)
    try:
        dataset_bytes_total = dir_size_bytes(dataset_root)
    except Exception:
        dataset_bytes_total = None

    # Aggregated modality counts
    agg_counts = {k: 0 for k in ["rgb", "semseg", "lidar"]}
    per_episode = []
    for e in ok_eps_sorted:
        ep_rel = e.get("collector_output_relpath")
        if not ep_rel:
            continue
        ep_dir = dataset_root / ep_rel

        ep_counts = {}
        for m, cfg in MODALITIES.items():
            d = ep_dir / cfg["dir"]
            c = glob_count(d, cfg["patterns"]) if d.exists() else 0
            ep_counts[m] = c
            agg_counts[m] += c

        per_episode.append({
            "episode_id": int(e["episode_id"]),
            "seed": int(e.get("seed", 0)),
            "collector_output_relpath": e.get("collector_output_relpath"),
            "episode_symlink_relpath": e.get("episode_relpath"),
            "status": e.get("status"),
            "frames_target": e.get("frames_target"),
            "verify": e.get("verify", {}),
            "counts": ep_counts,
            "bytes_total": e.get("bytes_total"),
            "sha256_meta": e.get("sha256_meta"),
            "sha256_state": e.get("sha256_state"),
            "sha256_index": e.get("sha256_index"),
        })

    # Splits summary
    splits_dir = dataset_root / "splits"
    splits = {}
    if splits_dir.exists():
        train_ids = read_episode_list(splits_dir / "train_episodes.txt")
        val_ids = read_episode_list(splits_dir / "val_episodes.txt")
        test_ids = read_episode_list(splits_dir / "test_episodes.txt")
        splits = {
            "train": {"count": len(train_ids), "episode_ids": train_ids},
            "val": {"count": len(val_ids), "episode_ids": val_ids},
            "test": {"count": len(test_ids), "episode_ids": test_ids},
        }

    run_cfg = manifest.get("run_config", {})
    frames_target = run_cfg.get("frames")
    total_frames = None
    if frames_target is not None:
        try:
            total_frames = int(frames_target) * len(ok_eps)
        except Exception:
            total_frames = None

    report = {
        "report_version": "1.0",
        "created_utc": manifest.get("created_utc", utcnow_iso()),
        "updated_utc": utcnow_iso(),
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_root.name,
        "manifest": {
            "schema_version": manifest.get("schema_version"),
            "path": MANIFEST_NAME,
        },
        "run_config": run_cfg,
        "summary": {
            "episodes_total": len(episodes),
            "episodes_ok": len(ok_eps),
            "episodes_failed": len(fail_eps),
            "episode_ids_ok": ok_ids,
            "episode_ids_failed": fail_ids,
            "frames_target": frames_target,
            "total_frames_ok": total_frames,
            "fixed_dt": run_cfg.get("fixed_dt"),
            "map": run_cfg.get("map"),
            "weather": run_cfg.get("weather"),
        },
        "global_index": {
            "path": str(index_path.relative_to(dataset_root)) if index_path.exists() else None,
            "lines": index_lines,
            "bytes": index_bytes,
            "sha256": index_sha256,
        },
        "modality_counts_total": agg_counts,
        "splits": splits,
        "disk_usage": {
            "dataset_bytes_total": dataset_bytes_total,
        },
        "episodes": per_episode,
    }
    return report


def write_dataset_report(dataset_root: Path, manifest: Dict, out_path: Path) -> None:
    report = build_dataset_report(dataset_root, manifest)
    safe_write_json(out_path, report)


def parse_args():
    ap = argparse.ArgumentParser()

    repo_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--cd-root", default=str(repo_root))
    ap.add_argument("--data-root", default=None, help="Default: <cd-root>/data")
    ap.add_argument("--dataset-name", default="egospatial_cl_v1")
    ap.add_argument("--collect-script", default=None, help="Default: <cd-root>/dataset_generation/collect_episode.py")

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)

    ap.add_argument("--map", dest="town_map", default="Town10HD_Opt")
    ap.add_argument("--weather", default=None, choices=["current", "clear", "cloudy", "wet", "hardrain"])
    ap.add_argument("--task-name", default=None)

    ap.add_argument("--num-episodes", type=int, required=True)
    ap.add_argument("--start-episode-id", type=int, default=0)

    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seed-step", type=int, default=1)

    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fixed-dt", type=float, default=0.05)

    ap.add_argument("--semseg", action="store_true", default=False)
    ap.add_argument("--lidar", action="store_true", default=False)

    ap.add_argument("--strict-verify", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--overwrite", action="store_true", default=False)

    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--episode-timeout", type=int, default=0)

    # collector dir naming: episode_0000
    ap.add_argument("--collector-episode-pad", type=int, default=4)
    # normalized link naming: episodes/ep_000000
    ap.add_argument("--normalized-episode-pad", type=int, default=6)

    # recovery
    ap.add_argument("--wait-server-attempts", type=int, default=30)
    ap.add_argument("--wait-server-sleep", type=float, default=2.0)
    ap.add_argument("--docker-container", default=None)
    ap.add_argument("--docker-restart-on-fail", action="store_true", default=False)
    ap.add_argument(
        "--accept-verified-nonzero-rc",
        action="store_true",
        default=False,
        help="Treat rc!=0 as success if episode outputs verify OK.",
    )

    # IMPORTANT: must be last in CLI (REMAINDER)
    ap.add_argument(
        "--collect-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to collect_episode.py. MUST BE LAST in the command.",
    )

    # splits
    ap.add_argument("--make-splits", action="store_true", default=False)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--train-p", type=float, default=0.8)
    ap.add_argument("--val-p", type=float, default=0.1)
    ap.add_argument("--test-p", type=float, default=0.1)

    # global index
    ap.add_argument("--write-global-index", action="store_true", default=True)

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    cd_root = Path(args.cd_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else (cd_root / "data")
    dataset_root = (data_root / args.dataset_name).resolve()

    scripts_dir = cd_root / "dataset_generation"
    collect_script = Path(args.collect_script).expanduser().resolve() if args.collect_script else (scripts_dir / "collect_episode.py")

    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "logs").mkdir(parents=True, exist_ok=True)
    (dataset_root / "episodes").mkdir(parents=True, exist_ok=True)

    # cd-root shim: <dataset_root>/_cdroot/data -> <dataset_root>
    shim_root = dataset_root / "_cdroot"
    shim_root.mkdir(parents=True, exist_ok=True)
    data_link = shim_root / "data"
    if data_link.exists() or data_link.is_symlink():
        if data_link.is_symlink():
            data_link.unlink()
        else:
            raise RuntimeError(f"[FATAL] {data_link} exists and is not a symlink. Remove it manually.")
    os.symlink(str(dataset_root), str(data_link))

    manifest_path = dataset_root / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    manifest["updated_utc"] = utcnow_iso()
    manifest["dataset_root"] = str(dataset_root)
    manifest.setdefault("run_config", {})
    manifest["run_config"].update({
        "host": args.host,
        "port": args.port,
        "map": args.town_map,
        "weather": args.weather,
        "frames": args.frames,
        "fixed_dt": args.fixed_dt,
        "seed0": args.seed0,
        "seed_step": args.seed_step,
        "modalities": {
            "rgb_dir": MODALITIES["rgb"]["dir"],
            "semseg_dir": MODALITIES["semseg"]["dir"],
            "lidar_dir": MODALITIES["lidar"]["dir"],
            "semseg_enabled": bool(args.semseg),
            "lidar_enabled": bool(args.lidar),
        },
        "collector": {"path": str(collect_script), "cd_root_shim": str(shim_root)},
        "collect_extra_args": list(args.collect_extra_args),
        "schema_contract": "episode_XXXX/{images,labels,lidar,meta.json,state.csv,index.csv}",
    })

    enabled_modalities = ["rgb"]
    if args.semseg:
        enabled_modalities.append("semseg")
    if args.lidar:
        enabled_modalities.append("lidar")

    # Pre-flight: server reachable (with optional docker restart)
    if not wait_for_server(args.host, args.port, attempts=args.wait_server_attempts, sleep_s=args.wait_server_sleep):
        if args.docker_restart_on_fail:
            if not args.docker_container:
                print(f"[FATAL] server no responde y --docker-container no fue provisto (host={args.host} port={args.port})")
                return 2
            print(f"[WARN] CARLA server no responde en {args.host}:{args.port} -> docker restart {args.docker_container}")
            docker_restart_container(args.docker_container)
            if not wait_for_server(args.host, args.port, attempts=args.wait_server_attempts, sleep_s=args.wait_server_sleep):
                print(f"[FATAL] CARLA server no responde tras docker restart en {args.host}:{args.port}")
                return 2
        else:
            print(f"[FATAL] CARLA server no responde en {args.host}:{args.port}")
            return 2

    pre_health = try_carla_health_check(args.host, args.port, timeout_s=2.0)
    if pre_health is not None:
        print("[INFO] Pre health:", pre_health)

    logs_dir = dataset_root / "logs"

    for i in range(int(args.num_episodes)):
        episode_id = int(args.start_episode_id) + i
        seed = int(args.seed0) + i * int(args.seed_step)

        collector_name = f"episode_{episode_id:0{args.collector_episode_pad}d}"
        collector_out_dir = dataset_root / collector_name
        collector_rel = str(collector_out_dir.relative_to(dataset_root))

        norm_name = f"ep_{episode_id:0{args.normalized_episode_pad}d}"
        normalized_link = dataset_root / "episodes" / norm_name
        normalized_rel = str(normalized_link.relative_to(dataset_root))

        # If overwrite requested, remove prior artifacts
        if args.overwrite:
            if collector_out_dir.exists():
                shutil.rmtree(collector_out_dir)
            if normalized_link.exists() or normalized_link.is_symlink():
                normalized_link.unlink()

        # Resume: skip only if verify + (existing manifest entry integrity) passes
        if args.resume and collector_out_dir.exists():
            rep = verify_episode_outputs(collector_out_dir, args.frames, enabled_modalities, strict=args.strict_verify)
            if rep["ok"]:
                existing = None
                for e in manifest.get("episodes", []):
                    if int(e.get("episode_id")) == episode_id:
                        existing = e
                        break

                # If no existing manifest entry, create it (so resume is self-healing)
                if existing is None:
                    try:
                        h_meta = sha256_file(collector_out_dir / "meta.json")
                        h_state = sha256_file(collector_out_dir / "state.csv")
                        h_index = sha256_file(collector_out_dir / "index.csv")
                        bytes_total = dir_size_bytes(collector_out_dir)
                    except Exception:
                        h_meta = h_state = h_index = None
                        bytes_total = None

                    entry = {
                        "episode_id": episode_id,
                        "seed": seed,
                        "map": args.town_map,
                        "weather": args.weather,
                        "task_name": args.task_name,
                        "frames_target": args.frames,
                        "fixed_dt": args.fixed_dt,
                        "modalities": enabled_modalities + ["imu", "gnss"],
                        "status": "ok",
                        "returncode": 0,
                        "attempts": 0,
                        "started_utc": None,
                        "ended_utc": None,
                        "collector_output_relpath": collector_rel,
                        "episode_relpath": normalized_rel,
                        "verify": rep,
                        "sha256_meta": h_meta,
                        "sha256_state": h_state,
                        "sha256_index": h_index,
                        "bytes_total": bytes_total,
                        "updated_utc": utcnow_iso(),
                    }
                    upsert_episode(manifest, entry)
                    manifest["updated_utc"] = utcnow_iso()
                    safe_write_json(manifest_path, manifest)

                    ensure_symlink(normalized_link, collector_out_dir)
                    print(f"[SKIP] {collector_name} verify OK (manifest entry created)")
                    continue

                # Otherwise: strict integrity check vs existing entry
                hash_ok = True
                size_ok = True
                try:
                    h_meta = sha256_file(collector_out_dir / "meta.json")
                    h_state = sha256_file(collector_out_dir / "state.csv")
                    h_index = sha256_file(collector_out_dir / "index.csv")
                    if existing.get("sha256_meta") and existing["sha256_meta"] != h_meta:
                        hash_ok = False
                    if existing.get("sha256_state") and existing["sha256_state"] != h_state:
                        hash_ok = False
                    if existing.get("sha256_index") and existing["sha256_index"] != h_index:
                        hash_ok = False
                except Exception:
                    hash_ok = False

                try:
                    bytes_total = dir_size_bytes(collector_out_dir)
                    if existing.get("bytes_total") and int(existing["bytes_total"]) != int(bytes_total):
                        size_ok = False
                except Exception:
                    size_ok = False

                if hash_ok and size_ok:
                    ensure_symlink(normalized_link, collector_out_dir)
                    print(f"[SKIP] {collector_name} verify+integrity OK")
                    continue
                else:
                    print(f"[RESUME] {collector_name} exists but integrity mismatch (hash_ok={hash_ok}, size_ok={size_ok}) -> re-collect")

        # Clean output folder to avoid collector "dir exists" errors
        if collector_out_dir.exists():
            shutil.rmtree(collector_out_dir)

        log_path = logs_dir / f"{norm_name}.log"
        attempt = 0
        status = "fail"
        last_rc = None
        last_rep = None
        started_utc = utcnow_iso()

        while attempt <= int(args.retries):
            attempt += 1
            print(f"[RUN] {norm_name} attempt {attempt}/{int(args.retries)+1} seed={seed}")

            if not wait_for_server(args.host, args.port, attempts=args.wait_server_attempts, sleep_s=args.wait_server_sleep):
                print(f"[WARN] server no responde antes de {norm_name}")
                if args.docker_restart_on_fail and args.docker_container:
                    run_cmd(["docker", "restart", args.docker_container])
                    wait_for_server(args.host, args.port, attempts=args.wait_server_attempts, sleep_s=args.wait_server_sleep)

            cmd = [sys.executable, str(collect_script)]
            cmd += ["--host", args.host, "--port", str(args.port)]
            cmd += ["--cd-root", str(shim_root)]
            cmd += ["--episode-id", str(episode_id)]
            cmd += ["--map-name", args.town_map]
            cmd += ["-N", str(args.frames)]
            cmd += ["--fixed-dt", str(args.fixed_dt)]
            cmd += ["--seed", str(seed)]

            if args.weather:
                cmd += ["--weather", args.weather]
            if args.semseg:
                cmd += ["--semseg"]
            if args.lidar:
                cmd += ["--lidar"]

            # Force overwrite for safety (collector error if dir exists)
            cmd += ["--overwrite"]
            cmd += ["--park-server"]

            # Extra args forwarded to collector (REMAINDER; must be last at CLI)
            if args.collect_extra_args:
                cmd += list(args.collect_extra_args)

            env = os.environ.copy()
            env["PYTHONHASHSEED"] = "0"
            timeout = None if int(args.episode_timeout) <= 0 else int(args.episode_timeout)

            rc = run_cmd(cmd, log_path=log_path, env=env, timeout=timeout)
            last_rc = rc

            rep = verify_episode_outputs(collector_out_dir, args.frames, enabled_modalities, strict=args.strict_verify)
            last_rep = rep

            soft_ok = bool(args.accept_verified_nonzero_rc) and (rc != 0) and bool(rep["ok"])

            if rc == 0 and rep["ok"]:
                status = "ok"
                ensure_symlink(normalized_link, collector_out_dir)
                break

            if soft_ok:
                status = "ok"
                ensure_symlink(normalized_link, collector_out_dir)
                print(f"[SOFT-OK] {norm_name} rc={rc} verify_ok={rep['ok']} missing={rep.get('missing')} counts={rep.get('counts')} notes={rep.get('notes')}")
                break

            print(f"[FAIL] {norm_name} rc={rc} verify_ok={rep['ok']} missing={rep.get('missing')} counts={rep.get('counts')} notes={rep.get('notes')}")

            if args.docker_restart_on_fail and args.docker_container:
                print(f"[RECOVERY] docker restart {args.docker_container}")
                run_cmd(["docker", "restart", args.docker_container])
                wait_for_server(args.host, args.port, attempts=args.wait_server_attempts, sleep_s=args.wait_server_sleep)

            if collector_out_dir.exists():
                shutil.rmtree(collector_out_dir)
            time.sleep(2.0)

        ended_utc = utcnow_iso()

        entry = {
            "episode_id": episode_id,
            "seed": seed,
            "map": args.town_map,
            "weather": args.weather,
            "task_name": args.task_name,
            "frames_target": args.frames,
            "fixed_dt": args.fixed_dt,
            "modalities": enabled_modalities + ["imu", "gnss"],
            "status": status,
            "returncode": last_rc,
            "attempts": attempt,
            "soft_ok_nonzero_rc": bool(status == "ok" and (last_rc is not None) and int(last_rc) != 0),
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "collector_output_relpath": collector_rel,
            "episode_relpath": normalized_rel,
            "verify": last_rep if last_rep else {},
            "updated_utc": utcnow_iso(),
        }

        if status == "ok" and collector_out_dir.exists():
            try:
                entry["sha256_meta"] = sha256_file(collector_out_dir / "meta.json")
                entry["sha256_state"] = sha256_file(collector_out_dir / "state.csv")
                entry["sha256_index"] = sha256_file(collector_out_dir / "index.csv")
                entry["bytes_meta"] = file_size_bytes(collector_out_dir / "meta.json")
                entry["bytes_state"] = file_size_bytes(collector_out_dir / "state.csv")
                entry["bytes_index"] = file_size_bytes(collector_out_dir / "index.csv")
                entry["bytes_total"] = dir_size_bytes(collector_out_dir)
            except Exception:
                pass

        post_health = try_carla_health_check(args.host, args.port, timeout_s=2.0)
        if post_health is not None:
            entry["post_health"] = post_health

        upsert_episode(manifest, entry)
        manifest["updated_utc"] = utcnow_iso()
        safe_write_json(manifest_path, manifest)

        if status != "ok":
            print(f"[STOP] episodio {norm_name} falló tras retries. Manifest: {manifest_path}")
            return 3

    # Post: global index + splits
    if args.write_global_index:
        out_index = dataset_root / GLOBAL_INDEX_NAME
        regenerate_global_index(dataset_root, manifest, out_index)
        print("[OK] Global index:", out_index)

    if args.make_splits:
        s = make_episode_splits(dataset_root, manifest, args.split_seed, args.train_p, args.val_p, args.test_p)
        print("[OK] splits:", s)

    # Post: dataset report
    report_path = dataset_root / REPORT_NAME
    write_dataset_report(dataset_root, manifest, report_path)
    print("[OK] Dataset report:", report_path)

    print("[DONE] manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
