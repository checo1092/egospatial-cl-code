#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_domain_schedule.py — EgoSpatial-CL domain schedule driver

Goal:
- Read the frozen JSON schedule that defines each episode to generate:
  town, domain, split, spawn point, seed, and collector settings.
- Generate the dataset by invoking the episode runner once per scheduled task.
- Persist provenance: copy applied schedule + sha256 into dataset root.
- Generate per-task episode splits (train/val/test) under <dataset_root>/splits/.

Designed for:
- EgoSpatial-CL dataset generation with CARLA 0.9.15, headless rendering,
  synchronous simulation, and fixed time steps.
- Python 3.7+ (no 3.8-only syntax).

Usage example:
  python $CD_ROOT/dataset_generation/run_domain_schedule.py \
    --cd-root $CD_ROOT \
    --schedule $CD_ROOT/configs/egospatial_cl_v1_domain_schedule.json \
    --dataset-name egospatial_cl_v1

Notes:
- This script does NOT modify run_episodes.py.
- Extra collector args are passed via run_episodes.py's --collect-extra-args (MUST BE LAST).
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    # Canonical (stable) serialization: sort keys, no whitespace variance
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return s.encode("utf-8")


def safe_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(tmp), str(path))


def safe_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(str(tmp), str(path))


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        # allow a single string, but discourage; user should pass list
        return [x]
    return [str(x)]


def get_required(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError("Missing required key '{}' in {}".format(key, ctx))
    return d[key]


def merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge b over a (dicts only)."""
    out = dict(a) if a else {}
    if b:
        out.update(b)
    return out


def resolve_task_fields(task: Dict[str, Any], defaults: Dict[str, Any], idx: int) -> Dict[str, Any]:
    # task_id
    task_id = int(task.get("task_id", idx))
    task_name = task.get("task_name", "Task{:02d}".format(task_id))

    num_episodes = int(get_required(task, "num_episodes", "task {}".format(task_name)))

    domain = merge_dict(defaults.get("domain", {}), task.get("domain", {}))
    # also accept top-level "map"/"weather" in domain OR directly in defaults (legacy)
    map_name = domain.get("map", defaults.get("map", "Town10HD_Opt"))
    weather = domain.get("weather", defaults.get("weather", None))

    # training collection params (can be overridden per task)
    frames = int(task.get("frames", defaults.get("frames", 300)))
    fixed_dt = float(task.get("fixed_dt", defaults.get("fixed_dt", 0.05)))
    seed0 = task.get("seed0", None)

    modalities = merge_dict(defaults.get("modalities", {}), task.get("modalities", {}))
    semseg = bool(modalities.get("semseg", False))
    lidar = bool(modalities.get("lidar", False))

    # networking
    host = str(task.get("host", defaults.get("host", "127.0.0.1")))
    port = int(task.get("port", defaults.get("port", 2000)))

    # extras forwarded to collect_episode.py (via run_episodes.py --collect-extra-args)
    extra_defaults = as_list(defaults.get("collect_extra_args", []))
    extra_task = as_list(task.get("collect_extra_args", []))
    collect_extra_args = extra_defaults + extra_task

    return {
        "task_id": task_id,
        "task_name": task_name,
        "num_episodes": num_episodes,
        "domain": {"map": map_name, "weather": weather, **{k: v for k, v in domain.items() if k not in ["map", "weather"]}},
        "frames": frames,
        "fixed_dt": fixed_dt,
        "seed0": None if seed0 is None else int(seed0),
        "modalities": {"semseg": semseg, "lidar": lidar},
        "host": host,
        "port": port,
        "collect_extra_args": collect_extra_args,
    }


def build_run_episodes_cmd(
    py_exe: str,
    run_episodes_path: Path,
    cd_root: Path,
    data_root: Optional[Path],
    dataset_name: str,
    task_resolved: Dict[str, Any],
    start_episode_id: int,
    seed0_task: int,
    seed_step: int,
    run_episodes_extra_args: Optional[List[str]] = None,
) -> List[str]:
    cmd = [py_exe, str(run_episodes_path)]
    cmd += ["--cd-root", str(cd_root)]
    if data_root is not None:
        cmd += ["--data-root", str(data_root)]
    cmd += ["--dataset-name", str(dataset_name)]

    cmd += ["--host", task_resolved["host"], "--port", str(int(task_resolved["port"]))]

    cmd += ["--map", task_resolved["domain"]["map"]]
    weather = task_resolved["domain"].get("weather", None)
    if weather is not None:
        cmd += ["--weather", str(weather)]
    cmd += ["--task-name", str(task_resolved["task_name"])]

    cmd += ["--num-episodes", str(int(task_resolved["num_episodes"]))]
    cmd += ["--start-episode-id", str(int(start_episode_id))]

    cmd += ["--seed0", str(int(seed0_task))]
    cmd += ["--seed-step", str(int(seed_step))]

    cmd += ["--frames", str(int(task_resolved["frames"]))]
    cmd += ["--fixed-dt", str(float(task_resolved["fixed_dt"]))]

    if bool(task_resolved["modalities"].get("semseg", False)):
        cmd += ["--semseg"]
    if bool(task_resolved["modalities"].get("lidar", False)):
        cmd += ["--lidar"]

    # IMPORTANT:
    # run_episodes_extra_args belong to run_episodes.py, so they MUST go
    # before --collect-extra-args (which is argparse.REMAINDER).
    if run_episodes_extra_args:
        cmd += [str(x) for x in run_episodes_extra_args]

    extra = task_resolved.get("collect_extra_args", [])
    if extra:
        cmd += ["--collect-extra-args"] + list(extra)

    return cmd


def make_per_task_splits(dataset_root: Path, tasks_manifest: Dict[str, Any], split_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    splits_dir = dataset_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_seed = int(split_cfg.get("split_seed", 2026))
    train_p = float(split_cfg.get("train_p", 0.8))
    val_p = float(split_cfg.get("val_p", 0.1))
    test_p = float(split_cfg.get("test_p", 0.1))

    # write per task
    out = {"policy": "per_task", "split_seed": split_seed, "train_p": train_p, "val_p": val_p, "test_p": test_p, "tasks": []}

    for t in tasks_manifest["tasks"]:
        task_id = int(t["task_id"])
        task_name = t["task_name"]
        episode_ids = [int(x) for x in t.get("episode_ids_ok", [])]
        episode_ids = sorted(episode_ids)

        rng = random.Random(split_seed + task_id)
        shuffled = episode_ids[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_p))
        n_val = int(round(n * val_p))
        n_test = max(0, n - n_train - n_val)

        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:n_train + n_val + n_test]

        def write_ids(fname: str, ids: List[int]) -> str:
            p = splits_dir / fname
            with open(p, "w", encoding="utf-8") as f:
                for eid in ids:
                    f.write(str(int(eid)) + "\n")
            return str(p.relative_to(dataset_root))

        base = "task_{:02d}".format(task_id)
        rel_train = write_ids(base + "_train_episodes.txt", train)
        rel_val = write_ids(base + "_val_episodes.txt", val)
        rel_test = write_ids(base + "_test_episodes.txt", test)

        out["tasks"].append({
            "task_id": task_id,
            "task_name": task_name,
            "n_ok": len(episode_ids),
            "train": train,
            "val": val,
            "test": test,
            "files": {"train": rel_train, "val": rel_val, "test": rel_test},
        })

    safe_write_json(splits_dir / "task_splits_manifest.json", out)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--cd-root", default=str(repo_root))
    ap.add_argument("--data-root", default=None, help="Default: <cd-root>/data")
    ap.add_argument("--schedule", required=True, help="Path to the executable EgoSpatial-CL schedule JSON.")
    ap.add_argument("--dataset-name", required=True, help="Dataset name under <data-root>/")
    ap.add_argument("--run-episodes", default=None, help="Default: <cd-root>/dataset_generation/run_episodes.py")
    ap.add_argument("--py-exe", default=sys.executable, help="Python executable to invoke run_episodes.py")
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--clean", action="store_true", default=False, help="Delete <dataset_root> before running (DANGEROUS)")
    ap.add_argument("--continue-on-fail", action="store_true", default=False)

    # Optional CLI overrides (useful for smoke/debug without editing JSON)
    ap.add_argument("--override-frames", type=int, default=None)
    ap.add_argument("--override-fixed-dt", type=float, default=None)
    ap.add_argument("--override-host", default=None)
    ap.add_argument("--override-port", type=int, default=None)

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    cd_root = Path(args.cd_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else (cd_root / "data")
    dataset_root = (data_root / args.dataset_name).resolve()

    schedule_path = Path(args.schedule).expanduser().resolve()
    if not schedule_path.exists():
        print("[ERR] schedule not found:", schedule_path, file=sys.stderr)
        return 2

    run_episodes_path = Path(args.run_episodes).expanduser().resolve() if args.run_episodes else (cd_root / "dataset_generation" / "run_episodes.py")
    if not run_episodes_path.exists():
        print("[ERR] run_episodes.py not found:", run_episodes_path, file=sys.stderr)
        return 2

    schedule_obj = json.loads(schedule_path.read_text(encoding="utf-8"))
    schema_version = str(schedule_obj.get("schema_version", ""))
    if schema_version and schema_version not in ["0.1", "0.2", "1.0"]:
        print("[WARN] Unrecognized schema_version:", schema_version)

    defaults = dict(schedule_obj.get("defaults", {}))
    run_episodes_extra_args = as_list(defaults.get("run_episodes_extra_args", []))
    split_cfg = dict(schedule_obj.get("splits", {}))
    tasks_in = schedule_obj.get("tasks", [])
    if not isinstance(tasks_in, list) or len(tasks_in) == 0:
        print("[ERR] schedule.tasks must be a non-empty list", file=sys.stderr)
        return 2

    # CLI overrides
    if args.override_frames is not None:
        defaults["frames"] = int(args.override_frames)
    if args.override_fixed_dt is not None:
        defaults["fixed_dt"] = float(args.override_fixed_dt)
    if args.override_host is not None:
        defaults["host"] = str(args.override_host)
    if args.override_port is not None:
        defaults["port"] = int(args.override_port)

    seed0_global = int(defaults.get("seed0", 12345))
    seed_step = int(defaults.get("seed_step", defaults.get("seed-step", 17)))
    if seed_step == 0:
        raise ValueError("seed_step must be non-zero")

    # Prepare dataset root
    if args.clean and dataset_root.exists():
        print("[CLEAN] rm -rf", dataset_root)
        if not args.dry_run:
            shutil.rmtree(dataset_root)

    dataset_root.mkdir(parents=True, exist_ok=True)

    # Freeze applied schedule + hash
    resolved_tasks: List[Dict[str, Any]] = []
    start_episode_id = 0
    for idx, t in enumerate(tasks_in):
        tr = resolve_task_fields(t, defaults, idx)
        tr["start_episode_id"] = int(start_episode_id)
        if tr.get("seed0") is not None:
            tr["seed0_task"] = int(tr["seed0"])
        else:
            tr["seed0_task"] = int(seed0_global + start_episode_id * seed_step)
        resolved_tasks.append(tr)
        start_episode_id += int(tr["num_episodes"])

    applied = {
        "applied_utc": utcnow_iso(),
        "driver": {
            "script": "run_domain_schedule.py",
            "py_exe": args.py_exe,
            "run_episodes": str(run_episodes_path),
            "cd_root": str(cd_root),
            "data_root": str(data_root),
            "dataset_root": str(dataset_root),
        },
        "source_schedule_path": str(schedule_path),
        "schedule": schedule_obj,
        "resolved": {
            "seed0_global": seed0_global,
            "seed_step": seed_step,
            "tasks": resolved_tasks,
        }
    }
    schedule_hash = sha256_bytes(canonical_json_bytes(schedule_obj))
    applied_hash = sha256_bytes(canonical_json_bytes(applied))

    safe_write_json(dataset_root / "domain_schedule.applied.json", applied)
    safe_write_text(dataset_root / "domain_schedule.sha256.txt", schedule_hash + "\n")
    safe_write_text(dataset_root / "domain_schedule_applied.sha256.txt", applied_hash + "\n")

    # Run tasks
    print("[INFO] dataset_root:", dataset_root)
    print("[INFO] schedule_sha256:", schedule_hash)
    print("[INFO] tasks:", len(resolved_tasks))

    rc_any = 0
    for tr in resolved_tasks:
        cmd = build_run_episodes_cmd(
            py_exe=str(args.py_exe),
            run_episodes_path=run_episodes_path,
            cd_root=cd_root,
            data_root=data_root,
            dataset_name=args.dataset_name,
            task_resolved=tr,
            start_episode_id=int(tr["start_episode_id"]),
            seed0_task=int(tr["seed0_task"]),
            seed_step=seed_step,
            run_episodes_extra_args=run_episodes_extra_args,
        )
        print("\n[RUN] task_id={:02d} name={} start_ep={} num_eps={} seed0_task={} weather={} map={}".format(
            int(tr["task_id"]), tr["task_name"], int(tr["start_episode_id"]), int(tr["num_episodes"]),
            int(tr["seed0_task"]), tr["domain"].get("weather", None), tr["domain"]["map"]
        ))
        print("      CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        rc = subprocess.call(cmd)
        if rc == 2:
            print("[FATAL] run_episodes reported server unreachable; aborting schedule.")
            return 2
        if rc != 0:
            print("[ERR] task failed rc={}".format(rc), file=sys.stderr)
            rc_any = rc
            if not args.continue_on_fail:
                break

    if args.dry_run:
        print("\n[DRY-RUN] done (no commands executed).")
        return 0

    if rc_any != 0:
        print("[ERR] schedule execution failed rc={}".format(rc_any), file=sys.stderr)
        return int(rc_any)

    # Build tasks_manifest.json by reading episodes_manifest.json produced by runner
    manifest_path = dataset_root / "episodes_manifest.json"
    if not manifest_path.exists():
        print("[ERR] episodes_manifest.json not found at {}".format(manifest_path), file=sys.stderr)
        return 3

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])

    # Map: task_name -> list of OK episode ids
    ok_by_task: Dict[str, List[int]] = {}
    for e in episodes:
        if e.get("status") != "ok":
            continue
        tname = e.get("task_name") or "UNKNOWN"
        ok_by_task.setdefault(tname, []).append(int(e["episode_id"]))

    tasks_manifest = {
        "created_utc": utcnow_iso(),
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "schedule_sha256": schedule_hash,
        "tasks": [],
    }

    # keep task order as schedule order
    for tr in resolved_tasks:
        tname = tr["task_name"]
        ok_ids = sorted(ok_by_task.get(tname, []))
        tasks_manifest["tasks"].append({
            "task_id": int(tr["task_id"]),
            "task_name": tname,
            "domain": tr["domain"],
            "start_episode_id": int(tr["start_episode_id"]),
            "num_episodes_target": int(tr["num_episodes"]),
            "episode_ids_ok": ok_ids,
            "n_ok": len(ok_ids),
        })

    safe_write_json(dataset_root / "tasks_manifest.json", tasks_manifest)

    # Per-task splits
    split_cfg_final = {
        "split_seed": int(split_cfg.get("split_seed", 2026)),
        "train_p": float(split_cfg.get("train_p", 0.8)),
        "val_p": float(split_cfg.get("val_p", 0.1)),
        "test_p": float(split_cfg.get("test_p", 0.1)),
    }
    splits_manifest = make_per_task_splits(dataset_root, tasks_manifest, split_cfg_final)

    # Patch dataset_report.json to include schedule provenance (best effort)
    report_path = dataset_root / "dataset_report.json"
    if report_path.exists():
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            rep["cl_benchmark"] = {
                "schedule_path": str(schedule_path),
                "schedule_sha256": schedule_hash,
                "domain_schedule_applied": "domain_schedule.applied.json",
                "tasks_manifest": "tasks_manifest.json",
                "task_splits_manifest": "splits/task_splits_manifest.json",
                "updated_utc": utcnow_iso(),
            }
            safe_write_json(report_path, rep)
        except Exception as ex:
            print("[WARN] could not patch dataset_report.json:", ex, file=sys.stderr)

    print("\n[OK] wrote:")
    print(" -", dataset_root / "domain_schedule.applied.json")
    print(" -", dataset_root / "domain_schedule.sha256.txt")
    print(" -", dataset_root / "tasks_manifest.json")
    print(" -", dataset_root / "splits" / "task_splits_manifest.json")
    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
