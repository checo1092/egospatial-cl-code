
#!/usr/bin/env python3
"""Verify SHA256 checksums for downloaded EgoSpatial-CL release files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_lfs_oid(path: Path) -> str | None:
    if path.stat().st_size > 1024:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            return line.split(":", 1)[1].strip().lower()
    return None


def iter_checksums(path: Path):
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid checksum line {line_no}: {raw!r}")
        digest, relpath = parts
        yield line_no, digest.lower(), relpath.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--include", action="append", default=[], help="Optional substring filter; may be repeated.")
    parser.add_argument("--strict", action="store_true", help="Fail when an included file is missing.")
    args = parser.parse_args()

    checked = 0
    missing = 0
    failed = 0
    for line_no, expected, relpath in iter_checksums(args.checksums):
        if args.include and not any(token in relpath for token in args.include):
            continue
        path = args.root / relpath
        if not path.exists():
            missing += 1
            print(f"MISSING line={line_no} path={relpath}")
            if args.strict:
                failed += 1
            continue
        lfs_oid = git_lfs_oid(path)
        actual = lfs_oid if lfs_oid is not None else sha256_file(path)
        checked += 1
        if actual != expected:
            failed += 1
            print(f"FAIL path={relpath} expected={expected} actual={actual}")
        else:
            status = "OK_LFS_POINTER" if lfs_oid is not None else "OK"
            print(f"{status} path={relpath}")

    print(f"checked={checked} missing={missing} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
