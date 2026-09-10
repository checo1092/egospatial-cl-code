
#!/usr/bin/env python3
"""Smoke-test the EgoSpatial-CL PyTorch loader on an extracted dataset copy."""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader

from pytorch_dataset import EgoSpatialCLFrameDataset, collate_egospatial_cl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--town", action="append", dest="towns")
    parser.add_argument("--domain-name", action="append", dest="domain_names")
    parser.add_argument("--frame-stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    ds = EgoSpatialCLFrameDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split=args.split,
        towns=args.towns,
        domain_names=args.domain_names,
        modalities=("rgb",),
        frame_stride=args.frame_stride,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_egospatial_cl)
    batch = next(iter(loader))
    print(f"samples={len(ds)} rgb_shape={tuple(batch['rgb'].shape)} target_shape={tuple(batch['target'].shape)}")
    print(f"first_meta={batch['meta'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
