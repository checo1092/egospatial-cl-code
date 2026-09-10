
# Reproducibility

Use the dataset repository metadata to validate the frozen public release:

- `benchmark_manifest.json`
- `spatial_splits.json`
- `shards_manifest.csv`
- `checksums.sha256`

Validate release metadata:

```bash
python validation/validate_release.py \
  --manifest /path/to/benchmark_manifest.json \
  --shards-manifest /path/to/shards_manifest.csv
```

Validate one downloaded shard:

```bash
python validation/validate_release.py \
  --manifest /path/to/benchmark_manifest.json \
  --tar /path/to/shards/Town06/val/Town06_val_000000.tar
```

Validate checksums:

```bash
python validation/verify_checksums.py \
  --checksums /path/to/checksums.sha256 \
  --root /path/to/downloaded/release \
  --include Town06_val_000000.tar
```

Run a short dynamic smoke test only after the CARLA RPC readiness check passes. Use a temporary output directory and a reduced frame count; the output is not part of the frozen dataset release.

```bash
python dataset_generation/run_episodes.py \
  --cd-root /path/to/egospatial-cl-code \
  --data-root /path/to/temporary/output \
  --dataset-name egospatial_cl_smoke \
  --host 127.0.0.1 \
  --port 2000 \
  --map Town10HD_Opt \
  --weather clear \
  --task-name smoke_Town10HD_Opt_clear \
  --num-episodes 1 \
  --start-episode-id 0 \
  --seed0 12345 \
  --seed-step 17 \
  --frames 20 \
  --fixed-dt 0.05 \
  --semseg \
  --lidar \
  --overwrite \
  --retries 0 \
  --episode-timeout 600 \
  --accept-verified-nonzero-rc \
  --collect-extra-args \
  --autopilot \
  --tm-port 8000 \
  --warmup-ticks 20 \
  --tm-distance-to-leading 3.5 \
  --tm-speed-diff-pct 60 \
  --tm-ignore-lights-pct 0 \
  --tm-ignore-signs-pct 0 \
  --tm-ignore-walkers-pct 0 \
  --post-load-sleep 2.0 \
  --capture-stride 10 \
  --shutdown-sleep 4.0 \
  --park-fixed-dt 0.05 \
  --rpc-timeout 30.0 \
  --sensor-timeout 20.0 \
  --map-load-warmup-ticks 40 \
  --num-vehicles 0 \
  --num-walkers 0 \
  --spawn-index 97
```

The Dataset Viewer is intentionally disabled for v1.0.0 because the release uses episode-level tar shards rather than a WebDataset-native sample layout. Do not repack the frozen release solely to enable the viewer.
