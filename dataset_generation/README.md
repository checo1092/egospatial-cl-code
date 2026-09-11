# Dataset Generation

This directory contains the public generation scripts for EgoSpatial-CL v1.0.0.

## Script Roles

- `collect_episode.py`: low-level CARLA collector for one episode. It connects to
  the simulator, configures sensors and synchronous stepping, captures RGB,
  semantic segmentation, LiDAR, vehicle state, and per-frame indices.
- `run_episodes.py`: batch runner for multiple episodes. It invokes
  `collect_episode.py`, assigns episode ids and seeds, verifies outputs, and
  writes episode-level manifests and reports.
- `run_domain_schedule.py`: schedule-level driver. It reads the frozen executable
  schedule and invokes `run_episodes.py` once per scheduled task.

The execution hierarchy is:

```text
configs/egospatial_cl_v1_domain_schedule.json
  -> run_domain_schedule.py
  -> run_episodes.py
  -> collect_episode.py
  -> episode outputs and manifests
```

## Environment

Use the CARLA 0.9.15 Python client environment for generation. See
`docs/environment.md` for the Docker image, additional-map setup, server startup,
and RPC readiness check.
