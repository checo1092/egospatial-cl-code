# Changelog

## 1.0.0 — 2026-11-12

First tagged release of the code accompanying EgoSpatial-CL.

## Included

- CARLA 0.9.15 collection scripts, Docker definitions and the generation schedule for seven towns and seven weather/traffic domains.
- Fixed starting-slot configurations and tools for manifest construction, release-metadata validation and checksum verification.
- A PyTorch loading example and a synchronized multimodal viewer with PNG/GIF export.
- Scripts for publication figures and documentation of the generation environment, protocol and reuse.
- Citation metadata for the six authors.

Associated dataset: [EgoSpatial-CL](https://doi.org/10.57967/hf/11602), version 1.0.0, with 490 episodes and 490,000 synchronized samples. Dataset files are distributed separately through Hugging Face.

## Requirements and validation

See [the environment documentation](https://github.com/checo1092/egospatial-cl-code/blob/v1.0.0/docs/environment.md) for CARLA generation requirements and the dependency files for the viewer and figure scripts.

Release checks passed: nine viewer tests using synthetic episodes, compilation of all tracked Python sources, dataset/shard metadata validation and generation-schedule dry run. Live CARLA collection and model training were not rerun for this release.

## License

Code: MIT. Dataset and dataset-derived images: CC BY 4.0. CARLA and its assets retain their upstream licenses.
