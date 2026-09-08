# EgoSpatial-CL Code

This repository contains the code and documentation used to reproduce the dataset release:

**EgoSpatial-CL: A Spatially Controlled Multimodal Dataset for Continual Learning in Simulated Autonomous Navigation**

The dataset is intended for public release as `egospatial-cl` version `v1.0.0`.

## Status

This repository is being prepared for the Scientific Data dataset release. The initial public repository contains only the audited structure and release documentation. Dataset-generation scripts will be added after path, dependency, and licensing review.

## Scope

Current scope:

- document the reproducible dataset generation pipeline;
- preserve the software environment and configuration assumptions;
- provide a stable code repository URL for the dataset paper;
- separate dataset release material from later benchmark and continual-learning code.

Out of scope for the initial release:

- trained model checkpoints;
- completed experiment outputs;
- raw dataset shards;
- benchmark results for KAN or continual-learning methods.

## Dataset Summary

The frozen `v1.0.0` dataset contains:

- 7 simulated towns;
- 7 compact domain conditions;
- 10 spatial slots per town/domain pair;
- 490 episodes;
- 490,000 synchronized samples;
- 392/49/49 train/validation/test episode split.

The data were generated with CARLA 0.9.15 in a containerized, synchronous simulation setup. CARLA is cited here as the simulator used to generate the data, not as part of the dataset title.

## Repository Layout

```text
dataset_generation/   Reproducible dataset generation scripts and configs, added after audit.
metadata/             Dataset metadata schemas, manifests, split descriptions, and release helpers.
benchmark/            Placeholder for later benchmark loaders, protocols, and evaluation scripts.
docs/                 Reproducibility, environment, and release documentation.
```

## Release Policy

The dataset and code releases are versioned separately:

- dataset: `v1.0.0`, archived through the dataset repository DOI;
- code: release DOI to be minted after the generation scripts are audited and frozen.

The first code DOI should be minted only after the reproducibility scripts are complete enough to support the Scientific Data submission.

## License

Code in this repository is released under the MIT License unless otherwise stated.

Dataset files are expected to be released separately under Creative Commons Attribution 4.0 International (CC BY 4.0). Third-party software and simulator assets remain governed by their respective upstream licenses.

## Citation

Please cite the dataset DOI once the public `v1.0.0` dataset DOI has been minted. A provisional citation file is included in `CITATION.cff` and will be updated before the code release DOI is minted.
