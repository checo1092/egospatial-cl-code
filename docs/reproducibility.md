# Reproducibility Notes

## Dataset Generation Environment

The frozen dataset release was generated with:

- simulator: CARLA 0.9.15;
- execution mode: synchronous simulation;
- fixed simulation step: 0.05 seconds;
- capture stride: 10 simulation frames;
- samples per episode: 1,000;
- deterministic seed rule: `seed = 12345 + 17 * episode_id`.

The CARLA Docker image digest recorded for the dataset generation audit is:

```text
sha256:1b5417ab03a91d2bad581d41fabeb91a8f3836130d7a112bcc8b4d2dac7dcfcd
```

## Dataset Release

The dataset release should be cited separately from this code repository. The dataset DOI must only be minted after metadata, versioning, license, manifests, and checksums are frozen.

## Code Release

The first code release DOI should be minted after the generation scripts are:

- cleaned of local absolute paths;
- documented with dependency and environment setup;
- validated against the frozen dataset manifests;
- tagged with a stable version.
