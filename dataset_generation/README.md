# Dataset Generation

This directory will contain the audited scripts and configuration files required to regenerate the dataset.

Files are intentionally not added in the initial repository scaffold. Before publication, scripts should be reviewed for:

- absolute local paths;
- machine-specific assumptions;
- unpublished credentials or tokens;
- dependencies and environment names;
- simulator version assumptions;
- output locations that could overwrite frozen data.

The generation pipeline should document the simulator version, container image digest, deterministic seeds, town/domain definitions, spatial slot selection, sensor configuration, and synchronization settings.
