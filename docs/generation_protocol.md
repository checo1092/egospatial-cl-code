
# Generation Protocol

EgoSpatial-CL v1.0.0 uses a frozen schedule with 490 episodes across 7 towns, 7 compact domains, and 10 spatial slots per town/domain pair.

Core generation parameters:

- fixed simulation step: `0.05` seconds.
- capture stride: `10` simulation ticks.
- samples per episode: `1000`.
- ego vehicle: `vehicle.tesla.model3`.
- autopilot enabled through the CARLA Traffic Manager.
- seed rule: `seed = 12345 + 17 * town_local_episode_id`.

The episode-level public schedule is `configs/egospatial_cl_v1_schedule.json`. The executable schedule for `dataset_generation/run_domain_schedule.py` is `configs/egospatial_cl_v1_domain_schedule.json`. Domain definitions are stored in `configs/domains.json`; spatial slots and splits are stored in `configs/spatial_splits.json`.

A full regeneration requires CARLA 0.9.15, the additional maps used by the selected towns, and a matching Python client environment.
