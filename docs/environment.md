
# Environment

The frozen EgoSpatial-CL v1.0.0 dataset was generated with CARLA 0.9.15 in synchronous mode.

Tested generation environment:

- Ubuntu 24.04.2 LTS host.
- NVIDIA GPU execution, originally using an RTX 2080 Ti.
- CARLA 0.9.15 Docker image digest: `sha256:1b5417ab03a91d2bad581d41fabeb91a8f3836130d7a112bcc8b4d2dac7dcfcd`.
- CARLA RPC endpoint: `127.0.0.1:2000`.
- Python 3.7 CARLA client environment for generation scripts.
- Python 3.10-compatible environment for validation and loading examples.

Additional maps are not vendored in this repository. Before building the full-map
container, obtain `AdditionalMaps_0.9.15.tar.gz` from the official CARLA 0.9.15
release assets and place it next to the import Dockerfile:

```text
docker/fullmaps_import/AdditionalMaps_0.9.15.tar.gz
```

The file is intentionally ignored by Git. It is used only as local Docker build
context for `docker/fullmaps_import/Dockerfile`.

Build order:

```bash
docker build -t carla:0.9.15-xdg docker/carla915-xdg
docker build -t carla:0.9.15-xdg-extra-maps docker/fullmaps_import
```

## CARLA Server Startup

The generation scripts expect a CARLA server reachable at `127.0.0.1:2000`. A known-good Docker startup pattern is:

```bash
docker run -d \
  --name carla-server-gpu0 \
  --gpus "device=0" \
  --net=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e SDL_AUDIODRIVER=dummy \
  carla:0.9.15-xdg-extra-maps \
  bash -lc './CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=0'
```

Optional local mounts may be added for data or log persistence:

```bash
-v /path/to/local/data:/workspace/data \
-v /path/to/local/logs:/workspace/logs
```

Check RPC readiness before running generation scripts:

```bash
python - <<'PY'
import time
import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)

for _ in range(60):
    try:
        world = client.get_world()
        print("READY", world.get_map().name)
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("CARLA RPC did not become ready")
PY
```
