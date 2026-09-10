#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
CARLA 0.9.15 — Dataset-first Episode Collector (stable teardown + optional park)

Features:
- Deterministic synchronous capture with fixed delta (sync mode).
- RGB (+ optional SemSeg) + optional LiDAR + IMU + GNSS written per episode.
- Domain controls:
  - weather presets: current/clear/cloudy/wet/hardrain (+ some *_sunset variants)
  - explicit overrides: sun altitude/azimuth, fog params, wetness, precipitation, wind, cloudiness
- Autopilot via CARLA Traffic Manager with conservative driving settings.
- Optional NPCs:
  - Spawn background traffic vehicles (autopilot via TM)
  - Spawn pedestrians (walkers) with AI controllers
  - All deterministic via --npc-seed
- Robust teardown: stop/destroy sensors, NPC controllers/actors, ego, TM sync disable.
- Optional "park server" after episode to reduce idle GPU: sync + no_rendering_mode=True.

IMPORTANT:
- Cameras require rendering ON during capture (no_rendering_mode=False).
- This collector is designed to be called by the public runner, which passes a cd-root shim:
  <dataset_root>/_cdroot/data -> <dataset_root>. So we write to <cd_root>/data/episode_XXXX.
"""

import os
import sys
import json
import csv
import time
import queue
import random
import argparse
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

carla = None


# ----------------------------
# Utilities
# ----------------------------

def mkdirp(path: str):
    os.makedirs(path, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def transform_to_dict(t: carla.Transform) -> Dict[str, Any]:
    return {
        "location": {"x": float(t.location.x), "y": float(t.location.y), "z": float(t.location.z)},
        "rotation": {"pitch": float(t.rotation.pitch), "yaw": float(t.rotation.yaw), "roll": float(t.rotation.roll)},
    }


def weather_to_dict(w: carla.WeatherParameters) -> Dict[str, Any]:
    keys = [
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
        "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
        "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm"
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if hasattr(w, k):
            out[k] = float(getattr(w, k))
    return out


def carla_image_to_rgb_array(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
    return rgb


def save_png(rgb_array: np.ndarray, path: str):
    Image.fromarray(rgb_array).save(path, format="PNG", optimize=False)


def lidar_to_npy(lidar_meas: carla.LidarMeasurement) -> np.ndarray:
    pts = np.frombuffer(lidar_meas.raw_data, dtype=np.float32)
    return pts.reshape((-1, 4))  # x,y,z,intensity


def same_map(current_map_full: str, requested: str) -> bool:
    """
    current_map_full often: '/Game/Carla/Maps/Town10HD_Opt' or 'Carla/Maps/Town10HD_Opt'
    requested can be: 'Town10HD_Opt' or full path.
    """
    if not requested:
        return True
    if current_map_full == requested:
        return True
    if current_map_full.endswith("/" + requested) or current_map_full.endswith(requested):
        return True
    return False


# ----------------------------
# Sensor buffering
# ----------------------------

class SensorBuffer:
    def __init__(self, name: str, maxsize: int = 64):
        self.name = name
        self.q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)

    def put(self, data: Any) -> None:
        try:
            self.q.put_nowait(data)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(data)
            except queue.Full:
                pass

    def get_for_frame(self, frame: int, timeout_s: float):
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            try:
                item = self.q.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            if hasattr(item, "frame") and item.frame == frame:
                return item
        return None


def safe_listen(sensor, callback):
    def _wrap(data):
        try:
            callback(data)
        except Exception:
            pass
    sensor.listen(_wrap)


# ----------------------------
# CARLA Traffic Manager helpers for conservative autopilot settings.
# ----------------------------

def _tm_try(log, tm, fn_name: str, *args):
    if tm is None:
        return
    fn = getattr(tm, fn_name, None)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception as e:
        try:
            log.warning("TM call failed: %s(%s) -> %s", fn_name, ",".join(map(str, args)), str(e))
        except Exception:
            pass


def configure_tm_global(log, tm, args) -> None:
    if tm is None:
        return
    _tm_try(log, tm, "set_synchronous_mode", True)
    _tm_try(log, tm, "set_random_device_seed", int(args.seed))
    _tm_try(log, tm, "set_global_distance_to_leading_vehicle", float(args.tm_distance_to_leading))

    if bool(args.tm_hybrid_physics):
        _tm_try(log, tm, "set_hybrid_physics_mode", True)
        _tm_try(log, tm, "set_hybrid_physics_radius", float(args.tm_hybrid_radius))

    _tm_try(log, tm, "set_respawn_dormant_vehicles", False)


def configure_tm_for_vehicle(log, tm, veh, args, is_ego: bool) -> None:
    if tm is None or veh is None:
        return

    _tm_try(log, tm, "ignore_lights_percentage", veh, float(args.tm_ignore_lights_pct))
    _tm_try(log, tm, "ignore_signs_percentage", veh, float(args.tm_ignore_signs_pct))
    _tm_try(log, tm, "ignore_walkers_percentage", veh, float(args.tm_ignore_walkers_pct))

    speed_diff = float(args.tm_speed_diff_pct) if is_ego else float(args.tm_npc_speed_diff_pct)
    _tm_try(log, tm, "vehicle_percentage_speed_difference", veh, speed_diff)

    lane_change = bool(args.tm_auto_lane_change) if is_ego else bool(args.tm_npc_auto_lane_change)
    _tm_try(log, tm, "auto_lane_change", veh, lane_change)


# ----------------------------
# Weather presets + overrides
# ----------------------------

def set_weather_preset(world, preset_name: str, log=None):
    name = (preset_name or "current").lower().strip()
    if name == "current":
        return

    mapping = {
        "clear": ["ClearNoon"],
        "cloudy": ["CloudyNoon"],
        "wet": ["WetNoon"],
        "hardrain": ["HardRainNoon"],

        "clear_sunset": ["ClearSunset", "ClearNoon"],
        "cloudy_sunset": ["CloudySunset", "CloudyNoon"],
        "wet_sunset": ["WetSunset", "WetNoon"],
        "hardrain_sunset": ["HardRainSunset", "HardRainNoon"],

        "soft_rain": ["SoftRainNoon", "MidRainyNoon", "HardRainNoon"],
        "soft_rain_sunset": ["SoftRainSunset", "MidRainSunset", "HardRainSunset", "HardRainNoon"],
    }

    if name not in mapping:
        raise ValueError("Unsupported --weather '%s'." % name)

    for attr in mapping[name]:
        if hasattr(carla.WeatherParameters, attr):
            world.set_weather(getattr(carla.WeatherParameters, attr))
            if log:
                log.info("Weather preset: %s (CARLA.%s)", name, attr)
            return

    raise ValueError("No CARLA WeatherParameters preset found for '%s' in this build." % name)


def apply_weather_overrides(world, args, log=None):
    fields = {
        "sun_altitude_angle": args.sun_altitude_angle,
        "sun_azimuth_angle": args.sun_azimuth_angle,
        "fog_density": args.fog_density,
        "fog_distance": args.fog_distance,
        "fog_falloff": args.fog_falloff,
        "cloudiness": args.cloudiness,
        "precipitation": args.precipitation,
        "precipitation_deposits": args.precipitation_deposits,
        "wetness": args.wetness,
        "wind_intensity": args.wind_intensity,
    }
    if all(v is None for v in fields.values()):
        return None

    w = world.get_weather()
    applied = {}
    for k, v in fields.items():
        if v is None:
            continue
        if hasattr(w, k):
            setattr(w, k, float(v))
            applied[k] = float(v)

    world.set_weather(w)
    if log and applied:
        log.info("Weather overrides applied: %s", applied)
    return applied


# ----------------------------
# NPC spawning (vehicles + walkers)
# ----------------------------

def _try_world_call(log, world, fn_name: str, *args):
    fn = getattr(world, fn_name, None)
    if fn is None:
        return
    try:
        return fn(*args)
    except Exception as e:
        log.warning("World call failed: %s(%s) -> %s", fn_name, ",".join(map(str, args)), str(e))
        return


def spawn_npc_vehicles(log, world, tm, args, ego_spawn: carla.Transform, rng: random.Random) -> List[carla.Actor]:
    n = int(args.num_vehicles)
    if n <= 0:
        return []

    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter(args.npc_vehicle_filter)
    if not vehicle_bps:
        log.warning("No blueprints matched npc_vehicle_filter=%s", args.npc_vehicle_filter)
        return []

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        log.warning("No spawn points found; cannot spawn NPC vehicles.")
        return []

    npc_vehicles: List[carla.Actor] = []
    max_attempts = int(max(1, args.npc_vehicle_spawn_attempts))
    min_dist = float(args.npc_min_dist_to_ego)
    max_dist = args.npc_max_dist_to_ego
    max_dist2 = (float(max_dist) * float(max_dist)) if max_dist is not None else None

    def dist2(a: carla.Location, b: carla.Location) -> float:
        dx = float(a.x - b.x); dy = float(a.y - b.y); dz = float(a.z - b.z)
        return dx*dx + dy*dy + dz*dz

    ego_loc = ego_spawn.location

    idxs = list(range(len(spawn_points)))
    rng.shuffle(idxs)

    attempts = 0
    for si in idxs:
        if len(npc_vehicles) >= n:
            break
        sp = spawn_points[si]
        if dist2(sp.location, ego_loc) < (min_dist * min_dist):
            continue
        if max_dist2 is not None and dist2(sp.location, ego_loc) > max_dist2:
            continue

        for _ in range(max_attempts):
            bp = rng.choice(vehicle_bps)
            try:
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", "npc")
                if bp.has_attribute("color"):
                    color = rng.choice(bp.get_attribute("color").recommended_values)
                    bp.set_attribute("color", color)
            except Exception:
                pass

            actor = world.try_spawn_actor(bp, sp)
            attempts += 1
            if actor is None:
                continue

            try:
                actor.set_autopilot(True, int(args.tm_port))
            except Exception:
                pass
            configure_tm_for_vehicle(log, tm, actor, args, is_ego=False)

            npc_vehicles.append(actor)
            break

    log.info("NPC vehicles: requested=%d spawned=%d attempts=%d", n, len(npc_vehicles), attempts)
    return npc_vehicles


def spawn_walkers(
    log,
    world,
    args,
    rng: random.Random,
    ego_loc=None,
) -> Tuple[List[carla.Actor], List[carla.Actor]]:
    n = int(args.num_walkers)
    if n <= 0:
        return ([], [])

    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    if not walker_bps:
        log.warning("No walker blueprints found.")
        return ([], [])

    controller_bp = bp_lib.find("controller.ai.walker")

    if args.pedestrians_cross_factor is not None:
        _try_world_call(log, world, "set_pedestrians_cross_factor", float(args.pedestrians_cross_factor))

    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []

    spawn_attempts = int(max(1, args.npc_walker_spawn_attempts))
    min_dist = float(args.npc_min_dist_to_ego)
    max_dist = args.npc_max_dist_to_ego
    max_dist2 = (float(max_dist) * float(max_dist)) if max_dist is not None else None
    min_dist2 = float(min_dist) * float(min_dist)

    def _valid_walker_loc(wloc) -> bool:
        if wloc is None:
            return False
        if ego_loc is None:
            return True
        dx = float(wloc.x) - float(ego_loc.x)
        dy = float(wloc.y) - float(ego_loc.y)
        dz = float(wloc.z) - float(ego_loc.z)
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < min_dist2:
            return False
        if max_dist2 is not None and d2 > max_dist2:
            return False
        return True

    for _ in range(n):
        wloc = None
        for _ in range(spawn_attempts):
            cand = world.get_random_location_from_navigation()
            if _valid_walker_loc(cand):
                wloc = cand
                break

        if wloc is None:
            continue

        bp = rng.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        yaw = float(rng.uniform(-180.0, 180.0))
        tf = carla.Transform(wloc, carla.Rotation(yaw=yaw))

        walker = world.try_spawn_actor(bp, tf)
        if walker is None:
            continue
        walkers.append(walker)

        try:
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            controllers.append(controller)
        except Exception:
            try:
                walker.destroy()
            except Exception:
                pass

    try:
        world.tick()
    except Exception:
        pass

    min_s = float(args.walker_speed_min)
    max_s = float(args.walker_speed_max)
    for ctrl in controllers:
        try:
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            spd = float(rng.uniform(min_s, max_s))
            ctrl.set_max_speed(spd)
        except Exception:
            pass

    log.info("Walkers: requested=%d spawned=%d controllers=%d", n, len(walkers), len(controllers))
    return (walkers, controllers)
# ----------------------------
# Teardown + optional park
# ----------------------------

def apply_park_settings(world, fixed_dt: float):
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = float(fixed_dt)
    s.no_rendering_mode = True
    world.apply_settings(s)


def classic_teardown(log, world, tm, original_settings,
                     ego, sensors,
                     npc_vehicles, walkers, walker_controllers,
                     shutdown_sleep: float,
                     park_server: bool,
                     park_fixed_dt: float):
    log.info("Shutdown: stopping sensors...")
    for s in sensors:
        try:
            s.stop()
        except Exception:
            pass
    time.sleep(0.2)

    log.info("Shutdown: destroying sensors...")
    for s in sensors:
        try:
            s.destroy()
        except Exception:
            pass

    if walker_controllers:
        log.info("Shutdown: stopping walker controllers...")
        for c in walker_controllers:
            try:
                c.stop()
            except Exception:
                pass
        time.sleep(0.1)

    if walker_controllers:
        log.info("Shutdown: destroying walker controllers...")
        for c in walker_controllers:
            try:
                c.destroy()
            except Exception:
                pass

    if walkers:
        log.info("Shutdown: destroying walkers...")
        for w in walkers:
            try:
                w.destroy()
            except Exception:
                pass

    if npc_vehicles:
        log.info("Shutdown: destroying npc vehicles...")
        for v in npc_vehicles:
            try:
                try:
                    v.set_autopilot(False)
                except Exception:
                    pass
                v.destroy()
            except Exception:
                pass

    log.info("Shutdown: destroying ego...")
    try:
        if ego is not None:
            try:
                ego.set_autopilot(False)
            except Exception:
                pass
            ego.destroy()
    except Exception:
        pass

    log.info("Shutdown: disabling TM sync...")
    try:
        if tm is not None:
            tm.set_synchronous_mode(False)
    except Exception:
        pass

    if park_server:
        log.info("Shutdown: parking server (sync + no_rendering) to reduce idle GPU...")
        try:
            apply_park_settings(world, park_fixed_dt)
        except Exception:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass
    else:
        log.info("Shutdown: restoring original world settings...")
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    time.sleep(float(shutdown_sleep))
    log.info("Shutdown: done.")


# ----------------------------
# Main collector
# ----------------------------

def collect_episode(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("collector")

    cd_root = os.path.expandvars(os.path.expanduser(args.cd_root))
    out_root = os.path.join(cd_root, "data")
    mkdirp(out_root)

    episode_id = args.episode_id
    if episode_id is None:
        existing = [d for d in os.listdir(out_root) if d.startswith("episode_")]
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_")[1]))
            except Exception:
                pass
        episode_id = (max(nums) + 1) if nums else 0

    ep_name = "episode_%04d" % int(episode_id)
    ep_dir = os.path.join(out_root, ep_name)
    if os.path.exists(ep_dir) and not args.overwrite:
        raise RuntimeError("Episode directory exists: %s (use --overwrite or --episode-id new)" % ep_dir)

    images_dir = os.path.join(ep_dir, "images")
    labels_dir = os.path.join(ep_dir, "labels")
    lidar_dir = os.path.join(ep_dir, "lidar")
    mkdirp(ep_dir)
    mkdirp(images_dir)
    if args.semseg:
        mkdirp(labels_dir)
    if args.lidar:
        mkdirp(lidar_dir)

    meta_path = os.path.join(ep_dir, "meta.json")
    state_path = os.path.join(ep_dir, "state.csv")
    index_path = os.path.join(ep_dir, "index.csv")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.rpc_timeout)

    world = client.get_world()
    current_map = world.get_map().name
    log.info("Connected OK. Map: %s", current_map)

    if args.map_name and not same_map(current_map, args.map_name):
        log.info("Loading map: %s", args.map_name)
        world = client.load_world(args.map_name)
        time.sleep(float(args.post_load_sleep))
        time.sleep(1.0)
        current_map = world.get_map().name
        log.info("Loaded map: %s", current_map)
        # Extra stabilization after map load (Town03_Opt has been fragile)
        try:
            world = client.get_world()
        except Exception:
            pass
        log.info("Post-load wait_for_tick: %d ticks (async)...", int(args.post_load_wait_ticks))
        for _ in range(int(args.post_load_wait_ticks)):
            try:
                world.wait_for_tick(timeout=5.0)
            except Exception:
                time.sleep(0.1)

    tm = None
    if args.autopilot or (args.num_vehicles > 0):
        tm = client.get_trafficmanager(args.tm_port)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.fixed_dt)
    settings.no_rendering_mode = False
    if hasattr(settings, "deterministic_ragdolls"):
        settings.deterministic_ragdolls = True
    if hasattr(settings, "substepping"):
        settings.substepping = False
    world.apply_settings(settings)

    if tm is not None:
        configure_tm_global(log, tm, args)

    set_weather_preset(world, args.weather, log=log)
    weather_overrides_applied = apply_weather_overrides(world, args, log=log)
    log.info("Post-load warmup: %d ticks (no sensors)...", int(args.map_load_warmup_ticks))
    for _ in range(int(args.map_load_warmup_ticks)):
        world.tick()

    rng = random.Random(int(args.seed))
    npc_rng = random.Random(int(args.npc_seed) if args.npc_seed is not None else int(args.seed))

    bp_lib = world.get_blueprint_library()

    if args.vehicle_bp == "random":
        veh_bp = rng.choice(bp_lib.filter("vehicle.*"))
    else:
        veh_bp = bp_lib.find(args.vehicle_bp)
    if veh_bp.has_attribute("role_name"):
        veh_bp.set_attribute("role_name", "ego")
    if veh_bp.has_attribute("color") and args.vehicle_color:
        veh_bp.set_attribute("color", args.vehicle_color)

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in map %s" % current_map)
    spawn_transform = spawn_points[int(args.spawn_index)] if args.spawn_index is not None else rng.choice(spawn_points)

    rgb_buf = SensorBuffer("rgb")
    imu_buf = SensorBuffer("imu")
    gnss_buf = SensorBuffer("gnss")
    semseg_buf = SensorBuffer("semseg") if args.semseg else None
    lidar_buf = SensorBuffer("lidar") if args.lidar else None

    ego = None
    sensors: List[carla.Actor] = []
    npc_vehicles: List[carla.Actor] = []
    walkers: List[carla.Actor] = []
    walker_controllers: List[carla.Actor] = []
    cleaned = False

    try:
        for _ in range(int(args.spawn_retries)):
            ego = world.try_spawn_actor(veh_bp, spawn_transform)
            if ego is not None:
                break
            spawn_transform.rotation.yaw += 5.0
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle after %d retries." % int(args.spawn_retries))

        log.info("Ego spawned: id=%s, bp=%s", ego.id, ego.type_id)

        hold_ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
        if args.hold_ego:
            try:
                ego.set_autopilot(False)
            except Exception:
                pass
            try:
                ego.apply_control(hold_ctrl)
            except Exception:
                pass

        if args.num_vehicles > 0:
            npc_vehicles = spawn_npc_vehicles(log, world, tm, args, spawn_transform, npc_rng)
        if args.num_walkers > 0:
            ego_loc = ego.get_location()
            walkers, walker_controllers = spawn_walkers(log, world, args, npc_rng, ego_loc=ego_loc)
        if args.autopilot and (not args.hold_ego):
            ego.set_autopilot(True, int(args.tm_port))
            configure_tm_for_vehicle(log, tm, ego, args, is_ego=True)
            log.info("Autopilot enabled via TM port %d", int(args.tm_port))
            log.info("TM ego config: ignore_lights=%.1f%% ignore_signs=%.1f%% speed_diff=%.1f%% lane_change=%s",
                     float(args.tm_ignore_lights_pct), float(args.tm_ignore_signs_pct),
                     float(args.tm_speed_diff_pct), str(bool(args.tm_auto_lane_change)))

        sensor_tick = float(args.fixed_dt)

        cam_tf = carla.Transform(
            carla.Location(x=args.cam_x, y=args.cam_y, z=args.cam_z),
            carla.Rotation(pitch=args.cam_pitch, yaw=args.cam_yaw, roll=args.cam_roll),
        )

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.cam_w))
        cam_bp.set_attribute("image_size_y", str(args.cam_h))
        cam_bp.set_attribute("fov", str(args.cam_fov))
        cam_bp.set_attribute("sensor_tick", str(sensor_tick))
        if cam_bp.has_attribute("enable_postprocess_effects"):
            cam_bp.set_attribute("enable_postprocess_effects", "False")

        rgb = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
        sensors.append(rgb)
        safe_listen(rgb, lambda img: (img.convert(carla.ColorConverter.Raw), rgb_buf.put(img)))
        log.info("RGB attached: %dx%d fov=%.1f", int(args.cam_w), int(args.cam_h), float(args.cam_fov))

        imu_bp = bp_lib.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(sensor_tick))
        imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(0, 0, 0)), attach_to=ego)
        sensors.append(imu)
        safe_listen(imu, lambda meas: imu_buf.put(meas))
        log.info("IMU attached")

        gnss_bp = bp_lib.find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", str(sensor_tick))
        gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(0, 0, 0)), attach_to=ego)
        sensors.append(gnss)
        safe_listen(gnss, lambda meas: gnss_buf.put(meas))
        log.info("GNSS attached")

        if args.semseg:
            ss_bp = bp_lib.find("sensor.camera.semantic_segmentation")
            ss_bp.set_attribute("image_size_x", str(args.cam_w))
            ss_bp.set_attribute("image_size_y", str(args.cam_h))
            ss_bp.set_attribute("fov", str(args.cam_fov))
            ss_bp.set_attribute("sensor_tick", str(sensor_tick))
            semseg = world.spawn_actor(ss_bp, cam_tf, attach_to=ego)
            sensors.append(semseg)
            safe_listen(semseg, lambda img: (img.convert(carla.ColorConverter.CityScapesPalette), semseg_buf.put(img)))
            log.info("SemSeg attached (CityScapesPalette)")

        if args.lidar:
            ld_bp = bp_lib.find("sensor.lidar.ray_cast")
            ld_bp.set_attribute("range", str(args.lidar_range))
            ld_bp.set_attribute("rotation_frequency", str(1.0 / sensor_tick))
            ld_bp.set_attribute("points_per_second", str(args.lidar_pps))
            ld_bp.set_attribute("upper_fov", str(args.lidar_upper_fov))
            ld_bp.set_attribute("lower_fov", str(args.lidar_lower_fov))
            ld_bp.set_attribute("channels", str(args.lidar_channels))
            ld_bp.set_attribute("sensor_tick", str(sensor_tick))

            ld_tf = carla.Transform(
                carla.Location(x=args.lidar_x, y=args.lidar_y, z=args.lidar_z),
                carla.Rotation(0.0, 0.0, 0.0),
            )
            lidar = world.spawn_actor(ld_bp, ld_tf, attach_to=ego)
            sensors.append(lidar)
            safe_listen(lidar, lambda meas: lidar_buf.put(meas))
            log.info("LiDAR attached")

        log.info("Warming up %d ticks...", int(args.warmup_ticks))
        for _ in range(int(args.warmup_ticks)):
            world.tick()

        meta: Dict[str, Any] = {
            "created_utc": now_utc_iso(),
            "carla_version": "0.9.15",
            "host": args.host,
            "port": int(args.port),
            "map": current_map,
            "seed": int(args.seed),
            "fixed_delta_seconds": float(args.fixed_dt),
            "synchronous_mode": True,
            "no_rendering_mode_capture": False,
            "weather": args.weather,
            "weather_params": weather_to_dict(world.get_weather()),
            "weather_overrides": weather_overrides_applied,
            "ego": {
                "blueprint": ego.type_id,
                "transform_spawn": transform_to_dict(spawn_transform),
                "autopilot": bool(args.autopilot),
                "traffic_manager_port": int(args.tm_port) if args.autopilot else None,
                "traffic_manager": {
                    "distance_to_leading": float(args.tm_distance_to_leading),
                    "ignore_lights_pct": float(args.tm_ignore_lights_pct),
                    "ignore_signs_pct": float(args.tm_ignore_signs_pct),
                    "ignore_walkers_pct": float(args.tm_ignore_walkers_pct),
                    "speed_diff_pct": float(args.tm_speed_diff_pct),
                    "auto_lane_change": bool(args.tm_auto_lane_change),
                    "hybrid_physics": bool(args.tm_hybrid_physics),
                    "hybrid_radius": float(args.tm_hybrid_radius),
                } if args.autopilot else None,
            },
            "npcs": {
                "npc_seed": int(args.npc_seed) if args.npc_seed is not None else int(args.seed),
                "vehicles_requested": int(args.num_vehicles),
                "vehicles_spawned": int(len(npc_vehicles)),
                "walkers_requested": int(args.num_walkers),
                "walkers_spawned": int(len(walkers)),
                "pedestrians_cross_factor": float(args.pedestrians_cross_factor) if args.pedestrians_cross_factor is not None else None,
                "npc_vehicle_filter": str(args.npc_vehicle_filter),
                "npc_min_dist_to_ego": float(args.npc_min_dist_to_ego),
                "tm_npc_speed_diff_pct": float(args.tm_npc_speed_diff_pct),
                "tm_npc_auto_lane_change": bool(args.tm_npc_auto_lane_change),
                "walker_speed_min": float(args.walker_speed_min),
                "walker_speed_max": float(args.walker_speed_max),
            },
            "sensors": {
                "rgb": {"transform": transform_to_dict(cam_tf), "w": int(args.cam_w), "h": int(args.cam_h),
                        "fov": float(args.cam_fov), "tick": float(sensor_tick)},
                "imu": {"tick": float(sensor_tick)},
                "gnss": {"tick": float(sensor_tick)},
                "semseg": bool(args.semseg),
                "lidar": bool(args.lidar),
            },
            "shutdown": {
                "park_server": bool(args.park_server),
                "park_fixed_dt": float(args.park_fixed_dt),
                "shutdown_sleep": float(args.shutdown_sleep),
            }
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        log.info("Wrote meta.json")

        state_f = open(state_path, "w", newline="", encoding="utf-8")
        index_f = open(index_path, "w", newline="", encoding="utf-8")
        state_writer = csv.writer(state_f)
        index_writer = csv.writer(index_f)

        state_writer.writerow([
            "frame", "sim_time", "ts_platform",
            "ego_x", "ego_y", "ego_z",
            "ego_roll", "ego_pitch", "ego_yaw",
            "vel_x", "vel_y", "vel_z",
            "acc_x", "acc_y", "acc_z",
            "angvel_x", "angvel_y", "angvel_z",
            "throttle", "steer", "brake", "hand_brake", "reverse", "gear",
            "imu_accel_x", "imu_accel_y", "imu_accel_z",
            "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
            "imu_compass",
            "gnss_lat", "gnss_lon", "gnss_alt",
        ])
        index_writer.writerow(["frame", "rgb_path", "semseg_path", "lidar_path"])

        log.info("Collecting episode: N=%d frames (fixed_dt=%.3f)", int(args.num_frames), float(args.fixed_dt))

        saved = 0
        t0 = time.time()

        for _ in range(int(args.num_frames)):
            # advance simulation capture_stride ticks; save sample from the last tick
            frame = None
            for _s in range(max(1, int(args.capture_stride))):
                frame = world.tick()
                if args.hold_ego:
                    try:
                        ego.apply_control(hold_ctrl)
                    except Exception:
                        pass

            rgb_meas = rgb_buf.get_for_frame(frame, args.sensor_timeout)
            imu_meas = imu_buf.get_for_frame(frame, args.sensor_timeout)
            gnss_meas = gnss_buf.get_for_frame(frame, args.sensor_timeout)

            if rgb_meas is None or imu_meas is None or gnss_meas is None:
                raise RuntimeError("Sensor timeout at frame=%d. Try --sensor-timeout higher." % frame)

            ss_meas = None
            if args.semseg:
                ss_meas = semseg_buf.get_for_frame(frame, args.sensor_timeout)
                if ss_meas is None:
                    raise RuntimeError("SemSeg timeout at frame=%d" % frame)

            ld_meas = None
            if args.lidar:
                ld_meas = lidar_buf.get_for_frame(frame, args.sensor_timeout)
                if ld_meas is None:
                    raise RuntimeError("LiDAR timeout at frame=%d" % frame)

            rgb_path = os.path.join(images_dir, "%06d.png" % frame)
            save_png(carla_image_to_rgb_array(rgb_meas), rgb_path)

            ss_path = ""
            if args.semseg:
                ss_path = os.path.join(labels_dir, "%06d.png" % frame)
                save_png(carla_image_to_rgb_array(ss_meas), ss_path)

            ld_path = ""
            if args.lidar:
                ld_path = os.path.join(lidar_dir, "%06d.npy" % frame)
                np.save(ld_path, lidar_to_npy(ld_meas), allow_pickle=False)

            snap = world.get_snapshot()
            ts = snap.timestamp
            ego_tf = ego.get_transform()
            vel = ego.get_velocity()
            acc = ego.get_acceleration()
            ang = ego.get_angular_velocity()
            ctrl = ego.get_control()

            imu_acc = imu_meas.accelerometer
            imu_gyro = imu_meas.gyroscope
            imu_compass = float(imu_meas.compass)

            state_writer.writerow([
                int(frame), float(ts.elapsed_seconds), float(time.time()),
                float(ego_tf.location.x), float(ego_tf.location.y), float(ego_tf.location.z),
                float(ego_tf.rotation.roll), float(ego_tf.rotation.pitch), float(ego_tf.rotation.yaw),
                float(vel.x), float(vel.y), float(vel.z),
                float(acc.x), float(acc.y), float(acc.z),
                float(ang.x), float(ang.y), float(ang.z),
                float(ctrl.throttle), float(ctrl.steer), float(ctrl.brake),
                int(bool(ctrl.hand_brake)), int(bool(ctrl.reverse)), int(ctrl.gear),
                float(imu_acc.x), float(imu_acc.y), float(imu_acc.z),
                float(imu_gyro.x), float(imu_gyro.y), float(imu_gyro.z),
                float(imu_compass),
                float(gnss_meas.latitude), float(gnss_meas.longitude), float(gnss_meas.altitude),
            ])

            rel_rgb = os.path.relpath(rgb_path, ep_dir)
            rel_ss = os.path.relpath(ss_path, ep_dir) if ss_path else ""
            rel_ld = os.path.relpath(ld_path, ep_dir) if ld_path else ""
            index_writer.writerow([int(frame), rel_rgb, rel_ss, rel_ld])

            saved += 1
            if args.log_every > 0 and (saved % int(args.log_every) == 0):
                log.info("Saved %d/%d frames (last frame=%d)", saved, int(args.num_frames), frame)

        dt_s = time.time() - t0
        log.info("DONE. Saved %d frames in %.2fs (%.2f FPS effective)", saved, dt_s, saved / max(dt_s, 1e-6))

        state_f.close()
        index_f.close()

        classic_teardown(
            log=log,
            world=world,
            tm=tm,
            original_settings=original_settings,
            ego=ego,
            sensors=sensors,
            npc_vehicles=npc_vehicles,
            walkers=walkers,
            walker_controllers=walker_controllers,
            shutdown_sleep=args.shutdown_sleep,
            park_server=args.park_server,
            park_fixed_dt=args.park_fixed_dt if args.park_fixed_dt > 0 else args.fixed_dt
        )
        cleaned = True
        return 0

    finally:
        if not cleaned:
            try:
                classic_teardown(
                    log=log,
                    world=world,
                    tm=tm,
                    original_settings=original_settings,
                    ego=ego,
                    sensors=sensors,
                    npc_vehicles=npc_vehicles,
                    walkers=walkers,
                    walker_controllers=walker_controllers,
                    shutdown_sleep=args.shutdown_sleep,
                    park_server=args.park_server,
                    park_fixed_dt=args.park_fixed_dt if args.park_fixed_dt > 0 else args.fixed_dt
                )
            except Exception:
                pass


def build_argparser():
    p = argparse.ArgumentParser(description="CARLA 0.9.15 dataset-first collector (stable teardown + park server)")

    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--rpc-timeout", dest="rpc_timeout", type=float, default=10.0)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--cd-root", type=str, default=repo_root)
    p.add_argument("--episode-id", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("-N", "--num-frames", type=int, default=300)
    p.add_argument("--fixed-dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--warmup-ticks", type=int, default=10)
    p.add_argument("--sensor-timeout", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--capture-stride", type=int, default=1,
                   help="Save one sample every N ticks (stride). Use fixed_dt=0.05 + stride=4 for 0.2s effective.")
    p.add_argument("--hold-ego", action="store_true", default=False,
                   help="Keep ego stationary (brake+handbrake). Useful for domain visual comparison.")

    p.add_argument("--map-name", type=str, default="Town10HD_Opt")
    p.add_argument("--post-load-sleep", type=float, default=2.0,
                   help="Seconds to sleep after load_world before applying settings/spawning.")
    p.add_argument("--map-load-warmup-ticks", type=int, default=30,
                   help="Warmup ticks after settings+weather, before spawning sensors (stabilize after map load).")
    p.add_argument("--post-load-wait-ticks", type=int, default=40,
                   help="Async wait_for_tick() iterations after load_world (stabilize streaming/world state).")
    p.add_argument("--weather", type=str, default="current",
                   choices=["current", "clear", "cloudy", "wet", "hardrain",
                            "clear_sunset", "cloudy_sunset", "wet_sunset", "hardrain_sunset",
                            "soft_rain", "soft_rain_sunset"])

    p.add_argument("--sun-altitude-angle", dest="sun_altitude_angle", type=float, default=None)
    p.add_argument("--sun-azimuth-angle", dest="sun_azimuth_angle", type=float, default=None)
    p.add_argument("--fog-density", dest="fog_density", type=float, default=None)
    p.add_argument("--fog-distance", dest="fog_distance", type=float, default=None)
    p.add_argument("--fog-falloff", dest="fog_falloff", type=float, default=None)
    p.add_argument("--cloudiness", dest="cloudiness", type=float, default=None)
    p.add_argument("--precipitation", dest="precipitation", type=float, default=None)
    p.add_argument("--precipitation-deposits", dest="precipitation_deposits", type=float, default=None)
    p.add_argument("--wetness", dest="wetness", type=float, default=None)
    p.add_argument("--wind-intensity", dest="wind_intensity", type=float, default=None)

    p.add_argument("--spawn-index", type=int, default=None)
    p.add_argument("--spawn-retries", type=int, default=10)

    p.add_argument("--vehicle-bp", type=str, default="vehicle.tesla.model3")
    p.add_argument("--vehicle-color", type=str, default=None)

    p.add_argument("--autopilot", action="store_true")
    p.add_argument("--tm-port", type=int, default=8000)

    p.add_argument("--tm-distance-to-leading", type=float, default=3.5)
    p.add_argument("--tm-speed-diff-pct", type=float, default=30.0)
    p.add_argument("--tm-ignore-lights-pct", type=float, default=0.0)
    p.add_argument("--tm-ignore-signs-pct", type=float, default=0.0)
    p.add_argument("--tm-ignore-walkers-pct", type=float, default=0.0)
    p.add_argument("--tm-auto-lane-change", action="store_true", default=False)
    p.add_argument("--tm-hybrid-physics", action="store_true", default=False)
    p.add_argument("--tm-hybrid-radius", type=float, default=70.0)

    p.add_argument("--tm-npc-speed-diff-pct", type=float, default=20.0,
                   help="NPC speed difference vs speed limit (percent slower).")
    p.add_argument("--tm-npc-auto-lane-change", action="store_true", default=False,
                   help="Enable lane changes for NPC vehicles (default OFF).")

    p.add_argument("--npc-seed", type=int, default=None,
                   help="Deterministic seed for NPC spawning. Default: --seed.")
    p.add_argument("--num-vehicles", type=int, default=0,
                   help="Background traffic vehicles to spawn (default 0).")
    p.add_argument("--num-walkers", type=int, default=0,
                   help="Pedestrians (walkers) to spawn (default 0).")
    p.add_argument("--npc-vehicle-filter", type=str, default="vehicle.*",
                   help="Blueprint filter for NPC vehicles.")
    p.add_argument("--npc-min-dist-to-ego", type=float, default=15.0,
                   help="Minimum distance from ego spawn for NPC vehicle spawns (meters).")
    p.add_argument("--npc-max-dist-to-ego", type=float, default=None,
                   help="Max distance (meters) from ego for spawning NPCs (vehicles/walkers). If unset, no upper bound.")
    p.add_argument("--npc-vehicle-spawn-attempts", type=int, default=3,
                   help="Attempts per spawn point for NPC vehicles.")
    p.add_argument("--npc-walker-spawn-attempts", type=int, default=10,
                   help="Attempts per walker to find nav location and spawn.")
    p.add_argument("--pedestrians-cross-factor", type=float, default=0.1,
                   help="Pedestrians cross factor (0..1). Conservative default 0.1.")
    p.add_argument("--walker-speed-min", type=float, default=1.0,
                   help="Walker min speed (m/s).")
    p.add_argument("--walker-speed-max", type=float, default=2.0,
                   help="Walker max speed (m/s).")

    p.add_argument("--cam-w", type=int, default=800)
    p.add_argument("--cam-h", type=int, default=600)
    p.add_argument("--cam-fov", type=float, default=90.0)
    p.add_argument("--cam-x", type=float, default=1.5)
    p.add_argument("--cam-y", type=float, default=0.0)
    p.add_argument("--cam-z", type=float, default=1.7)
    p.add_argument("--cam-pitch", type=float, default=-5.0)
    p.add_argument("--cam-yaw", type=float, default=0.0)
    p.add_argument("--cam-roll", type=float, default=0.0)

    p.add_argument("--semseg", action="store_true")
    p.add_argument("--lidar", action="store_true")

    p.add_argument("--lidar-x", type=float, default=0.0)
    p.add_argument("--lidar-y", type=float, default=0.0)
    p.add_argument("--lidar-z", type=float, default=2.0)
    p.add_argument("--lidar-range", type=float, default=50.0)
    p.add_argument("--lidar-pps", type=int, default=100000)
    p.add_argument("--lidar-channels", type=int, default=32)
    p.add_argument("--lidar-upper-fov", type=float, default=10.0)
    p.add_argument("--lidar-lower-fov", type=float, default=-30.0)

    p.add_argument("--shutdown-sleep", type=float, default=2.0)
    p.add_argument("--park-server", action="store_true", default=True)
    p.add_argument("--no-park-server", dest="park_server", action="store_false")
    p.add_argument("--park-fixed-dt", type=float, default=0.05)

    return p


def main():
    args = build_argparser().parse_args()
    global carla
    if carla is None:
        try:
            import carla as carla_module
        except ImportError:
            print("ERROR: Python package 'carla' is required to collect episodes.", file=sys.stderr)
            print("Install the CARLA 0.9.15 Python client in the generation environment.", file=sys.stderr)
            return 2
        carla = carla_module
    try:
        return collect_episode(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print("ERROR:", str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
