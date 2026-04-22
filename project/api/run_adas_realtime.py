"""Run modular ADAS pipeline in CARLA with real-time HUD output.

Pipeline per frame:
Perception (detection/lane/sign) -> Analysis (distance/TTC) -> Decision/UI (HUD alert)
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

import carla
from api.adas_pipeline import ADASPipeline, PipelineConfig


class SharedFrame:
    def __init__(self) -> None:
        self.frame: Optional[np.ndarray] = None


def _to_bgr(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]
    return bgr


def run(steps: int = 1200, show_window: bool = True) -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=0.5, z=1.3), carla.Rotation(pitch=0)),
        attach_to=vehicle,
    )

    shared = SharedFrame()

    def on_image(image: carla.Image) -> None:
        shared.frame = _to_bgr(image)

    camera.listen(on_image)

    pipeline = ADASPipeline(PipelineConfig())
    vehicle.set_autopilot(True)

    last_time = time.time()
    try:
        for _ in range(steps):
            world.tick()
            frame = shared.frame
            if frame is None:
                continue

            now = time.time()
            dt = max(1e-3, now - last_time)
            last_time = now

            vel = vehicle.get_velocity()
            speed_mps = float(np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))
            speed_kmh = speed_mps * 3.6

            out = pipeline.process_frame(frame=frame, speed_kmh=speed_kmh, dt_s=dt)
            hud = out["hud_frame"]

            if show_window:
                cv2.imshow("VisionDriver ADAS", hud)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        vehicle.set_autopilot(False)
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        if show_window:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADAS realtime demo in CARLA")
    parser.add_argument("--steps", type=int, default=1200, help="Simulation ticks to run")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable OpenCV window output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(steps=args.steps, show_window=not args.headless)
