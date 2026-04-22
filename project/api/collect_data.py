"""Collect CARLA driving data: RGB frames + control labels.

Run with CARLA server active on localhost:2000.
"""

import os

import carla
import cv2
import numpy as np
from carla.agents.navigation.behavior_agent import BehaviorAgent


def main(total_frames: int = 5000, data_dir: str = "collected_data") -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "640")
    camera_bp.set_attribute("image_size_y", "480")
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(
        carla.Location(x=0.5, z=1.3),
        carla.Rotation(pitch=0),
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    os.makedirs(data_dir, exist_ok=True)
    frame_count = 0

    def process_image(image: carla.Image) -> None:
        nonlocal frame_count

        control = vehicle.get_control()
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_array = image_array.reshape((image.height, image.width, 4))
        image_rgb = image_array[:, :, :3]

        image_path = os.path.join(data_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(image_path, image_rgb)

        with open(os.path.join(data_dir, "labels.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"{frame_count} {control.throttle} {control.brake} {control.steer} {speed}\n"
            )

        frame_count += 1

    camera.listen(process_image)

    agent = BehaviorAgent(vehicle, behavior="normal")
    destination = world.get_map().get_spawn_points()[10]
    agent.set_destination(destination.location)

    world.tick()
    try:
        for _ in range(total_frames):
            control = agent.run_step()
            vehicle.apply_control(control)
            world.tick()
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    main()
