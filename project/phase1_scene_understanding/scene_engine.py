"""Scene understanding engine for Phase 1."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from .utils import safe_braking_distance


class ObjectType(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    TRAFFIC_SIGN = "traffic_sign"
    OBSTACLE = "obstacle"


@dataclass
class DetectedObject:
    obj_type: ObjectType
    distance: float
    position: Tuple[float, float]
    speed: float
    direction: float
    confidence: float
    label: str


@dataclass
class SceneContext:
    speed_limit: int
    road_type: str
    weather: str
    traffic_density: float
    current_speed: float
    ego_position: Tuple[float, float]
    obstacles: List[DetectedObject]
    traffic_signs: List[str]
    safe_distance_ahead: float
    safe_distance_behind: float
    can_change_lane: bool


class SceneUnderstandingEngine:
    """Converts sensor inputs to a compact scene context."""

    def __init__(self) -> None:
        self.vehicle_detector = None
        self.pedestrian_detector = None
        self.sign_detector = None
        self.lane_detector = None

    def process_frame(self, rgb_image: np.ndarray) -> SceneContext:
        # Placeholder outputs for MVP; replace with detector outputs incrementally.
        obstacles: List[DetectedObject] = []
        current_speed = 0.0
        safe_distance = safe_braking_distance(current_speed)

        return SceneContext(
            speed_limit=50,
            road_type="urban",
            weather="clear",
            traffic_density=0.2,
            current_speed=current_speed,
            ego_position=(0.0, 0.0),
            obstacles=obstacles,
            traffic_signs=[],
            safe_distance_ahead=safe_distance,
            safe_distance_behind=100.0,
            can_change_lane=True,
        )
