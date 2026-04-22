"""Utilities for Phase 1 calculations."""

from typing import Tuple

import numpy as np


def speed_mps(velocity_xyz: Tuple[float, float, float]) -> float:
    x, y, z = velocity_xyz
    return float(np.sqrt(x * x + y * y + z * z))


def safe_braking_distance(speed_mps_value: float, reaction_time: float = 1.0, max_decel: float = 8.0) -> float:
    braking_distance = (speed_mps_value ** 2) / (2.0 * max_decel)
    reaction_distance = speed_mps_value * reaction_time
    return float(braking_distance + reaction_distance)
