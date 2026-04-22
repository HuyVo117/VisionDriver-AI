"""Analysis layer: monocular distance and TTC estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DistanceEstimate:
    distance_m: Optional[float]
    relative_speed_mps: Optional[float]
    ttc_s: Optional[float]


class DistanceMeasurer:
    """Estimate distance from bounding-box height and compute TTC."""

    def __init__(self, focal_length_px: float = 900.0) -> None:
        self.focal_length_px = focal_length_px
        self._last_height_by_id: Dict[str, float] = {}

    def estimate_distance(self, real_height_m: float, bbox_height_px: int) -> Optional[float]:
        if bbox_height_px <= 0:
            return None
        return (real_height_m * self.focal_length_px) / float(bbox_height_px)

    def compute_ttc(
        self,
        track_id: str,
        distance_m: Optional[float],
        bbox_height_px: int,
        dt_s: float,
    ) -> DistanceEstimate:
        if distance_m is None or dt_s <= 0:
            return DistanceEstimate(None, None, None)

        last_h = self._last_height_by_id.get(track_id)
        self._last_height_by_id[track_id] = float(bbox_height_px)

        if last_h is None:
            return DistanceEstimate(distance_m, None, None)

        height_growth = (bbox_height_px - last_h) / dt_s
        if height_growth <= 0:
            return DistanceEstimate(distance_m, 0.0, None)

        # Approximate closing speed from inverse depth change proxy.
        relative_speed = min(60.0, height_growth * 0.05)
        if relative_speed <= 1e-6:
            return DistanceEstimate(distance_m, relative_speed, None)

        ttc = distance_m / relative_speed
        return DistanceEstimate(distance_m, relative_speed, ttc)
