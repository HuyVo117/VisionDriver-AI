"""Minimal ADAS pipeline: Perception -> Analysis -> Decision/UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from modules.detection import ObjectDetectionModule
from modules.distance_measurer import DistanceMeasurer
from modules.lane_processing import LaneProcessingModule
from modules.sign_classifier import TrafficSignRecognitionModule
from utils.visualization import draw_hud


@dataclass
class PipelineConfig:
    vehicle_model_path: Optional[str] = None
    traffic_sign_model_path: Optional[str] = None
    focal_length_px: float = 900.0


class ADASPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        cfg = config or PipelineConfig()
        self.detector = ObjectDetectionModule(cfg.vehicle_model_path)
        self.lane = LaneProcessingModule()
        self.sign = TrafficSignRecognitionModule(cfg.traffic_sign_model_path)
        self.distance = DistanceMeasurer(cfg.focal_length_px)

    def process_frame(self, frame: np.ndarray, speed_kmh: float = 0.0, dt_s: float = 0.1) -> Dict[str, Any]:
        detections = self.detector.detect(frame)
        lane_state = self.lane.process(frame)
        signs = self.sign.detect_and_classify(frame)

        closest_distance_m = None
        ttc_s = None
        for i, det in enumerate(detections):
            if det.label.lower() not in {"car", "truck", "bus", "motorbike", "vehicle"}:
                continue
            x1, y1, x2, y2 = det.bbox
            bbox_h = max(1, y2 - y1)
            distance_m = self.distance.estimate_distance(real_height_m=1.5, bbox_height_px=bbox_h)
            estimate = self.distance.compute_ttc(
                track_id=f"det_{i}",
                distance_m=distance_m,
                bbox_height_px=bbox_h,
                dt_s=dt_s,
            )
            if closest_distance_m is None or (estimate.distance_m is not None and estimate.distance_m < closest_distance_m):
                closest_distance_m = estimate.distance_m
                ttc_s = estimate.ttc_s

        lane_text = "ok" if lane_state.quality > 0.5 else "uncertain"
        hud_frame = draw_hud(
            frame.copy(),
            speed_kmh=speed_kmh,
            lane_state=lane_text,
            safe_distance_m=closest_distance_m,
            ttc_s=ttc_s,
        )

        return {
            "detections": detections,
            "lane_state": lane_state,
            "traffic_signs": signs,
            "closest_distance_m": closest_distance_m,
            "ttc_s": ttc_s,
            "hud_frame": hud_frame,
        }
