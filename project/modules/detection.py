"""Perception: object detection wrapper with optional YOLO backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class ObjectDetectionModule:
    """Vehicle/pedestrian detector.

    Uses Ultralytics YOLO when available and falls back to an empty output.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model = None
        if model_path:
            self._try_load_yolo(model_path)

    def _try_load_yolo(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
        except Exception:
            self.model = None

    def detect(self, frame: np.ndarray, conf_thres: float = 0.25) -> List[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(frame, conf=conf_thres, verbose=False)
        detections: List[Detection] = []

        for res in results:
            names = res.names
            for box in res.boxes:
                cls_idx = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        label=str(names.get(cls_idx, cls_idx)),
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections
