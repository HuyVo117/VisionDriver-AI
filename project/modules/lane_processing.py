"""Perception: lane detection and lane state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class LaneState:
    center_offset_px: float
    left_x: Optional[int]
    right_x: Optional[int]
    quality: float


class LaneProcessingModule:
    """Simple Hough-based lane estimator for MVP."""

    def process(self, frame: np.ndarray) -> LaneState:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        roi = np.zeros_like(edges)
        polygon = np.array(
            [[(0, h), (w, h), (int(0.6 * w), int(0.6 * h)), (int(0.4 * w), int(0.6 * h))]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi, polygon, 255)
        cropped = cv2.bitwise_and(edges, roi)

        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=80)

        left_x = []
        right_x = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / float(x2 - x1)
                if abs(slope) < 0.3:
                    continue
                if slope < 0:
                    left_x.extend([x1, x2])
                else:
                    right_x.extend([x1, x2])

        left_lane_x = int(np.mean(left_x)) if left_x else None
        right_lane_x = int(np.mean(right_x)) if right_x else None

        if left_lane_x is not None and right_lane_x is not None:
            lane_center = (left_lane_x + right_lane_x) / 2.0
            center_offset = lane_center - (w / 2.0)
            quality = 1.0
        else:
            center_offset = 0.0
            quality = 0.0

        return LaneState(
            center_offset_px=float(center_offset),
            left_x=left_lane_x,
            right_x=right_lane_x,
            quality=quality,
        )
