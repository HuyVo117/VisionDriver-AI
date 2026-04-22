"""Decision/UI layer visualization helpers (HUD + alerts)."""

from __future__ import annotations

import cv2
import numpy as np


def draw_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str, color=(0, 255, 0)) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def alert_level_from_ttc(ttc_s: float | None) -> tuple[str, tuple[int, int, int]]:
    if ttc_s is None:
        return "SAFE", (0, 200, 0)
    if ttc_s < 1.5:
        return "WARNING", (0, 0, 255)
    if ttc_s < 3.0:
        return "ATTENTION", (0, 255, 255)
    return "SAFE", (0, 200, 0)


def draw_hud(
    frame: np.ndarray,
    speed_kmh: float,
    lane_state: str,
    safe_distance_m: float | None,
    ttc_s: float | None,
) -> np.ndarray:
    status, color = alert_level_from_ttc(ttc_s)

    lines = [
        f"Speed: {speed_kmh:.1f} km/h",
        f"Lane: {lane_state}",
        f"Safe Distance: {safe_distance_m:.1f} m" if safe_distance_m is not None else "Safe Distance: n/a",
        f"TTC: {ttc_s:.2f} s" if ttc_s is not None else "TTC: n/a",
        f"Alert: {status}",
    ]

    y = 30
    for text in lines:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color if text.startswith("Alert") else (255, 255, 255), 2)
        y += 30

    return frame
