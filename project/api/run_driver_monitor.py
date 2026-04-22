"""
Phase 2: Driver Monitoring Pipeline — Entry point.

Chạy song song với Phase 1 ADAS pipeline:
  Driver Camera → FaceMonitor → DriverStateClassifier → AlertEngine → EmergencyManager
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from phase2_driver_monitoring.face_monitor import FaceMonitor
from phase2_driver_monitoring.state_classifier import DriverState, DriverStateClassifier
from phase2_driver_monitoring.emergency_manager import EmergencyConfig, EmergencyManager
from modules.alert_engine import AlertEngine, Severity


class DriverMonitoringPipeline:
    """
    Pipeline đầy đủ cho Phase 2.

    Usage:
        pipeline = DriverMonitoringPipeline()
        pipeline.start(camera_index=1)   # Camera thứ 2 (driver-facing)
    """

    def __init__(
        self,
        emergency_config: Optional[EmergencyConfig] = None,
        show_window: bool = True,
    ) -> None:
        self.face_monitor = FaceMonitor()
        self.state_classifier = DriverStateClassifier()
        self.alert_engine = AlertEngine()
        self.emergency_manager = EmergencyManager(emergency_config)
        self.show_window = show_window

        # Wiring: alert callbacks
        self.alert_engine.on_alert(
            lambda a: print(f"[ALERT] [{a.severity.name}] {a.message}"),
        )

    def process_frame(self, bgr_frame: np.ndarray) -> dict:
        """Xử lý một frame từ camera tài xế."""
        metrics = self.face_monitor.process_frame(bgr_frame)
        report = self.state_classifier.update(metrics)

        # Map state → alert
        state = report.state
        if state == DriverState.DROWSY_WARN:
            self.alert_engine.trigger("drowsiness_warning", {})
        elif state == DriverState.DROWSY_CRIT:
            alert = self.alert_engine.trigger("drowsiness_critical", {})
            if alert:
                self.emergency_manager.handle_emergency("DROWSY_CRIT")
        elif state == DriverState.UNRESPONSIVE:
            alert = self.alert_engine.trigger("driver_unresponsive", {})
            if alert:
                self.emergency_manager.handle_emergency("UNRESPONSIVE", report.details)
        elif state == DriverState.EMERGENCY:
            alert = self.alert_engine.trigger("medical_emergency", {})
            if alert:
                self.emergency_manager.handle_emergency("EMERGENCY", report.details)

        return {
            "metrics": metrics,
            "report": report,
            "active_alerts": self.alert_engine.get_active_alerts(within_s=5.0),
        }

    def _draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Vẽ thông tin lên frame."""
        overlay = frame.copy()
        metrics = result["metrics"]
        report = result["report"]

        h, w = overlay.shape[:2]

        # Background panel
        cv2.rectangle(overlay, (0, 0), (320, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, overlay)

        state_color = {
            DriverState.NORMAL: (0, 255, 0),
            DriverState.DISTRACTED: (0, 255, 255),
            DriverState.DROWSY_WARN: (0, 165, 255),
            DriverState.DROWSY_CRIT: (0, 0, 255),
            DriverState.UNRESPONSIVE: (0, 0, 200),
            DriverState.EMERGENCY: (0, 0, 180),
        }.get(report.state, (255, 255, 255))

        cv2.putText(overlay, f"State: {report.state.name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
        cv2.putText(overlay, f"EAR: {metrics.ear:.3f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(overlay, f"PERCLOS: {metrics.perclos:.1%}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(overlay, f"Drowsy: {metrics.drowsiness_score:.2f}", (10, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
        cv2.putText(overlay, f"Pitch:{metrics.head_pitch:.0f} Yaw:{metrics.head_yaw:.0f}", (10, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Active alerts
        for i, alert in enumerate(result["active_alerts"][-3:]):
            y = h - 30 - i * 22
            cv2.putText(overlay, alert.message[:60], (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        return overlay

    def start(self, camera_index: int = 0, max_frames: int = 0) -> None:
        """Chạy monitoring realtime từ webcam."""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"[ERROR] Không mở được camera index {camera_index}")
            return

        print(f"[Phase2] Driver monitoring started (camera={camera_index})")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Không đọc được frame")
                    time.sleep(0.1)
                    continue

                result = self.process_frame(frame)

                if self.show_window:
                    display = self._draw_overlay(frame, result)
                    cv2.imshow("VisionDriver — Driver Monitor", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if max_frames > 0 and frame_count >= max_frames:
                    break

        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            print("[Phase2] Driver monitoring stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionDriver Phase 2 — Driver Monitoring")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--headless", action="store_true", help="Chạy không cần cửa sổ")
    parser.add_argument("--emergency-number", default="115", help="Số cấp cứu")
    parser.add_argument("--contact", action="append", default=[], help="Số điện thoại liên hệ khẩn cấp")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    em_config = EmergencyConfig(
        emergency_number=args.emergency_number,
        contact_numbers=args.contact,
        mock_mode=True,  # Đổi sang False khi có Twilio account thật
    )
    pipeline = DriverMonitoringPipeline(
        emergency_config=em_config,
        show_window=not args.headless,
    )
    pipeline.start(camera_index=args.camera)
