"""
VisionDriver AI — Unified Pipeline (Phase 1 + Phase 2)
Chạy song song:
  - Thread 1: ADAS (road camera → scene understanding + alerts)
  - Thread 2: Driver Monitor (driver camera → face + state)
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from api.adas_pipeline import ADASPipeline, PipelineConfig
from api.run_driver_monitor import DriverMonitoringPipeline
from phase2_driver_monitoring.emergency_manager import EmergencyConfig
from modules.alert_engine import AlertEngine


def run_unified(
    road_camera: int = 0,
    driver_camera: int = 1,
    show: bool = True,
) -> None:
    """Chạy cả 2 pipeline đồng thời."""
    shared_alerts: queue.Queue = queue.Queue(maxsize=50)

    alert_engine = AlertEngine()
    alert_engine.on_alert(lambda a: shared_alerts.put_nowait(a))

    adas = ADASPipeline(PipelineConfig())
    em_config = EmergencyConfig(mock_mode=True)
    driver_monitor = DriverMonitoringPipeline(
        emergency_config=em_config,
        show_window=False,
    )
    # Wire alert engine
    driver_monitor.alert_engine = alert_engine

    road_cap = cv2.VideoCapture(road_camera)
    driver_cap = cv2.VideoCapture(driver_camera)

    stop_event = threading.Event()

    def road_thread() -> None:
        last_t = time.time()
        while not stop_event.is_set():
            ret, frame = road_cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            now = time.time()
            dt = max(0.001, now - last_t)
            last_t = now
            try:
                adas.process_frame(frame=frame, dt_s=dt)
            except Exception as e:
                print(f"[ADAS error] {e}")

    def driver_thread() -> None:
        while not stop_event.is_set():
            ret, frame = driver_cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            try:
                driver_monitor.process_frame(frame)
            except Exception as e:
                print(f"[Driver monitor error] {e}")

    t1 = threading.Thread(target=road_thread, daemon=True, name="ADAS")
    t2 = threading.Thread(target=driver_thread, daemon=True, name="DriverMonitor")
    t1.start()
    t2.start()

    print("VisionDriver AI — Unified Pipeline running")
    print("Press Q to quit")

    try:
        while True:
            try:
                alert = shared_alerts.get_nowait()
                print(f"[{alert.severity.name}] {alert.message}")
            except queue.Empty:
                pass
            time.sleep(0.05)

            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        road_cap.release()
        driver_cap.release()
        cv2.destroyAllWindows()
        print("Unified pipeline stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-cam", type=int, default=0)
    parser.add_argument("--driver-cam", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run_unified(
        road_camera=args.road_cam,
        driver_camera=args.driver_cam,
        show=not args.headless,
    )
