"""
VisionDriver AI — Entry Point

Chạy pipeline tương ứng theo mode:

  python main.py phase1          # ADAS realtime (cần CARLA)
  python main.py phase2          # Driver monitoring (cần webcam)
  python main.py unified         # Cả 2 (cần CARLA + 2 camera)
  python main.py qa              # Hỏi đáp luật giao thông (demo)
  python main.py download        # Tải datasets
"""

from __future__ import annotations

import argparse
import sys


def cmd_phase1(args: argparse.Namespace) -> None:
    from api.run_adas_realtime import run
    run(steps=args.steps, show_window=not args.headless)


def cmd_phase2(args: argparse.Namespace) -> None:
    from api.run_driver_monitor import DriverMonitoringPipeline
    from phase2_driver_monitoring.emergency_manager import EmergencyConfig

    cfg = EmergencyConfig(
        emergency_number=args.emergency_number,
        contact_numbers=args.contact or [],
        mock_mode=not args.live_call,
    )
    pipeline = DriverMonitoringPipeline(emergency_config=cfg, show_window=not args.headless)
    pipeline.start(camera_index=args.camera)


def cmd_unified(args: argparse.Namespace) -> None:
    from api.run_unified import run_unified
    run_unified(
        road_camera=args.road_cam,
        driver_camera=args.driver_cam,
        show=not args.headless,
    )


def cmd_qa(_args: argparse.Namespace) -> None:
    from phase1_scene_understanding.qa_engine import TrafficQAEngine

    qa = TrafficQAEngine()
    print("VisionDriver QA — Hỏi đáp luật giao thông VN (gõ 'exit' để thoát)\n")
    while True:
        try:
            question = input("Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit", "thoát"):
            break
        result = qa.answer(question)
        print(f"[{result.source.upper()}] {result.answer}\n")


def cmd_download(args: argparse.Namespace) -> None:
    import subprocess
    cmd = [sys.executable, "scripts/download_datasets.py"]
    if args.all:
        cmd.append("--all")
    elif args.signs:
        cmd.append("--signs")
    elif args.drowsiness:
        cmd.append("--drowsiness")
    else:
        cmd.append("--check")
    subprocess.run(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visiondriver",
        description="VisionDriver AI — Hệ thống AI hỗ trợ lái xe",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # phase1
    p1 = sub.add_parser("phase1", help="ADAS realtime (cần CARLA server)")
    p1.add_argument("--steps", type=int, default=1200)
    p1.add_argument("--headless", action="store_true")

    # phase2
    p2 = sub.add_parser("phase2", help="Driver monitoring (cần webcam)")
    p2.add_argument("--camera", type=int, default=0)
    p2.add_argument("--headless", action="store_true")
    p2.add_argument("--emergency-number", default="115")
    p2.add_argument("--contact", action="append", default=[])
    p2.add_argument("--live-call", action="store_true", help="Dùng Twilio thật (cần config .env)")

    # unified
    pu = sub.add_parser("unified", help="Cả 2 pipeline song song")
    pu.add_argument("--road-cam", type=int, default=0)
    pu.add_argument("--driver-cam", type=int, default=1)
    pu.add_argument("--headless", action="store_true")

    # qa
    sub.add_parser("qa", help="Hỏi đáp luật giao thông VN (interactive)")

    # download
    dl = sub.add_parser("download", help="Tải datasets")
    dl.add_argument("--all", action="store_true")
    dl.add_argument("--signs", action="store_true")
    dl.add_argument("--drowsiness", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "phase1": cmd_phase1,
        "phase2": cmd_phase2,
        "unified": cmd_unified,
        "qa": cmd_qa,
        "download": cmd_download,
    }
    dispatch[args.mode](args)
