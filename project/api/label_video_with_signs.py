"""Label traffic signs on a local video using detection + optional sign classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from modules.sign_classifier import TrafficSignRecognitionModule, build_label_map_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label traffic signs in a video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", default="outputs/labeled_signs.mp4", help="Output video path")
    parser.add_argument("--detector", default=None, help="YOLO detector model path for traffic signs")
    parser.add_argument("--classifier", default=None, help="Optional YOLO classification model path")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset root used to build class labels (e.g. kagglehub download path)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    label_map = build_label_map_from_dataset(args.dataset) if args.dataset else {}

    tsr = TrafficSignRecognitionModule(
        detector_model_path=args.detector,
        classifier_model_path=args.classifier,
        label_map=label_map,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            signs = tsr.detect_and_classify(frame)
            for sign in signs:
                x1, y1, x2, y2 = sign.bbox
                label = f"{sign.sign_type} {sign.confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"Saved labeled video to: {out_path}")


if __name__ == "__main__":
    main()
