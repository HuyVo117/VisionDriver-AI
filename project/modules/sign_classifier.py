"""Perception: traffic sign detection + optional fine-grained classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .detection import Detection, ObjectDetectionModule


@dataclass
class TrafficSign:
    sign_type: str
    confidence: float
    bbox: tuple[int, int, int, int]
    text: Optional[str] = None


class TrafficSignRecognitionModule:
    """Two-step TSR: detect sign region, then classify/OCR if available."""

    def __init__(
        self,
        detector_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self.detector = ObjectDetectionModule(detector_model_path)
        self.label_map = label_map or {}
        self.classifier = None
        if classifier_model_path:
            self._try_load_classifier(classifier_model_path)

    def _try_load_classifier(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO

            self.classifier = YOLO(model_path)
        except Exception:
            self.classifier = None

    def detect_and_classify(self, frame: np.ndarray) -> List[TrafficSign]:
        detections = self.detector.detect(frame)
        sign_detections = [d for d in detections if "sign" in d.label.lower()]
        return [self._classify_crop(frame, d) for d in sign_detections]

    def _classify_crop(self, frame: np.ndarray, det: Detection) -> TrafficSign:
        x1, y1, x2, y2 = det.bbox
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

        # MVP rule-based label refinement placeholder.
        sign_type = det.label
        text = None

        if self.classifier is not None and crop.size > 0:
            try:
                results = self.classifier.predict(crop, verbose=False)
                if results:
                    probs = results[0].probs
                    if probs is not None:
                        cls_idx = int(probs.top1)
                        cls_name = self.label_map.get(cls_idx)
                        if cls_name:
                            sign_type = cls_name
                        else:
                            sign_type = str(results[0].names.get(cls_idx, cls_idx))
            except Exception:
                # Keep detector label if classification fails.
                pass

        return TrafficSign(sign_type=sign_type, confidence=det.confidence, bbox=det.bbox, text=text)


def build_label_map_from_dataset(dataset_path: str) -> Dict[int, str]:
    """Build class-index mapping from a classification dataset folder structure.

    Expected layout example:
    - dataset_path/train/<class_name>/...
    - dataset_path/valid/<class_name>/...
    """
    base = Path(dataset_path)
    if not base.exists():
        return {}

    candidate_roots = [base / "train", base / "Train", base / "training", base]
    class_dirs: List[Path] = []
    for root in candidate_roots:
        if root.exists() and root.is_dir():
            class_dirs = [p for p in root.iterdir() if p.is_dir()]
            if class_dirs:
                break

    class_names = sorted({p.name for p in class_dirs})
    return {idx: name for idx, name in enumerate(class_names)}
