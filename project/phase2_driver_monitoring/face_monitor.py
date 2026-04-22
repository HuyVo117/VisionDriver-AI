"""
Phase 2: Face & Eye Monitoring Pipeline
Sử dụng MediaPipe FaceMesh để track 468 facial landmarks.

Các chỉ số được tính:
- EAR (Eye Aspect Ratio): phát hiện mắt nhắm
- PERCLOS: % thời gian mắt nhắm trong 60s
- Head pose (pitch/yaw/roll): phát hiện gật đầu
- MAR (Mouth Aspect Ratio): phát hiện ngáp
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# MediaPipe FaceMesh landmark indices
# Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_IRIS = [474, 475, 476, 477]
_RIGHT_IRIS = [469, 470, 471, 472]
_MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EAR_TIP = 234
_RIGHT_EAR_TIP = 454
_LEFT_BROW = 70
_RIGHT_BROW = 300


@dataclass
class FaceMetrics:
    """Tất cả chỉ số từ một frame."""
    ear: float                      # Eye Aspect Ratio (0 = nhắm, 0.3+ = mở)
    left_ear: float
    right_ear: float
    mar: float                      # Mouth Aspect Ratio (ngáp khi > 0.6)
    head_pitch: float               # Gật đầu lên/xuống (độ)
    head_yaw: float                 # Quay trái/phải (độ)
    head_roll: float                # Nghiêng đầu (độ)
    perclos: float                  # % thời gian mắt nhắm (60s window)
    eyes_closed: bool               # True nếu EAR < threshold hiện tại
    looking_away: bool              # True nếu |yaw| > 30° hoặc |pitch| > 20°
    drowsiness_score: float         # 0.0 – 1.0
    face_detected: bool
    timestamp: float


class FaceMonitor:
    """
    Theo dõi khuôn mặt tài xế realtime.

    Usage:
        monitor = FaceMonitor()
        monitor.start()
        metrics = monitor.process_frame(bgr_frame)
        if metrics.drowsiness_score > 0.7:
            alert_engine.trigger("drowsiness_critical")
    """

    EAR_THRESHOLD = 0.22          # EAR dưới mức này = mắt nhắm
    EAR_CONSEC_FRAMES = 15        # Frame liên tiếp mắt nhắm → buồn ngủ
    MAR_YAWN_THRESHOLD = 0.6      # MAR trên mức này = ngáp
    HEAD_PITCH_LIMIT = 20.0       # Độ gật xuống → buồn ngủ
    PERCLOS_WINDOW_S = 60.0       # Cửa sổ tính PERCLOS (giây)
    PERCLOS_DROWSY_THRESH = 0.15  # PERCLOS > 15% → buồn ngủ

    def __init__(self) -> None:
        self._mp_face = None
        self._face_mesh = None
        self._closed_frames = 0
        self._eye_states: Deque[Tuple[float, bool]] = deque()  # (timestamp, is_closed)
        self._yawn_count = 0
        self._last_yawn_time = 0.0

        if _MP_AVAILABLE:
            self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        mp_face = mp.solutions.face_mesh  # type: ignore[attr-defined]
        self._face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,   # Iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    @classmethod
    def _compute_ear(cls, landmarks: np.ndarray, indices: List[int]) -> float:
        """Eye Aspect Ratio theo Soukupová & Čech (2016)."""
        p = landmarks[indices]
        # Vertical distances
        a = cls._distance(p[1], p[5])
        b = cls._distance(p[2], p[4])
        # Horizontal distance
        c = cls._distance(p[0], p[3])
        return (a + b) / (2.0 * c + 1e-6)

    @classmethod
    def _compute_mar(cls, landmarks: np.ndarray) -> float:
        """Mouth Aspect Ratio."""
        p = landmarks[[61, 291, 39, 181, 0, 17]]  # simplified
        v1 = cls._distance(p[2], p[5])
        v2 = cls._distance(p[3], p[4])
        h = cls._distance(p[0], p[1])
        return (v1 + v2) / (2.0 * h + 1e-6)

    @staticmethod
    def _compute_head_pose(
        landmarks: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float, float]:
        """
        Ước lượng head pose đơn giản từ landmarks.
        Trả về (pitch, yaw, roll) theo độ.
        """
        nose = landmarks[1]
        chin = landmarks[152]
        left_eye_outer = landmarks[263]
        right_eye_outer = landmarks[33]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        # Yaw: tỷ lệ trái/phải của mắt so với mũi
        face_width = abs(left_eye_outer[0] - right_eye_outer[0])
        nose_offset = nose[0] - (left_eye_outer[0] + right_eye_outer[0]) / 2.0
        yaw = (nose_offset / (face_width + 1e-6)) * 90.0

        # Pitch: vị trí mũi so với mắt và cằm
        eye_mid_y = (left_eye_outer[1] + right_eye_outer[1]) / 2.0
        face_height = abs(chin[1] - eye_mid_y)
        nose_v_offset = nose[1] - eye_mid_y
        pitch = ((nose_v_offset / (face_height + 1e-6)) - 0.4) * 90.0

        # Roll: độ nghiêng của đường nối 2 mắt
        dy = left_eye_outer[1] - right_eye_outer[1]
        dx = left_eye_outer[0] - right_eye_outer[0]
        roll = math.degrees(math.atan2(dy, dx + 1e-6))

        return float(pitch), float(yaw), float(roll)

    def _compute_perclos(self, now: float) -> float:
        """Tỷ lệ % thời gian mắt nhắm trong cửa sổ PERCLOS_WINDOW_S."""
        cutoff = now - self.PERCLOS_WINDOW_S
        # Loại bỏ entries cũ
        while self._eye_states and self._eye_states[0][0] < cutoff:
            self._eye_states.popleft()

        if not self._eye_states:
            return 0.0

        closed = sum(1 for _, c in self._eye_states if c)
        return closed / len(self._eye_states)

    def _compute_drowsiness_score(
        self,
        ear: float,
        perclos: float,
        pitch: float,
        mar: float,
    ) -> float:
        """Tổng hợp mức độ buồn ngủ thành một điểm 0–1."""
        score = 0.0

        # EAR component
        if ear < self.EAR_THRESHOLD:
            score += 0.4 * min(1.0, (self.EAR_THRESHOLD - ear) / self.EAR_THRESHOLD)

        # PERCLOS component
        if perclos > self.PERCLOS_DROWSY_THRESH:
            score += 0.35 * min(1.0, perclos / 0.4)

        # Head pitch (gật đầu) component
        if pitch > self.HEAD_PITCH_LIMIT:
            score += 0.25 * min(1.0, (pitch - self.HEAD_PITCH_LIMIT) / 20.0)

        # Yawn (MAR) – nhẹ
        if mar > self.MAR_YAWN_THRESHOLD:
            score = min(1.0, score + 0.1)

        return min(1.0, score)

    def process_frame(self, bgr_frame: np.ndarray) -> FaceMetrics:
        """Xử lý một frame BGR từ camera tài xế."""
        now = time.time()

        if not _MP_AVAILABLE or self._face_mesh is None:
            return self._empty_metrics(now)

        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = bgr_frame.shape[:2]

        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self._eye_states.append((now, False))
            return self._empty_metrics(now, face_detected=False)

        face_lm = result.multi_face_landmarks[0]
        lm_array = np.array([[lm.x * w, lm.y * h] for lm in face_lm.landmark])

        left_ear = self._compute_ear(lm_array, _LEFT_EYE)
        right_ear = self._compute_ear(lm_array, _RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        mar = self._compute_mar(lm_array)
        pitch, yaw, roll = self._compute_head_pose(lm_array, w, h)

        eyes_closed = ear < self.EAR_THRESHOLD
        self._eye_states.append((now, eyes_closed))

        if eyes_closed:
            self._closed_frames += 1
        else:
            self._closed_frames = 0

        perclos = self._compute_perclos(now)
        drowsiness = self._compute_drowsiness_score(ear, perclos, pitch, mar)

        # Yawn detection
        if mar > self.MAR_YAWN_THRESHOLD and now - self._last_yawn_time > 10.0:
            self._yawn_count += 1
            self._last_yawn_time = now

        looking_away = abs(yaw) > 30.0 or abs(pitch) > 20.0

        return FaceMetrics(
            ear=ear,
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            perclos=perclos,
            eyes_closed=eyes_closed,
            looking_away=looking_away,
            drowsiness_score=drowsiness,
            face_detected=True,
            timestamp=now,
        )

    @staticmethod
    def _empty_metrics(
        now: float,
        face_detected: bool = False,
    ) -> FaceMetrics:
        return FaceMetrics(
            ear=0.0,
            left_ear=0.0,
            right_ear=0.0,
            mar=0.0,
            head_pitch=0.0,
            head_yaw=0.0,
            head_roll=0.0,
            perclos=0.0,
            eyes_closed=False,
            looking_away=False,
            drowsiness_score=0.0,
            face_detected=face_detected,
            timestamp=now,
        )
