"""
Phase 2: Driver State Machine & Incident Classifier

States:
  NORMAL       → Tất cả OK
  DISTRACTED   → Nhìn đi nơi khác (> 3s)
  DROWSY_WARN  → Buồn ngủ nhẹ (cảnh báo)
  DROWSY_CRIT  → Buồn ngủ nặng (yêu cầu dừng xe)
  UNRESPONSIVE → Không phản ứng (face absent > 5s, hoặc mắt nhắm > 8s)
  EMERGENCY    → Đột quỵ / tai nạn → kích hoạt Failsafe
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .face_monitor import FaceMetrics


class DriverState(Enum):
    NORMAL = auto()
    DISTRACTED = auto()
    DROWSY_WARN = auto()
    DROWSY_CRIT = auto()
    UNRESPONSIVE = auto()
    EMERGENCY = auto()


@dataclass
class IncidentReport:
    """Báo cáo sự cố từ Driver State Machine."""
    state: DriverState
    severity_score: float          # 0–1
    duration_s: float              # Thời gian ở trạng thái này
    recommended_action: str
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)


class DriverStateClassifier:
    """
    State machine phân loại trạng thái tài xế.

    Thresholds có thể điều chỉnh dựa trên tuning thực tế.
    """

    # Thresholds
    DROWSY_SCORE_WARN = 0.45
    DROWSY_SCORE_CRIT = 0.70
    UNRESPONSIVE_NO_FACE_S = 5.0    # Không nhận ra mặt > 5s
    UNRESPONSIVE_EYES_CLOSED_S = 8.0  # Mắt nhắm > 8s
    DISTRACTED_YAW_LIMIT = 30.0
    DISTRACTED_MIN_S = 3.0           # Nhìn đi > 3s

    def __init__(self) -> None:
        self._current_state = DriverState.NORMAL
        self._state_entry_time = time.time()
        self._no_face_since: Optional[float] = None
        self._eyes_closed_since: Optional[float] = None
        self._distracted_since: Optional[float] = None

    def _time_in_state(self) -> float:
        return time.time() - self._state_entry_time

    def _transition(self, new_state: DriverState) -> None:
        if new_state != self._current_state:
            self._current_state = new_state
            self._state_entry_time = time.time()

    def update(self, metrics: FaceMetrics) -> IncidentReport:
        """Cập nhật state machine với FaceMetrics mới nhất."""
        now = metrics.timestamp

        # Không nhận ra mặt
        if not metrics.face_detected:
            if self._no_face_since is None:
                self._no_face_since = now
            no_face_duration = now - self._no_face_since
        else:
            self._no_face_since = None
            no_face_duration = 0.0

        # Mắt nhắm liên tục
        if metrics.eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
            closed_duration = now - self._eyes_closed_since
        else:
            self._eyes_closed_since = None
            closed_duration = 0.0

        # Nhìn đi nơi khác
        if metrics.looking_away:
            if self._distracted_since is None:
                self._distracted_since = now
            distracted_duration = now - self._distracted_since
        else:
            self._distracted_since = None
            distracted_duration = 0.0

        # ---- State machine logic ----
        new_state = self._current_state

        if no_face_duration > self.UNRESPONSIVE_NO_FACE_S:
            new_state = DriverState.UNRESPONSIVE
        elif closed_duration > self.UNRESPONSIVE_EYES_CLOSED_S:
            new_state = DriverState.EMERGENCY  # Bất tỉnh/đột quỵ
        elif metrics.drowsiness_score >= self.DROWSY_SCORE_CRIT:
            new_state = DriverState.DROWSY_CRIT
        elif metrics.drowsiness_score >= self.DROWSY_SCORE_WARN:
            new_state = DriverState.DROWSY_WARN
        elif distracted_duration > self.DISTRACTED_MIN_S:
            new_state = DriverState.DISTRACTED
        else:
            new_state = DriverState.NORMAL

        self._transition(new_state)

        # Build report
        action = self._recommended_action(
            new_state,
            no_face_duration=no_face_duration,
            closed_duration=closed_duration,
        )

        return IncidentReport(
            state=new_state,
            severity_score=metrics.drowsiness_score,
            duration_s=self._time_in_state(),
            recommended_action=action,
            timestamp=now,
            details={
                "ear": metrics.ear,
                "perclos": metrics.perclos,
                "head_pitch": metrics.head_pitch,
                "head_yaw": metrics.head_yaw,
                "no_face_s": no_face_duration,
                "eyes_closed_s": closed_duration,
                "distracted_s": distracted_duration,
            },
        )

    def _recommended_action(
        self,
        state: DriverState,
        no_face_duration: float = 0.0,
        closed_duration: float = 0.0,
    ) -> str:
        actions = {
            DriverState.NORMAL: "Tiếp tục lái xe an toàn.",
            DriverState.DISTRACTED: (
                "⚠️ Chú ý đường trước mặt! Mắt đang hướng ra khỏi đường."
            ),
            DriverState.DROWSY_WARN: (
                "😴 Bạn có vẻ buồn ngủ. Hãy mở cửa sổ, nghe nhạc sôi động "
                "hoặc dừng xe tại trạm dừng gần nhất để nghỉ ngơi."
            ),
            DriverState.DROWSY_CRIT: (
                "🚨 NGUY HIỂM! Dừng xe ngay bên lề đường an toàn. "
                "Đừng tiếp tục lái khi buồn ngủ nặng!"
            ),
            DriverState.UNRESPONSIVE: (
                "🆘 Không nhận ra tài xế! Đang kích hoạt cảnh báo khẩn cấp... "
                f"(Mất nhận dạng: {no_face_duration:.0f}s)"
            ),
            DriverState.EMERGENCY: (
                "🆘 KHẨN CẤP! Phát hiện dấu hiệu bất tỉnh hoặc đột quỵ. "
                "Đang gọi cấp cứu 115 và thông báo liên hệ khẩn cấp!"
            ),
        }
        return actions.get(state, "Không xác định.")
