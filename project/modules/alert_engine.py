"""
VisionDriver AI — Alert Engine
Quản lý cảnh báo với cooldown, severity levels và temporal smoothing.

Severity Levels:
  LOW      → Thông báo nhẹ (voice alert)
  MEDIUM   → Cảnh báo rõ ràng + âm thanh
  HIGH     → Cảnh báo khẩn + haptic (nếu có)
  CRITICAL → Kích hoạt Failsafe (gọi cứu thương, dừng xe)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Deque, Dict, List, Optional


class Severity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    """Một sự kiện cảnh báo."""
    alert_id: str
    severity: Severity
    message: str
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    data: dict = field(default_factory=dict)


@dataclass
class AlertRule:
    """Quy tắc sinh cảnh báo với cooldown."""
    alert_id: str
    severity: Severity
    message_template: str
    cooldown_s: float = 5.0          # Không nhắc lại trong N giây
    min_trigger_count: int = 1       # Phải kích hoạt N lần liên tiếp
    source: str = "system"


class AlertEngine:
    """
    Engine quản lý cảnh báo tập trung.

    Usage:
        engine = AlertEngine()
        engine.on_alert(Severity.HIGH, lambda a: print(a.message))

        # Trong pipeline:
        engine.trigger("obstacle_close", data={"distance_m": 2.5})
    """

    # Quy tắc mặc định cho Phase 1 (Scene Understanding)
    DEFAULT_RULES: List[AlertRule] = [
        AlertRule(
            alert_id="obstacle_close",
            severity=Severity.HIGH,
            message_template="⚠️ Chướng ngại vật nguy hiểm phía trước! Khoảng cách {distance_m:.1f}m",
            cooldown_s=3.0,
            min_trigger_count=2,
            source="scene",
        ),
        AlertRule(
            alert_id="collision_imminent",
            severity=Severity.CRITICAL,
            message_template="🚨 NGUY HIỂM! Va chạm sắp xảy ra! TTC = {ttc_s:.1f}s",
            cooldown_s=1.5,
            min_trigger_count=1,
            source="scene",
        ),
        AlertRule(
            alert_id="lane_departure",
            severity=Severity.MEDIUM,
            message_template="↔️ Cảnh báo: Xe lệch khỏi làn đường!",
            cooldown_s=4.0,
            min_trigger_count=3,
            source="lane",
        ),
        AlertRule(
            alert_id="speed_limit_exceeded",
            severity=Severity.LOW,
            message_template="🚗 Vượt tốc độ giới hạn {limit_kmh} km/h (hiện tại: {speed_kmh:.0f} km/h)",
            cooldown_s=10.0,
            min_trigger_count=1,
            source="scene",
        ),
        AlertRule(
            alert_id="unknown_object",
            severity=Severity.MEDIUM,
            message_template="❓ Phát hiện vật thể lạ phía trước, hãy chú ý!",
            cooldown_s=5.0,
            min_trigger_count=2,
            source="scene",
        ),
        # Phase 2 rules
        AlertRule(
            alert_id="drowsiness_warning",
            severity=Severity.MEDIUM,
            message_template="😴 Cảnh báo: Bạn có vẻ buồn ngủ! Hãy dừng xe nghỉ ngơi.",
            cooldown_s=8.0,
            min_trigger_count=5,
            source="driver",
        ),
        AlertRule(
            alert_id="drowsiness_critical",
            severity=Severity.HIGH,
            message_template="🚨 NGUY HIỂM: Tài xế đang ngủ gật! Đề nghị dừng xe ngay!",
            cooldown_s=3.0,
            min_trigger_count=10,
            source="driver",
        ),
        AlertRule(
            alert_id="driver_unresponsive",
            severity=Severity.CRITICAL,
            message_template="🆘 KHẨN CẤP: Tài xế không phản ứng! Đang liên hệ cấp cứu...",
            cooldown_s=1.0,
            min_trigger_count=1,
            source="driver",
        ),
        AlertRule(
            alert_id="medical_emergency",
            severity=Severity.CRITICAL,
            message_template="🆘 CẤP CỨU Y TẾ: Phát hiện dấu hiệu đột quỵ! Đang gọi 115...",
            cooldown_s=0.5,
            min_trigger_count=1,
            source="driver",
        ),
        AlertRule(
            alert_id="panic_detected",
            severity=Severity.HIGH,
            message_template="⚠️ Phát hiện tài xế đang hoảng loạn. Hít thở sâu và bình tĩnh!",
            cooldown_s=5.0,
            min_trigger_count=3,
            source="driver",
        ),
    ]

    def __init__(self, rules: Optional[List[AlertRule]] = None) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._last_fired: Dict[str, float] = {}
        self._trigger_counts: Dict[str, Deque[float]] = {}
        self._callbacks: Dict[Severity, List[Callable[[Alert], None]]] = {
            s: [] for s in Severity
        }
        self._global_callbacks: List[Callable[[Alert], None]] = []
        self._history: Deque[Alert] = deque(maxlen=200)

        for rule in (rules or self.DEFAULT_RULES):
            self.register_rule(rule)

    def register_rule(self, rule: AlertRule) -> None:
        self._rules[rule.alert_id] = rule
        self._trigger_counts[rule.alert_id] = deque(maxlen=30)

    def on_alert(
        self,
        callback: Callable[[Alert], None],
        severity: Optional[Severity] = None,
    ) -> None:
        """Đăng ký callback khi có alert. severity=None → nhận tất cả."""
        if severity is None:
            self._global_callbacks.append(callback)
        else:
            self._callbacks[severity].append(callback)

    def trigger(self, alert_id: str, data: Optional[dict] = None) -> Optional[Alert]:
        """
        Kích hoạt một alert_id. Trả về Alert object nếu thực sự được phát,
        None nếu bị cooldown hoặc chưa đủ trigger count.
        """
        rule = self._rules.get(alert_id)
        if rule is None:
            return None

        now = time.time()
        counts = self._trigger_counts[alert_id]
        counts.append(now)

        # Temporal smoothing: đếm số trigger trong cửa sổ 2s
        recent = sum(1 for t in counts if now - t <= 2.0)
        if recent < rule.min_trigger_count:
            return None

        # Cooldown check
        last = self._last_fired.get(alert_id, 0.0)
        if now - last < rule.cooldown_s:
            return None

        self._last_fired[alert_id] = now

        # Format message
        ctx = data or {}
        try:
            message = rule.message_template.format(**ctx)
        except KeyError:
            message = rule.message_template

        alert = Alert(
            alert_id=alert_id,
            severity=rule.severity,
            message=message,
            timestamp=now,
            source=rule.source,
            data=ctx,
        )
        self._history.append(alert)
        self._dispatch(alert)
        return alert

    def _dispatch(self, alert: Alert) -> None:
        for cb in self._callbacks.get(alert.severity, []):
            try:
                cb(alert)
            except Exception:  # noqa: BLE001
                pass
        for cb in self._global_callbacks:
            try:
                cb(alert)
            except Exception:  # noqa: BLE001
                pass

    def get_history(self, last_n: int = 20) -> List[Alert]:
        return list(self._history)[-last_n:]

    def get_active_alerts(self, within_s: float = 10.0) -> List[Alert]:
        now = time.time()
        return [a for a in self._history if now - a.timestamp <= within_s]
