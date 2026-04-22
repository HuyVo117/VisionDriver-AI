"""
Phase 2: Emergency Manager
Xử lý các tình huống khẩn cấp: gọi cứu thương, SMS, GPS log.

Hoạt động theo pipeline:
  1. Nhận IncidentReport từ DriverStateClassifier
  2. Quyết định action dựa trên severity
  3. Thực thi: gọi điện (Twilio), SMS, ghi log, phát lệnh dừng xe
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class FailsafeAction(Enum):
    AUDIO_ALERT = auto()
    SLOW_DOWN = auto()
    PULL_OVER = auto()
    CALL_EMERGENCY = auto()
    SMS_CONTACT = auto()
    HAZARD_LIGHTS = auto()


@dataclass
class EmergencyConfig:
    """Cấu hình Emergency Manager."""
    # Twilio (để gọi điện/SMS thực - cần tài khoản Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    # Số điện thoại khẩn cấp
    emergency_number: str = "115"         # Số cấp cứu VN
    contact_numbers: List[str] = field(default_factory=list)

    # GPS (mock hoặc từ sensor thực)
    gps_lat: float = 10.7769
    gps_lon: float = 106.7009

    # Log
    incident_log_path: str = "logs/incidents.jsonl"

    # Mock mode (khi chưa có Twilio account thực)
    mock_mode: bool = True


@dataclass
class EmergencyEvent:
    """Bản ghi một sự kiện khẩn cấp."""
    event_type: str
    actions_taken: List[str]
    gps_lat: float
    gps_lon: float
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)


class EmergencyManager:
    """
    Quản lý toàn bộ hành động khẩn cấp.

    Usage:
        config = EmergencyConfig(
            contact_numbers=["+84901234567"],
            emergency_number="115",
            mock_mode=True,   # Set False khi có Twilio thật
        )
        em = EmergencyManager(config)

        # Hook để điều khiển xe (optional)
        em.on_slow_down(lambda speed: vehicle.set_target_speed(speed))
        em.on_pull_over(lambda: vehicle.activate_safe_stop())

        # Kích hoạt khi phát hiện khẩn cấp
        em.handle_emergency(incident_report)
    """

    def __init__(self, config: Optional[EmergencyConfig] = None) -> None:
        self.config = config or EmergencyConfig()
        self._slow_down_fn: Optional[Callable[[float], None]] = None
        self._pull_over_fn: Optional[Callable[[], None]] = None
        self._events: List[EmergencyEvent] = []
        self._last_call_time: float = 0.0
        self._call_cooldown_s: float = 30.0   # Không gọi lặp trong 30s

        os.makedirs(os.path.dirname(self.config.incident_log_path), exist_ok=True)

    def on_slow_down(self, callback: Callable[[float], None]) -> None:
        """Đăng ký hook để giảm tốc (nhận target speed km/h)."""
        self._slow_down_fn = callback

    def on_pull_over(self, callback: Callable[[], None]) -> None:
        """Đăng ký hook để dừng xe vào lề."""
        self._pull_over_fn = callback

    def handle_emergency(self, state: str, details: dict | None = None) -> EmergencyEvent:
        """
        Xử lý tình huống khẩn cấp.

        Args:
            state: 'DROWSY_WARN', 'DROWSY_CRIT', 'UNRESPONSIVE', 'EMERGENCY'
            details: Thông tin bổ sung từ detector
        """
        actions_taken: List[str] = []

        if state == "DROWSY_WARN":
            actions_taken.append(self._audio_alert("Cảnh báo buồn ngủ! Hãy nghỉ ngơi."))

        elif state == "DROWSY_CRIT":
            actions_taken.append(self._audio_alert("NGUY HIỂM! Dừng xe ngay!"))
            actions_taken.append(self._slow_down(30.0))

        elif state == "UNRESPONSIVE":
            actions_taken.append(self._audio_alert("Cảnh báo khẩn cấp! Tài xế không phản ứng!"))
            actions_taken.append(self._slow_down(10.0))
            actions_taken.append(self._sms_contact("VisionDriver AI: Tài xế không phản ứng! Vị trí: " + self._gps_link()))

        elif state == "EMERGENCY":
            actions_taken.append(self._audio_alert("KHẨN CẤP! Đang gọi cấp cứu 115!"))
            actions_taken.append(self._pull_over_action())
            actions_taken.append(self._call_emergency())
            actions_taken.append(self._sms_contact(
                f"🚨 VisionDriver AI KHẨN CẤP: Tài xế cần cấp cứu!\n"
                f"Vị trí: {self._gps_link()}\n"
                f"Thời gian: {time.strftime('%H:%M %d/%m/%Y')}"
            ))

        event = EmergencyEvent(
            event_type=state,
            actions_taken=[a for a in actions_taken if a],
            gps_lat=self.config.gps_lat,
            gps_lon=self.config.gps_lon,
            details=details or {},
        )
        self._events.append(event)
        self._log_event(event)
        return event

    def _audio_alert(self, message: str) -> str:
        """Phát cảnh báo âm thanh."""
        logger.warning("[AUDIO ALERT] %s", message)
        try:
            import threading
            from gtts import gTTS
            import tempfile
            import os as _os

            def _play() -> None:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        tts = gTTS(text=message, lang="vi")
                        tts.save(f.name)
                        _os.system(f'start /min "" "{f.name}"')  # Windows
                except Exception as e:
                    logger.warning("TTS failed: %s", e)

            threading.Thread(target=_play, daemon=True).start()
        except ImportError:
            pass  # gTTS chưa cài
        return f"audio_alert: {message}"

    def _slow_down(self, target_kmh: float) -> str:
        """Giảm tốc độ xe."""
        if self._slow_down_fn:
            try:
                self._slow_down_fn(target_kmh)
                return f"slow_down_to_{target_kmh}kmh"
            except Exception as e:
                logger.error("slow_down hook error: %s", e)
        return f"slow_down_requested_{target_kmh}kmh (no hook)"

    def _pull_over_action(self) -> str:
        """Dừng xe vào lề."""
        if self._pull_over_fn:
            try:
                self._pull_over_fn()
                return "pull_over_activated"
            except Exception as e:
                logger.error("pull_over hook error: %s", e)
        return "pull_over_requested (no hook)"

    def _call_emergency(self) -> str:
        """Gọi điện khẩn cấp (Twilio hoặc mock)."""
        now = time.time()
        if now - self._last_call_time < self._call_cooldown_s:
            return "call_skipped_cooldown"

        self._last_call_time = now
        number = self.config.emergency_number

        if self.config.mock_mode:
            logger.critical("[MOCK CALL] Gọi %s – Đây là mock, chưa gọi thật!", number)
            return f"mock_call_{number}"

        # Twilio thực
        try:
            from twilio.rest import Client
            client = Client(
                self.config.twilio_account_sid,
                self.config.twilio_auth_token,
            )
            call = client.calls.create(
                to=number,
                from_=self.config.twilio_from_number,
                twiml=f"<Response><Say language='vi-VN'>Đây là cảnh báo khẩn cấp từ VisionDriver AI. Tài xế cần cấp cứu ngay tại vị trí {self._gps_link()}.</Say></Response>",
            )
            logger.critical("[CALL] Twilio call SID: %s", call.sid)
            return f"call_initiated_{call.sid}"
        except Exception as e:
            logger.error("Twilio call failed: %s", e)
            return f"call_failed: {e}"

    def _sms_contact(self, message: str) -> str:
        """Gửi SMS đến số liên hệ khẩn cấp."""
        results = []
        for number in self.config.contact_numbers:
            if self.config.mock_mode:
                logger.critical("[MOCK SMS] → %s: %s", number, message)
                results.append(f"mock_sms_{number}")
                continue

            try:
                from twilio.rest import Client
                client = Client(
                    self.config.twilio_account_sid,
                    self.config.twilio_auth_token,
                )
                msg = client.messages.create(
                    to=number,
                    from_=self.config.twilio_from_number,
                    body=message,
                )
                results.append(f"sms_sent_{msg.sid}")
            except Exception as e:
                logger.error("SMS to %s failed: %s", number, e)
                results.append(f"sms_failed_{number}")

        return "; ".join(results) if results else "no_contacts_configured"

    def _gps_link(self) -> str:
        lat, lon = self.config.gps_lat, self.config.gps_lon
        return f"https://maps.google.com/?q={lat},{lon}"

    def _log_event(self, event: EmergencyEvent) -> None:
        try:
            with open(self.config.incident_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to log incident: %s", e)
