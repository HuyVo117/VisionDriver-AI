"""Unit tests cho các module cốt lõi của VisionDriver AI."""

from __future__ import annotations

import time
import numpy as np
import pytest

from modules.alert_engine import Alert, AlertEngine, Severity
from phase1_scene_understanding.qa_engine import TrafficQAEngine
from phase2_driver_monitoring.state_classifier import DriverState, DriverStateClassifier
from phase2_driver_monitoring.face_monitor import FaceMetrics


# ================================================================
# AlertEngine Tests
# ================================================================

class TestAlertEngine:
    def test_trigger_returns_alert(self):
        engine = AlertEngine()
        # "obstacle_close" cần min_trigger_count=2, trigger 2 lần
        engine.trigger("obstacle_close", {"distance_m": 2.5})
        alert = engine.trigger("obstacle_close", {"distance_m": 2.5})
        assert alert is not None
        assert alert.severity == Severity.HIGH

    def test_cooldown_suppresses_repeat(self):
        engine = AlertEngine()
        # Trigger collision_imminent (min_trigger_count=1)
        alert1 = engine.trigger("collision_imminent", {"ttc_s": 0.5})
        assert alert1 is not None
        # Ngay lập tức trigger lại → phải bị cooldown
        alert2 = engine.trigger("collision_imminent", {"ttc_s": 0.5})
        assert alert2 is None

    def test_callback_called(self):
        engine = AlertEngine()
        received = []
        engine.on_alert(lambda a: received.append(a))
        engine.trigger("collision_imminent", {"ttc_s": 0.3})
        assert len(received) == 1
        assert received[0].alert_id == "collision_imminent"

    def test_severity_callback_filter(self):
        engine = AlertEngine()
        high_alerts = []
        engine.on_alert(lambda a: high_alerts.append(a), severity=Severity.HIGH)
        # CRITICAL không lọt vào HIGH list
        engine.trigger("collision_imminent", {"ttc_s": 0.3})  # CRITICAL
        assert len(high_alerts) == 0
        # Trigger HIGH
        engine.trigger("obstacle_close", {})
        engine.trigger("obstacle_close", {})
        assert len(high_alerts) == 1

    def test_get_active_alerts(self):
        engine = AlertEngine()
        engine.trigger("collision_imminent", {"ttc_s": 0.2})
        actives = engine.get_active_alerts(within_s=10.0)
        assert len(actives) >= 1

    def test_unknown_alert_id(self):
        engine = AlertEngine()
        result = engine.trigger("non_existent_alert", {})
        assert result is None


# ================================================================
# QA Engine Tests
# ================================================================

class TestTrafficQAEngine:
    def test_exact_match_parking(self):
        qa = TrafficQAEngine()
        result = qa.answer("cấm đỗ xe")
        assert result.confidence >= 0.8
        assert "cấm" in result.answer.lower() or "P.130" in result.answer

    def test_keyword_match_camera(self):
        qa = TrafficQAEngine()
        result = qa.answer("đường này có camera phạt nguội không?")
        assert result.confidence > 0.0
        assert result.source in ("rule", "keyword", "semantic")

    def test_speed_limit_highway(self):
        qa = TrafficQAEngine()
        result = qa.answer("tốc độ tối đa trên cao tốc")
        assert result.confidence > 0.0
        assert "120" in result.answer or "km/h" in result.answer

    def test_unknown_question_returns_fallback(self):
        qa = TrafficQAEngine()
        result = qa.answer("cho tôi hỏi về chính sách bảo hiểm xe")
        assert result.source == "fallback"
        assert result.confidence == 0.0

    def test_query_from_sign(self):
        qa = TrafficQAEngine()
        answer = qa.query_from_sign("no_parking")
        assert answer is not None
        assert len(answer) > 0

    def test_query_from_sign_no_entry(self):
        qa = TrafficQAEngine()
        answer = qa.query_from_sign("P.101")
        assert answer is not None


# ================================================================
# DriverStateClassifier Tests
# ================================================================

def make_metrics(**kwargs) -> FaceMetrics:
    defaults = dict(
        ear=0.30,
        left_ear=0.30,
        right_ear=0.30,
        mar=0.2,
        head_pitch=5.0,
        head_yaw=5.0,
        head_roll=2.0,
        perclos=0.0,
        eyes_closed=False,
        looking_away=False,
        drowsiness_score=0.0,
        face_detected=True,
        timestamp=time.time(),
    )
    defaults.update(kwargs)
    return FaceMetrics(**defaults)


class TestDriverStateClassifier:
    def test_normal_state(self):
        clf = DriverStateClassifier()
        metrics = make_metrics()
        report = clf.update(metrics)
        assert report.state == DriverState.NORMAL

    def test_drowsy_warn(self):
        clf = DriverStateClassifier()
        metrics = make_metrics(drowsiness_score=0.55, ear=0.18, perclos=0.20)
        report = clf.update(metrics)
        assert report.state in (DriverState.DROWSY_WARN, DriverState.DROWSY_CRIT)

    def test_drowsy_critical(self):
        clf = DriverStateClassifier()
        metrics = make_metrics(drowsiness_score=0.80, ear=0.10, perclos=0.40)
        report = clf.update(metrics)
        assert report.state == DriverState.DROWSY_CRIT

    def test_unresponsive_no_face(self):
        clf = DriverStateClassifier()
        # Simulate 6 giây không có mặt
        clf._no_face_since = time.time() - 6.0
        metrics = make_metrics(face_detected=False)
        report = clf.update(metrics)
        assert report.state == DriverState.UNRESPONSIVE

    def test_distracted(self):
        clf = DriverStateClassifier()
        # Simulate 4 giây nhìn đi
        clf._distracted_since = time.time() - 4.0
        metrics = make_metrics(looking_away=True, head_yaw=40.0)
        report = clf.update(metrics)
        assert report.state == DriverState.DISTRACTED

    def test_report_has_recommendation(self):
        clf = DriverStateClassifier()
        metrics = make_metrics()
        report = clf.update(metrics)
        assert isinstance(report.recommended_action, str)
        assert len(report.recommended_action) > 0


# ================================================================
# Integration smoke test
# ================================================================

class TestIntegration:
    def test_alert_engine_with_state_classifier(self):
        """Kiểm tra vòng lặp: metrics → state → alert."""
        clf = DriverStateClassifier()
        engine = AlertEngine()
        alerts_fired = []
        engine.on_alert(lambda a: alerts_fired.append(a))

        # Drowsy warn state
        metrics = make_metrics(drowsiness_score=0.55, ear=0.18, perclos=0.20)
        report = clf.update(metrics)

        # Kích hoạt alert tương ứng
        if report.state == DriverState.DROWSY_WARN:
            for _ in range(6):  # min_trigger_count = 5
                engine.trigger("drowsiness_warning", {})
            assert len(alerts_fired) >= 1
