# VisionDriver AI Roadmap (Phase 1 -> Phase 2)

## Objectives

- Build a reliable driving copilot for scene understanding, Q&A, and real-time alerts.
- Extend into driver monitoring with safety-focused failsafe actions.
- Keep architecture modular so each capability can be tested and upgraded independently.

## Delivery Plan

### Phase 1 (Weeks 1-4): Driving Assistant

1. Week 1 - Data and baseline pipelines

- Finalize data collection with CARLA RGB + control labels.
- Validate train/inference scripts in `api/`.
- Define schema for scene context and detected objects.

2. Week 2 - Scene understanding core

- Integrate obstacle and lane processing modules.
- Add speed-limit and road-context extraction.
- Add confidence thresholds and alert cooldown rules.

3. Week 3 - Knowledge QA and rule engine

- Build traffic sign and parking rule knowledge base.
- Add deterministic QA for parking/speed/camera questions.
- Add optional LLM fallback for open-domain questions.

4. Week 4 - Integration and evaluation

- End-to-end run: camera frame -> context -> alerts + QA.
- Track metrics: alert precision, false alerts, response latency.
- Freeze Phase 1 baseline tag.

### Phase 2 (Weeks 5-10): Driver Monitoring

1. Weeks 5-6 - Driver state detection

- Face/eye/head-pose pipeline for drowsiness estimation.
- Attention classification (road vs distraction).
- Driver-state confidence and temporal smoothing.

2. Weeks 7-8 - Incident detection

- Add rules for prolonged eye closure and face-missing events.
- Add emergency severity levels and escalation policy.
- Add structured incident logs to memory store.

3. Weeks 9-10 - Failsafe orchestration

- Implement staged failsafe: alert -> slow down -> stop safely.
- Integrate hooks for emergency contacts (mock first).
- Validate behavior in controlled CARLA scenarios.

## Cross-cutting Workstreams

- Testing: unit tests for routing, QA rules, memory retrieval, alert logic.
- Observability: structured logs for model outputs and decisions.
- Safety: conservative defaults and explicit confidence checks.
- Performance: target <= 100 ms/frame for Phase 1 core loop.

## Definition of Done

### Phase 1 DoD

- Real-time scene context generated from incoming frames.
- Alerts produced with cooldown and severity.
- QA answers for parking, speed, and camera queries.
- Reproducible scripts for collect/train/inference.

### Phase 2 DoD

- Driver state (eyes/head/attention) available every frame window.
- Health incident detector produces actionable severity levels.
- Failsafe actions can be simulated and logged end-to-end.

## Suggested Execution Commands

```powershell
# 1) Data collection
python api/collect_data.py

# 2) Training
python api/train_imitation.py

# 3) Inference in CARLA
python api/run_trained_model.py
```

## Risks and Mitigations

- Domain gap (sim -> real): use augmentation and staged real-data validation.
- False positives in alerts: require confidence + temporal consistency.
- Latency spikes: profile per-module and cache expensive lookups.
- Safety-critical errors: fallback to conservative control recommendations.
