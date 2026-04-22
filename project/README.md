# VisionDriver AI

<div align="center">

**Hệ thống AI hỗ trợ lái xe thông minh**  
*Phát hiện nguy hiểm · Hỏi đáp giao thông · Giám sát tài xế*

</div>

---

## Tổng Quan

VisionDriver AI gồm **2 Phase** chạy song song:

| Module | Mô tả |
|---|---|
| **Phase 1** | Camera đường → Phát hiện chướng ngại vật, biển báo, cảnh báo va chạm, hỏi đáp luật giao thông |
| **Phase 2** | Camera tài xế → Phát hiện buồn ngủ, đột quỵ, hoảng loạn → Gọi cấp cứu, dừng xe an toàn |

---

## Cấu Trúc Dự Án

```
project/
├── api/                        # Scripts chạy và training
│   ├── collect_data.py         # Thu thập dữ liệu từ CARLA
│   ├── train_imitation.py      # Train model Phase 1 (imitation learning)
│   ├── train_driver_state.py   # Train model Phase 2 (driver state)
│   ├── adas_pipeline.py        # Pipeline ADAS chính (Phase 1)
│   ├── run_adas_realtime.py    # Chạy Phase 1 realtime (cần CARLA)
│   ├── run_driver_monitor.py   # Chạy Phase 2 realtime (cần webcam)
│   └── run_unified.py          # Chạy cả 2 Phase song song
│
├── modules/                    # Core modules
│   ├── detection.py            # YOLOv8 object detector
│   ├── lane_processing.py      # Lane detection
│   ├── sign_classifier.py      # Traffic sign recognition
│   ├── distance_measurer.py    # Khoảng cách & TTC
│   └── alert_engine.py         # ⭐ Alert engine (cooldown, severity)
│
├── phase1_scene_understanding/ # Phase 1 modules
│   ├── scene_engine.py         # Scene context builder
│   ├── qa_engine.py            # ⭐ QA Engine (hỏi đáp luật giao thông)
│   ├── knowledge_base.py       # Knowledge base cơ bản
│   └── utils.py
│
├── phase2_driver_monitoring/   # ⭐ Phase 2 modules (MỚI)
│   ├── face_monitor.py         # EAR, PERCLOS, head pose (MediaPipe)
│   ├── state_classifier.py     # State machine: NORMAL → EMERGENCY
│   └── emergency_manager.py    # Gọi cấp cứu, SMS, log sự cố
│
├── agents/                     # AI Agent layer
├── models/                     # Model weights (để vào đây)
├── data/                       # Datasets
├── scripts/                    # Utility scripts
│   └── download_datasets.py    # ⭐ Tự động tải datasets
├── utils/
├── requirements.txt
└── main.py
```

---

## Cài Đặt Nhanh

### 1. Tạo môi trường ảo

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tải Datasets

```powershell
# Xem hướng dẫn và tải tất cả
python scripts/download_datasets.py --all

# Hoặc từng phần
python scripts/download_datasets.py --signs      # Biển báo VN
python scripts/download_datasets.py --drowsiness # Buồn ngủ
python scripts/download_datasets.py --emotion    # Cảm xúc
```

### 3. Cài CARLA (Phase 1)

```powershell
# Tải từ https://carla.org/ (version 0.9.14)
# Chạy server
CARLA_0.9.14\CarlaUE4.exe

# Cài Python API
pip install carla==0.9.14
```

---

## Chạy Dự Án

### Phase 1 — ADAS (cần CARLA)

```powershell
# Thu thập dữ liệu
python api/collect_data.py

# Train model
python api/train_imitation.py

# Chạy realtime
python api/run_adas_realtime.py
python api/run_adas_realtime.py --headless   # Không cần cửa sổ
```

### Phase 2 — Driver Monitor (chỉ cần webcam)

```powershell
# Chạy với camera mặc định (index 0)
python api/run_driver_monitor.py --camera 0

# Thêm số liên hệ khẩn cấp
python api/run_driver_monitor.py --camera 0 --contact "+84901234567"

# Train model nhận diện trạng thái
python api/train_driver_state.py --data-dir data/driver_state/ --epochs 30
```

### Unified — Cả 2 Phase

```powershell
# Camera 0 = road, Camera 1 = driver
python api/run_unified.py --road-cam 0 --driver-cam 1
```

### Hỏi đáp giao thông (test nhanh)

```python
from phase1_scene_understanding.qa_engine import TrafficQAEngine

qa = TrafficQAEngine()
print(qa.answer("Đường này có cấm đỗ xe không?"))
print(qa.answer("Tốc độ tối đa trên cao tốc là bao nhiêu?"))
print(qa.answer("Vượt đèn đỏ bị phạt bao nhiêu tiền?"))
```

---

## Cấu Hình Emergency (Phase 2)

```python
from phase2_driver_monitoring.emergency_manager import EmergencyConfig

config = EmergencyConfig(
    emergency_number="115",                     # Số cấp cứu
    contact_numbers=["+84901234567"],            # Số gia đình/bạn bè
    twilio_account_sid="ACxxxxxxxx",            # Twilio (gọi điện thật)
    twilio_auth_token="xxxxxxxx",
    twilio_from_number="+1234567890",
    mock_mode=False,                             # True = chỉ log, không gọi thật
)
```

---

## Biến Môi Trường (.env)

```env
ROBOFLOW_API_KEY=your_roboflow_key
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_api_key
GEMINI_API_KEY=your_gemini_key
TWILIO_ACCOUNT_SID=ACxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890
```

---

## Roadmap

Xem [ROADMAP_PHASE1_PHASE2.md](ROADMAP_PHASE1_PHASE2.md) để biết kế hoạch chi tiết.

| Phase | Timeline | Trạng thái |
|---|---|---|
| Phase 1: Scene Understanding + QA | Tuần 1–6 | 🟡 Đang xây dựng |
| Phase 2: Driver Monitoring | Tuần 7–12 | 🟡 Skeleton sẵn sàng |
| Integration: Unified Pipeline | Tuần 11–12 | ⚪ Chưa bắt đầu |
