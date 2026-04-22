"""
Script tự động tải các dataset cần thiết cho VisionDriver AI.

Datasets:
  1. VN Traffic Signs — từ Kaggle/Roboflow
  2. Drowsiness dataset (MRL Eye / CEW)
  3. AffectNet subset (emotion recognition)

Usage:
  python scripts/download_datasets.py --all
  python scripts/download_datasets.py --signs
  python scripts/download_datasets.py --drowsiness
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

DATA_DIR = Path("data")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_vn_traffic_signs_roboflow() -> None:
    """Tải dataset biển báo giao thông VN từ Roboflow."""
    print("[1/3] Tải VN Traffic Signs từ Roboflow...")
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY"))
        # Dataset biển báo VN phổ biến nhất trên Roboflow
        project = rf.workspace("traffic-sign-vn").project("vietnam-traffic-signs")
        dataset = project.version(1).download("yolov8", location=str(DATA_DIR / "vn_traffic_signs"))
        print(f"   ✅ Đã tải về: {DATA_DIR / 'vn_traffic_signs'}")
    except Exception as e:
        print(f"   ❌ Lỗi Roboflow: {e}")
        print("   👉 Tải thủ công tại: https://roboflow.com/search?q=vietnam+traffic+signs")


def download_vn_traffic_signs_kaggle() -> None:
    """Tải dataset biển báo VN từ Kaggle."""
    print("[1/3b] Tải VN Traffic Signs từ Kaggle...")
    try:
        import kagglehub

        path = kagglehub.dataset_download("valentynsichkar/traffic-signs-preprocessed")
        print(f"   ✅ Đã tải về: {path}")
    except Exception as e:
        print(f"   ❌ Lỗi Kaggle: {e}")
        print("   👉 Đảm bảo đã setup kaggle.json: https://www.kaggle.com/docs/api")


def download_mrl_eye_dataset() -> None:
    """Tải MRL Eye Dataset cho drowsiness detection."""
    print("[2/3] Thông tin MRL Eye Dataset...")
    print("   ℹ️  MRL Eye Dataset cần tải thủ công tại:")
    print("   👉 http://mrl.cs.vsb.cz/eyedataset")
    print("   👉 Hoặc tìm trên Kaggle: 'MRL eye dataset drowsiness'")
    print("   📁 Giải nén vào: data/mrl_eye/")

    # Tải CEW dataset từ Kaggle (Closed Eyes in the Wild)
    try:
        import kagglehub
        print("   Đang thử tải CEW dataset...")
        path = kagglehub.dataset_download("kutaykutlu/drowsiness-detection")
        print(f"   ✅ CEW dataset: {path}")
    except Exception as e:
        print(f"   ❌ CEW Kaggle: {e}")


def download_nthu_drowsiness_info() -> None:
    """Thông tin về NTHU Drowsy Driver Dataset."""
    print("[2/3b] NTHU Drowsy Driver Dataset...")
    print("   ℹ️  NTHU dataset cần đăng ký tại:")
    print("   👉 http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/")
    print("   📁 Sau khi tải, giải nén vào: data/nthu_driver/")


def download_emotion_dataset() -> None:
    """Tải dataset nhận diện cảm xúc."""
    print("[3/3] Tải Emotion Dataset...")
    try:
        import kagglehub
        # FER2013 - dataset nhận diện cảm xúc phổ biến
        path = kagglehub.dataset_download("msambare/fer2013")
        print(f"   ✅ FER2013: {path}")
    except Exception as e:
        print(f"   ❌ FER2013: {e}")
        print("   👉 Tải thủ công: https://www.kaggle.com/datasets/msambare/fer2013")


def check_carla_installation() -> None:
    """Kiểm tra CARLA đã cài chưa."""
    print("\n[INFO] Kiểm tra CARLA...")
    try:
        import carla  # noqa: F401
        print("   ✅ CARLA Python API đã sẵn sàng.")
    except ImportError:
        print("   ❌ CARLA chưa được cài đặt.")
        print("   👉 Tải CARLA 0.9.14: https://carla.org/")
        print("   👉 Windows: Tải CARLA_0.9.14.zip và chạy CarlaUE4.exe")
        print("   👉 Python API: pip install carla==0.9.14")


def print_setup_summary() -> None:
    print("\n" + "="*60)
    print("📋 HƯỚNG DẪN SETUP NHANH")
    print("="*60)
    print("""
1. Tạo môi trường ảo:
   python -m venv venv
   venv\\Scripts\\activate    (Windows)

2. Cài đặt dependencies:
   pip install -r requirements.txt

3. Setup API keys (thêm vào .env):
   ROBOFLOW_API_KEY=your_key_here
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   GEMINI_API_KEY=your_gemini_key    (optional - cho LLM QA)
   TWILIO_ACCOUNT_SID=...            (optional - cho emergency call)
   TWILIO_AUTH_TOKEN=...
   TWILIO_PHONE_NUMBER=...

4. Cài CARLA 0.9.14:
   - Tải từ https://carla.org/
   - Chạy: CARLA_0.9.14\\CarlaUE4.exe
   - pip install carla==0.9.14

5. Chạy Phase 1 (cần CARLA):
   python api/run_adas_realtime.py

6. Chạy Phase 2 (chỉ cần webcam):
   python api/run_driver_monitor.py --camera 0

7. Hỏi AI về giao thông (test):
   python -c "
   from phase1_scene_understanding.qa_engine import TrafficQAEngine
   qa = TrafficQAEngine()
   print(qa.answer('Đường này có cấm đỗ xe không?'))
   "
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VisionDriver AI datasets")
    parser.add_argument("--all", action="store_true", help="Tải tất cả datasets")
    parser.add_argument("--signs", action="store_true", help="Tải VN traffic signs")
    parser.add_argument("--drowsiness", action="store_true", help="Tải drowsiness datasets")
    parser.add_argument("--emotion", action="store_true", help="Tải emotion dataset")
    parser.add_argument("--check", action="store_true", help="Kiểm tra dependencies")
    args = parser.parse_args()

    ensure_dir(DATA_DIR)

    if args.all or args.signs:
        download_vn_traffic_signs_roboflow()
        download_vn_traffic_signs_kaggle()

    if args.all or args.drowsiness:
        download_mrl_eye_dataset()
        download_nthu_drowsiness_info()

    if args.all or args.emotion:
        download_emotion_dataset()

    if args.check or args.all:
        check_carla_installation()

    if not any([args.all, args.signs, args.drowsiness, args.emotion, args.check]):
        parser.print_help()
        print_setup_summary()
    else:
        print_setup_summary()


if __name__ == "__main__":
    main()
