from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ppe_detection.config import DetectionConfig


def parse_source(value: str) -> int | str:
    """Chuyển chỉ số camera dạng chuỗi thành số nguyên."""
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phát hiện người và vi phạm trang bị bảo hộ trong ảnh, video hoặc webcam."
    )
    parser.add_argument("--source", default="0", help="Đường dẫn ảnh/video hoặc chỉ số camera")
    parser.add_argument("--person-model", required=True, help="Đường dẫn model YOLO phát hiện người")
    parser.add_argument("--ppe-model", required=True, help="Đường dẫn model PPE đã huấn luyện")
    parser.add_argument("--img-size", type=int, default=640, help="Kích thước ảnh đầu vào của YOLO")
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=4,
        help="Chạy phát hiện sau mỗi N khung hình",
    )
    parser.add_argument("--no-beep", action="store_true", help="Tắt âm thanh cảnh báo")
    return parser


def configure_console() -> None:
    """Dùng UTF-8 để thông báo tiếng Việt hiển thị đúng trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    configure_console()
    args = build_parser().parse_args()
    try:
        from ppe_detection.pipeline import PPEPipeline
    except ModuleNotFoundError as error:
        if error.name in {"cv2", "torch", "ultralytics", "numpy"}:
            raise SystemExit(
                f"Thiếu thư viện {error.name!r}. "
                "Hãy chạy: pip install -r requirements.txt"
            ) from None
        raise

    config = DetectionConfig(
        person_model_path=Path(args.person_model),
        ppe_model_path=Path(args.ppe_model),
        image_size=args.img_size,
        detection_interval=args.detect_interval,
        enable_beep=not args.no_beep,
    )
    PPEPipeline(config).run(parse_source(args.source))


if __name__ == "__main__":
    main()
