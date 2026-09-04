"""Script huấn luyện mô hình YOLO PPE với theo dõi metadata thí nghiệm."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train")


def get_git_commit() -> str:
    """Lấy hash commit git hiện tại nếu có."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def record_environment_metadata(config_path: Path, output_dir: Path) -> dict:
    """Ghi lại metadata đầy đủ của môi trường phần cứng và thư viện phần mềm."""
    import ultralytics

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_file": str(config_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "git_commit": get_git_commit(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = output_dir / "experiment_env.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    LOGGER.info("Ghi nhận metadata thí nghiệm tại: %s", meta_file)
    return metadata


def train_model(config_file: str) -> None:
    """Thực thi huấn luyện YOLO dựa trên file cấu hình YAML."""
    from ultralytics import YOLO

    cfg_path = Path(config_file)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_file}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "yolov8n_ppe_exp")
    output_dir = Path("runs") / "train" / exp_name
    record_environment_metadata(cfg_path, output_dir)

    model_type = cfg.get("model_type", "yolov8n.pt")
    LOGGER.info("Khởi tạo mô hình %s...", model_type)
    model = YOLO(model_type)

    data_cfg = cfg.get("data_config", "training/data.yaml")
    epochs = cfg.get("epochs", 100)
    imgsz = cfg.get("image_size", 640)
    batch = cfg.get("batch_size", 16)
    seed = cfg.get("seed", 42)

    LOGGER.info("Bắt đầu huấn luyện mô hình [%s] trong %d epochs...", exp_name, epochs)
    model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        name=exp_name,
        project="runs/train",
        exist_ok=True,
    )
    LOGGER.info("Hoàn tất huấn luyện!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Script huấn luyện YOLO PPE")
    parser.add_argument(
        "--config",
        default="training/configs/yolov8n_ppe.yaml",
        help="Đường dẫn tới file cấu hình YAML thí nghiệm",
    )
    args = parser.parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
