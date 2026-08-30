"""Kiểm thử tự động cho module ppe_detection.pipeline."""

from pathlib import Path

import cv2
import numpy as np

from ppe_detection.config import DetectionConfig
from ppe_detection.pipeline import PPEPipeline


def test_pipeline_demo_mode_image_processing(tmp_path: Path):
    """Kiểm tra chạy pipeline thành công ở demo_mode với 1 ảnh tổng hợp."""
    # Tạo 1 ảnh đen giả lập kích thước 640x480
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "synthetic_test.jpg"
    cv2.imwrite(str(img_path), test_img)

    config = DetectionConfig(
        demo_mode=True,
        show_window=False,
        save_output=True,
        save_snapshots=True,
        output_dir=tmp_path / "outputs",
        enable_beep=False,
    )

    pipeline = PPEPipeline(config)
    report = pipeline.run(str(img_path))

    assert report.total_frames == 1
    # Ở demo mode MockDetector tạo ra 2 người mô phỏng
    assert len(report.unique_track_ids) == 2

    # Kiểm tra file ảnh đầu ra đã được tạo
    out_img = tmp_path / "outputs" / "synthetic_test_detected.jpg"
    assert out_img.is_file()
