from __future__ import annotations

import torch
from ultralytics import YOLO

from .config import DetectionConfig


def select_device() -> str:
    """Chọn thiết bị tăng tốc tốt nhất đang khả dụng."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DualModelDetector:
    """Phát hiện người trước, sau đó phát hiện PPE trên từng vùng người."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = select_device()
        self.person_model = YOLO(str(config.person_model_path))
        self.ppe_model = YOLO(str(config.ppe_model_path))
        self.person_model.to(self.device)
        self.ppe_model.to(self.device)

    def detect(self, frame) -> list[dict]:
        """Phát hiện người và trạng thái PPE trong một khung hình."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng")

        results = self.person_model.predict(
            frame, imgsz=self.config.image_size, conf=self.config.person_confidence,
            iou=self.config.nms_iou, classes=[0], device=self.device, verbose=False,
        )
        height, width = frame.shape[:2]
        detections: list[dict] = []
        if not results:
            return detections

        for box in results[0].boxes:
            raw_box = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = map(int, raw_box)
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(width, x2 + 10), min(height, y2 + 10)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            ppe = self._detect_ppe(roi)
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "ppe": ppe,
            })
        return detections

    def _detect_ppe(self, roi) -> dict:
        """Phát hiện trang bị bảo hộ trong vùng ảnh của một người."""
        labels: list[dict] = []
        results = self.ppe_model.predict(
            roi, imgsz=self.config.image_size, conf=self.config.ppe_confidence,
            iou=self.config.nms_iou, device=self.device, verbose=False,
        )
        if not results:
            return self._build_ppe_status(labels)

        result = results[0]
        for box in result.boxes:
            name = str(self.ppe_model.names[int(box.cls[0])])
            labels.append({"label": name, "confidence": float(box.conf[0])})

        return self._build_ppe_status(labels)

    @staticmethod
    def _build_ppe_status(labels: list[dict]) -> dict:
        """Chuẩn hóa tên class và xác định hai loại vi phạm."""
        normalized = {
            item["label"].strip().lower().replace("_", "-").replace(" ", "-")
            for item in labels
        }
        return {
            "detections": labels,
            "helmet_violation": "no-helmet" in normalized and "helmet" not in normalized,
            "vest_violation": "no-vest" in normalized and "vest" not in normalized,
        }
