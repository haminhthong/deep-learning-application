from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Các tham số dùng chung cho toàn bộ quá trình phát hiện."""

    person_model_path: Path
    ppe_model_path: Path
    image_size: int = 640
    detection_interval: int = 4
    person_confidence: float = 0.3
    ppe_confidence: float = 0.3
    nms_iou: float = 0.5
    tracker_iou: float = 0.3
    max_disappeared: int = 30
    enable_beep: bool = True

    def validate(self) -> None:
        """Kiểm tra đường dẫn model và miền giá trị của các tham số."""
        for model_path in (self.person_model_path, self.ppe_model_path):
            if not model_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy model: {model_path.resolve()}")
        if self.image_size <= 0 or self.detection_interval <= 0:
            raise ValueError("image_size và detection_interval phải lớn hơn 0")
        for name, value in (
            ("person_confidence", self.person_confidence),
            ("ppe_confidence", self.ppe_confidence),
            ("nms_iou", self.nms_iou),
            ("tracker_iou", self.tracker_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong khoảng từ 0 đến 1")
        if self.max_disappeared < 0:
            raise ValueError("max_disappeared không được là số âm")
