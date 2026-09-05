"""Module quản lý cấu hình cho hệ thống phát hiện trang bị bảo hộ (PPE).

Cung cấp dataclass `DetectionConfig` để lưu trữ, kiểm tra tính hợp lệ
của các tham số đầu vào và điều chỉnh hành vi của toàn bộ pipeline theo các hợp đồng kiến trúc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectionConfig:
    """Lớp lưu trữ toàn bộ cấu hình tham số cho hệ thống phát hiện PPE.

    Attributes:
        person_model_path: Đường dẫn tới file trọng số YOLO phát hiện người.
        ppe_model_path: Đường dẫn tới file trọng số YOLO phát hiện PPE.
        image_size: Kích thước chiều của ảnh đầu vào cho mô hình YOLO (mặc định 640).
        detection_interval: Số khung hình bỏ qua giữa các lần inference detector (mặc định 4).
        person_confidence: Ngưỡng tin cậy phát hiện người (0.0 đến 1.0).
        ppe_confidence: Ngưỡng tin cậy phát hiện PPE (0.0 đến 1.0).
        nms_iou: Ngưỡng IoU cho Non-Maximum Suppression của YOLO.
        tracker_type: Thuật toán tracking ('bytetrack' hoặc 'iou').
        tracker_iou: Ngưỡng IoU tối thiểu để ghép cặp vết theo dõi.
        max_disappeared: Số chu kỳ detector tối đa giữ lại track khi đối tượng mất dấu (alias của max_missed_detections).
        max_missed_detections: Số chu kỳ detector bỏ lỡ tối đa trước khi xóa ID khỏi bộ nhớ.
        violation_confirmations: Số chu kỳ phát hiện vi phạm liên tiếp từ detector để xác nhận (mặc định 2).
        resolution_confirmations: Số chu kỳ phát hiện tuân thủ liên tiếp để xác nhận khắc phục (mặc định 3).
        enable_beep: Cho phép phát âm thanh cảnh báo khi có vi phạm mới.
        show_window: Hiển thị cửa sổ xem trực tiếp bằng OpenCV.
        save_output: Lưu video/ảnh kết quả và file báo cáo ra đĩa.
        save_snapshots: Lưu ảnh cắt (ROI snapshot) của đối tượng khi vi phạm.
        output_dir: Thư mục gốc lưu trữ kết quả đầu ra.
        demo_mode: Bật chế độ chạy thử nghiệm (mô phỏng) không cần file trọng số ngoài.
        roi_polygon: Danh sách tọa độ đỉnh (x, y) tạo thành vùng nguy hiểm ROI.
        person_roi_padding: Số pixel padding mở rộng khi cắt ROI người (mặc định 10).
        conflict_margin: Ngưỡng chênh lệch độ tin cậy để giải quyết nhãn xung đột (mặc định 0.1).
        enable_body_zone_filter: Kích hoạt bộ lọc liên kết không gian giải phẫu cơ thể.
        head_zone_max: Tỷ lệ chiều cao tối đa cho vùng đầu (mặc định 0.35).
        torso_zone_min: Tỷ lệ chiều cao tối thiểu cho vùng thân (mặc định 0.30).
        torso_zone_max: Tỷ lệ chiều cao tối đa cho vùng thân (mặc định 0.75).
    """

    person_model_path: Path | None = None
    ppe_model_path: Path | None = None
    image_size: int = 640
    detection_interval: int = 4
    person_confidence: float = 0.3
    ppe_confidence: float = 0.3
    nms_iou: float = 0.5
    tracker_type: str = "bytetrack"
    tracker_iou: float = 0.3
    max_disappeared: int = 30
    max_missed_detections: int = 30
    violation_confirmations: int = 2
    resolution_confirmations: int = 3
    enable_beep: bool = True
    show_window: bool = True
    save_output: bool = False
    save_snapshots: bool = True
    output_dir: Path = Path("outputs")
    demo_mode: bool = False
    roi_polygon: list[tuple[int, int]] | None = None
    person_roi_padding: int = 10
    conflict_margin: float = 0.1
    enable_body_zone_filter: bool = True
    head_zone_max: float = 0.35
    torso_zone_min: float = 0.30
    torso_zone_max: float = 0.75

    def __post_init__(self) -> None:
        # Đồng bộ hóa max_disappeared và max_missed_detections
        if self.max_missed_detections != 30 and self.max_disappeared == 30:
            self.max_disappeared = self.max_missed_detections
        elif self.max_disappeared != 30 and self.max_missed_detections == 30:
            self.max_missed_detections = self.max_disappeared

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của đường dẫn và miền giá trị tham số."""
        if not self.demo_mode:
            if self.person_model_path is None or not self.person_model_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy model phát hiện người: {self.person_model_path}"
                )
            if self.ppe_model_path is None or not self.ppe_model_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy model phát hiện PPE: {self.ppe_model_path}"
                )

        if self.image_size <= 0 or self.detection_interval <= 0:
            raise ValueError("image_size và detection_interval phải lớn hơn 0.")

        if self.violation_confirmations <= 0:
            raise ValueError("violation_confirmations phải lớn hơn 0.")

        if self.resolution_confirmations <= 0:
            raise ValueError("resolution_confirmations phải lớn hơn 0.")

        if self.person_roi_padding < 0:
            raise ValueError("person_roi_padding không được là số âm.")

        for name, value in (
            ("person_confidence", self.person_confidence),
            ("ppe_confidence", self.ppe_confidence),
            ("nms_iou", self.nms_iou),
            ("tracker_iou", self.tracker_iou),
            ("conflict_margin", self.conflict_margin),
            ("head_zone_max", self.head_zone_max),
            ("torso_zone_min", self.torso_zone_min),
            ("torso_zone_max", self.torso_zone_max),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong khoảng từ 0.0 đến 1.0.")

        if self.max_disappeared < 0 or self.max_missed_detections < 0:
            raise ValueError("max_disappeared và max_missed_detections không được là số âm.")

    @classmethod
    def load_from_policy(cls, policy_path: Path, **overrides: Any) -> DetectionConfig:
        """Nạp cấu hình từ file YAML runtime policy."""
        if not policy_path.exists():
            return cls(**overrides)

        with policy_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        inf_cfg = data.get("inference_contract", {})
        trk_cfg = data.get("tracking_policy", {})
        dec_cfg = data.get("decision_policy", {})

        kwargs: dict[str, Any] = {
            "person_confidence": inf_cfg.get("person_confidence", 0.3),
            "ppe_confidence": inf_cfg.get("ppe_confidence", 0.3),
            "nms_iou": inf_cfg.get("nms_iou", 0.5),
            "image_size": inf_cfg.get("image_size", 640),
            "person_roi_padding": inf_cfg.get("person_roi_padding_px", 10),
            "tracker_type": trk_cfg.get("tracker_type", "bytetrack"),
            "max_missed_detections": trk_cfg.get("max_missed_detections", 30),
            "conflict_margin": dec_cfg.get("conflict_margin", 0.1),
            "violation_confirmations": dec_cfg.get("violation_confirmations", 2),
            "resolution_confirmations": dec_cfg.get("resolution_confirmations", 3),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
