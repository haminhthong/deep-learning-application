"""Module quản lý cấu hình cho hệ thống phát hiện trang bị bảo hộ (PPE).

Cung cấp dataclass `DetectionConfig` để lưu trữ, kiểm tra tính hợp lệ
của các tham số đầu vào và điều chỉnh hành vi của toàn bộ pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DetectionConfig:
    """Lớp lưu trữ toàn bộ cấu hình tham số cho hệ thống phát hiện PPE.

    Attributes:
        person_model_path: Đường dẫn tới file trọng số YOLO phát hiện người.
        ppe_model_path: Đường dẫn tới file trọng số YOLO phát hiện PPE.
        image_size: Kích thước chiều của ảnh đầu vào cho mô hình YOLO (mặc định 640).
        detection_interval: Số khung hình bỏ qua giữa các lần inference (mặc định 4).
        person_confidence: Ngưỡng tin cậy phát hiện người (0.0 đến 1.0).
        ppe_confidence: Ngưỡng tin cậy phát hiện PPE (0.0 đến 1.0).
        nms_iou: Ngưỡng IoU cho Non-Maximum Suppression của YOLO.
        tracker_iou: Ngưỡng IoU tối thiểu để ghép cặp vết theo dõi (IoU Tracker).
        max_disappeared: Số khung hình tối đa giữ lại track khi đối tượng mất dấu.
        violation_confirmations: Số khung hình vi phạm liên tiếp để ghi nhận chính thức.
        enable_beep: Cho phép phát âm thanh cảnh báo khi có vi phạm mới.
        show_window: Hiển thị cửa sổ xem trực tiếp bằng OpenCV.
        save_output: Lưu video/ảnh kết quả và file báo cáo ra đĩa.
        save_snapshots: Lưu ảnh cắt (ROI snapshot) của đối tượng khi vi phạm.
        output_dir: Thư mục gốc lưu trữ kết quả đầu ra.
        snapshot_dir: Thư mục lưu trữ ảnh bằng chứng vi phạm.
        demo_mode: Bật chế độ chạy thử nghiệm (mô phỏng) không cần file trọng số ngoài.
        roi_polygon: Danh sách tọa độ đỉnh (x, y) tạo thành vùng nguy hiểm ROI.
    """

    person_model_path: Path | None = None
    ppe_model_path: Path | None = None
    image_size: int = 640
    detection_interval: int = 4
    person_confidence: float = 0.3
    ppe_confidence: float = 0.3
    nms_iou: float = 0.5
    tracker_iou: float = 0.3
    max_disappeared: int = 30
    violation_confirmations: int = 2
    enable_beep: bool = True
    show_window: bool = True
    save_output: bool = False
    save_snapshots: bool = True
    output_dir: Path = Path("outputs")
    snapshot_dir: Path = Path("outputs/snapshots")
    demo_mode: bool = False
    roi_polygon: list[tuple[int, int]] | None = None

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của đường dẫn và miền giá trị tham số.

        Raises:
            FileNotFoundError: Nếu không tìm thấy file model khi không ở demo_mode.
            ValueError: Nếu bất kỳ tham số số nào nằm ngoài khoảng cho phép.
        """
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

        for name, value in (
            ("person_confidence", self.person_confidence),
            ("ppe_confidence", self.ppe_confidence),
            ("nms_iou", self.nms_iou),
            ("tracker_iou", self.tracker_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong khoảng từ 0.0 đến 1.0.")

        if self.max_disappeared < 0:
            raise ValueError("max_disappeared không được là số âm.")
