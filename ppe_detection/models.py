"""Module định nghĩa cấu trúc dữ liệu cho kết quả phát hiện PPE.

Cung cấp các dataclass đại diện cho từng đối tượng phát hiện:
- `PPEDetection`: Trang bị bảo hộ cá nhân (Mũ, Áo phản quang, v.v.).
- `PPEStatus`: Tổng hợp trạng thái tuân thủ bảo hộ của một cá nhân.
- `PersonDetection`: Đối tượng người cùng thông tin vùng ảnh và trạng thái PPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PPEDetection:
    """Kết quả nhận diện một trang bị bảo hộ cá nhân trong vùng ảnh ROI.

    Attributes:
        label: Tên nhãn lớp PPE (ví dụ: 'helmet', 'no-helmet', 'vest', 'no-vest').
        confidence: Độ tin cậy dự đoán từ mô hình (từ 0.0 đến 1.0).
    """

    label: str
    confidence: float


@dataclass(slots=True)
class PPEStatus:
    """Trạng thái tổng hợp trang bị bảo hộ của một người.

    Attributes:
        detections: Danh sách các trang bị bảo hộ được tìm thấy.
        helmet_violation: Cờ báo vi phạm không đội mũ bảo hộ.
        vest_violation: Cờ báo vi phạm không mặc áo phản quang.
    """

    detections: list[PPEDetection] = field(default_factory=list)
    helmet_violation: bool = False
    vest_violation: bool = False


@dataclass(slots=True)
class PersonDetection:
    """Thông tin vị trí và trạng thái PPE của một người trước khi đưa vào tracker.

    Attributes:
        box: Tọa độ bounding box [x1, y1, x2, y2] tính theo pixel.
        confidence: Độ tin cậy phát hiện người của mô hình YOLO.
        ppe: Trạng thái kiểm tra trang bị bảo hộ tương ứng.
    """

    box: list[float]
    confidence: float
    ppe: PPEStatus
