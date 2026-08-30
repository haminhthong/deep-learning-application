"""Module thực hiện suy luận (inference) hai giai đoạn bằng YOLO.

Giai đoạn 1: Mô hình YOLO phát hiện đối tượng người (Person) trong khung hình.
Giai đoạn 2: Trích xuất vùng ảnh ROI người và đưa vào mô hình YOLO thứ 2 để nhận diện PPE
(Mũ bảo hộ 'helmet'/'no-helmet', Áo phản quang 'vest'/'no-vest').
Đồng thời hỗ trợ lớp `MockDetector` cho chế độ chạy thử nghiệm không cần weight ngoài.
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import DetectionConfig
from .models import PPEDetection, PPEStatus, PersonDetection

LOGGER = logging.getLogger(__name__)


def select_device() -> str:
    """Tự động chọn phần cứng tăng tốc inference tốt nhất khả dụng.

    Returns:
        Một trong các chuỗi: 'cuda', 'mps', hoặc 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_center_in_roi(box: list[float], roi_polygon: list[tuple[int, int]]) -> bool:
    """Kiểm tra xem điểm tâm của bounding box có nằm trong vùng nguy hiểm ROI hay không.

    Args:
        box: Tọa độ bounding box [x1, y1, x2, y2].
        roi_polygon: Danh sách các đỉnh (x, y) của vùng polygon ROI.

    Returns:
        True nếu điểm tâm nằm trong hoặc trên biên polygon, ngược lại False.
    """
    if not roi_polygon or len(roi_polygon) < 3:
        return True

    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    pts = np.array(roi_polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(pts, (float(center_x), float(center_y)), False)
    return res >= 0


class DetectorProtocol(Protocol):
    """Protocol chuẩn cho các lớp detector trong ứng dụng."""

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Phát hiện danh sách đối tượng người và PPE trên khung hình."""
        ...


class DualModelDetector:
    """Bộ phát hiện hai giai đoạn (Two-Stage Detector) dựa trên Ultralytics YOLO."""

    def __init__(self, config: DetectionConfig) -> None:
        """Khởi tạo và nạp trọng số hai mô hình YOLO (Person và PPE).

        Args:
            config: Cấu hình phát hiện `DetectionConfig`.
        """
        self.config = config
        self.device = select_device()
        LOGGER.info("Khởi tạo DualModelDetector trên thiết bị: %s", self.device)
        self.person_model = YOLO(str(config.person_model_path))
        self.ppe_model = YOLO(str(config.ppe_model_path))
        self.person_model.to(self.device)
        self.ppe_model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Phát hiện người và kiểm tra trạng thái trang bị bảo hộ trên khung hình.

        Args:
            frame: Ảnh/Khung hình BGR từ OpenCV.

        Returns:
            Danh sách đối tượng `PersonDetection`.

        Raises:
            ValueError: Nếu khung hình đầu vào rỗng hoặc None.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        results = self.person_model.predict(
            frame,
            imgsz=self.config.image_size,
            conf=self.config.person_confidence,
            iou=self.config.nms_iou,
            classes=[0],  # Class ID 0 trong COCO dataset là 'person'
            device=self.device,
            verbose=False,
        )

        height, width = frame.shape[:2]
        detections: list[PersonDetection] = []
        if not results:
            return detections

        for box in results[0].boxes:
            raw_box = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = map(int, raw_box)
            # Thêm lề (padding) 10px để mở rộng vùng nhận diện PPE
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(width, x2 + 10), min(height, y2 + 10)

            # Lọc theo vùng giám sát ROI nếu được thiết lập
            if self.config.roi_polygon and not is_center_in_roi([x1, y1, x2, y2], self.config.roi_polygon):
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            ppe_status = self._detect_ppe(roi)
            detections.append(
                PersonDetection(
                    box=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(box.conf[0]),
                    ppe=ppe_status,
                )
            )

        return detections

    def _detect_ppe(self, roi: np.ndarray) -> PPEStatus:
        """Phát hiện trang bị bảo hộ trong vùng ảnh ROI cắt của người.

        Args:
            roi: Vùng ảnh cắt của một cá nhân.

        Returns:
            `PPEStatus` chứa thông tin trang bị và các cờ báo vi phạm.
        """
        labels: list[PPEDetection] = []
        results = self.ppe_model.predict(
            roi,
            imgsz=self.config.image_size,
            conf=self.config.ppe_confidence,
            iou=self.config.nms_iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return self._build_ppe_status(labels)

        result = results[0]
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(self.ppe_model.names[class_id])
            labels.append(PPEDetection(label=name, confidence=float(box.conf[0])))

        return self._build_ppe_status(labels)

    @staticmethod
    def _build_ppe_status(labels: list[PPEDetection]) -> PPEStatus:
        """Chuẩn hóa tên class nhận diện và phân tích logic vi phạm.

        Args:
            labels: Danh sách phát hiện PPE từ mô hình.

        Returns:
            Đối tượng `PPEStatus` đã tính toán cờ vi phạm mũ và áo.
        """
        normalized = {
            item.label.strip().lower().replace("_", "-").replace(" ", "-")
            for item in labels
        }

        # Logic vi phạm: Có nhãn khẳng định vi phạm (no-helmet/no-vest)
        # và không đồng thời có nhãn tuân thủ (helmet/vest)
        helmet_violated = "no-helmet" in normalized and "helmet" not in normalized
        vest_violated = "no-vest" in normalized and "vest" not in normalized

        return PPEStatus(
            detections=labels,
            helmet_violation=helmet_violated,
            vest_violation=vest_violated,
        )


class MockDetector:
    """Detector giả lập (Mock) cho chế độ `--demo` phục vụ trải nghiệm không cần weights ngoài."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.frame_counter = 0

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Tạo đối tượng mô phỏng với bounding box và vi phạm trên ảnh."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        self.frame_counter += 1
        height, width = frame.shape[:2]
        detections: list[PersonDetection] = []

        # Mô phỏng người 1 (An toàn - Đủ mũ & áo)
        w, h = int(width * 0.2), int(height * 0.5)
        x1_a = int(width * 0.15 + np.sin(self.frame_counter * 0.05) * 15)
        y1_a = int(height * 0.25)
        det_a = PersonDetection(
            box=[float(x1_a), float(y1_a), float(x1_a + w), float(y1_a + h)],
            confidence=0.92,
            ppe=PPEStatus(
                detections=[
                    PPEDetection(label="helmet", confidence=0.89),
                    PPEDetection(label="vest", confidence=0.86),
                ],
                helmet_violation=False,
                vest_violation=False,
            ),
        )
        detections.append(det_a)

        # Mô phỏng người 2 (Vi phạm mũ & áo)
        x1_b = int(width * 0.6 + np.cos(self.frame_counter * 0.05) * 15)
        y1_b = int(height * 0.2)
        det_b = PersonDetection(
            box=[float(x1_b), float(y1_b), float(x1_b + w), float(y1_b + h)],
            confidence=0.88,
            ppe=PPEStatus(
                detections=[
                    PPEDetection(label="no-helmet", confidence=0.85),
                    PPEDetection(label="no-vest", confidence=0.81),
                ],
                helmet_violation=True,
                vest_violation=True,
            ),
        )
        detections.append(det_b)

        return detections
