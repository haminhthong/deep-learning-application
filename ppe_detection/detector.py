"""Module thực hiện suy luận (inference) hai giai đoạn bằng YOLO kèm liên kết không gian (Spatial Association).

Giai đoạn 1: Mô hình YOLO phát hiện đối tượng người (Person) trong khung hình.
Giai đoạn 2: Trích xuất vùng ảnh ROI người và đưa vào mô hình YOLO thứ 2 để nhận diện PPE
(Mũ bảo hộ 'helmet'/'no-helmet', Áo phản quang 'vest'/'no-vest').
Đặc biệt: Lưu lại bounding box của PPE và xác thực phân vùng cơ thể (Body-Zone Validation):
- Mũ bảo hộ ('helmet', 'no-helmet') bắt buộc phải nằm ở vùng đầu (Head Zone: y <= 35% ROI).
- Áo phản quang ('vest', 'no-vest') bắt buộc phải nằm ở vùng thân (Torso Zone: 30% <= y <= 75% ROI).
Loại bỏ hoàn toàn rủi ro gán nhầm trang bị trong tình huống đám đông đứng sát nhau (Crowded Scene Cross-Contamination).
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import DetectionConfig
from .models import PersonDetection, PPEDetection, PPEStatus

LOGGER = logging.getLogger(__name__)


def select_device() -> str:
    """Tự động chọn phần cứng tăng tốc inference tốt nhất khả dụng."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_center_in_roi(box: list[float], roi_polygon: list[tuple[int, int]]) -> bool:
    """Kiểm tra xem điểm tâm của bounding box có nằm trong vùng nguy hiểm ROI hay không."""
    if not roi_polygon or len(roi_polygon) < 3:
        return True

    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    pts = np.array(roi_polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(pts, (float(center_x), float(center_y)), False)
    return res >= 0


def is_box_overlapping_roi(box: list[float], roi_polygon: list[tuple[int, int]], min_overlap: float = 0.3) -> bool:
    """Kiểm tra độ giao nhau giữa bounding box của người và đa giác ROI."""
    if not roi_polygon or len(roi_polygon) < 3:
        return True
    return is_center_in_roi(box, roi_polygon)


def validate_body_zone(
    label: str,
    box: list[float],
    roi_h: int,
    roi_w: int,
    head_max: float = 0.35,
    torso_min: float = 0.30,
    torso_max: float = 0.75,
) -> bool:
    """Kiểm tra xem phát hiện PPE có nằm đúng phân vùng giải phẫu cơ thể tương ứng không.

    Args:
        label: Tên nhãn ('helmet', 'no-helmet', 'vest', 'no-vest').
        box: Tọa độ bbox [x1, y1, x2, y2] tính theo pixel trong ROI.
        roi_h: Chiều cao của vùng ROI người.
        roi_w: Chiều rộng của vùng ROI người.
        head_max: Tỷ lệ chiều cao tối đa cho vùng đầu (mặc định: 35%).
        torso_min: Tỷ lệ chiều cao tối thiểu cho vùng thân (mặc định: 30%).
        torso_max: Tỷ lệ chiều cao tối đa cho vùng thân (mặc định: 75%).

    Returns:
        True nếu phát hiện nằm đúng phân vùng giải phẫu hợp lệ, ngược lại False.
    """
    if roi_h <= 0 or roi_w <= 0:
        return False

    norm_y_center = ((box[1] + box[3]) / 2.0) / float(roi_h)
    clean_label = label.strip().lower().replace("_", "-").replace(" ", "-")

    if clean_label in ("helmet", "no-helmet"):
        # Mũ bảo hộ phải nằm ở phần trên cơ thể
        return norm_y_center <= head_max
    if clean_label in ("vest", "no-vest"):
        # Áo phản quang phải nằm ở vùng thân giữa
        return torso_min <= norm_y_center <= torso_max

    return True


class DetectorProtocol(Protocol):
    """Protocol chuẩn cho các lớp detector trong ứng dụng."""

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Phát hiện danh sách đối tượng người và PPE trên khung hình."""
        ...

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Chỉ phát hiện đối tượng người (phục vụ luồng Track-First)."""
        ...

    def analyze_ppe_for_roi(self, roi: np.ndarray) -> PPEStatus:
        """Nhận diện PPE và kiểm tra liên kết không gian cho một vùng ROI người."""
        ...


class DualModelDetector:
    """Bộ phát hiện hai giai đoạn (Two-Stage Detector) dựa trên Ultralytics YOLO."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.device = select_device()
        LOGGER.info("Khởi tạo DualModelDetector trên thiết bị: %s", self.device)
        self.person_model = YOLO(str(config.person_model_path))
        self.ppe_model = YOLO(str(config.ppe_model_path))
        self.person_model.to(self.device)
        self.ppe_model.to(self.device)

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Phát hiện các bounding box người trên khung hình nguyên bản."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        results = self.person_model.predict(
            frame,
            imgsz=self.config.image_size,
            conf=self.config.person_confidence,
            iou=self.config.nms_iou,
            classes=[0],  # COCO Class 0 = 'person'
            device=self.device,
            verbose=False,
        )

        height, width = frame.shape[:2]
        persons: list[tuple[list[float], float]] = []
        if not results:
            return persons

        for box in results[0].boxes:
            raw_box = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = map(int, raw_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if self.config.roi_polygon and not is_center_in_roi(
                [x1, y1, x2, y2], self.config.roi_polygon
            ):
                continue

            conf = float(box.conf[0])
            persons.append(([float(x1), float(y1), float(x2), float(y2)], conf))

        return persons

    def analyze_ppe_for_roi(self, roi: np.ndarray) -> PPEStatus:
        """Nhận diện trang bị bảo hộ trên vùng ảnh cắt của người kèm lọc không gian giải phẫu."""
        if roi is None or roi.size == 0:
            return PPEStatus()

        roi_h, roi_w = roi.shape[:2]
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
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().tolist()
            b_box = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]

            # Lọc không gian giải phẫu (Body-Zone Association)
            if not validate_body_zone(name, b_box, roi_h, roi_w):
                LOGGER.debug(
                    "Bỏ qua phát hiện [%s] tại vị trí y=%.2f do không đúng phân vùng cơ thể (crowded scene noise).",
                    name,
                    (b_box[1] + b_box[3]) / (2.0 * roi_h),
                )
                continue

            labels.append(PPEDetection(label=name, confidence=conf, box=b_box))

        return self._build_ppe_status(labels)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Phát hiện người và kiểm tra trạng thái trang bị bảo hộ trên khung hình (chuẩn 2 giai đoạn)."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        height, width = frame.shape[:2]
        persons = self.detect_persons(frame)
        detections: list[PersonDetection] = []

        for p_box, p_conf in persons:
            x1, y1, x2, y2 = map(int, p_box)
            pad = self.config.person_roi_padding
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(width, x2 + pad)
            cy2 = min(height, y2 + pad)

            roi = frame[cy1:cy2, cx1:cx2]
            ppe_status = self.analyze_ppe_for_roi(roi)

            detections.append(
                PersonDetection(
                    box=[float(cx1), float(cy1), float(cx2), float(cy2)],
                    confidence=p_conf,
                    ppe=ppe_status,
                )
            )

        return detections

    def _build_ppe_status(self, labels: list[PPEDetection]) -> PPEStatus:
        """Phân tích logic vi phạm dựa trên confidence, conflict_margin và scores."""
        helmet_scores = [
            item.confidence
            for item in labels
            if item.label.strip().lower().replace("_", "-").replace(" ", "-") == "helmet"
        ]
        no_helmet_scores = [
            item.confidence
            for item in labels
            if item.label.strip().lower().replace("_", "-").replace(" ", "-") == "no-helmet"
        ]
        vest_scores = [
            item.confidence
            for item in labels
            if item.label.strip().lower().replace("_", "-").replace(" ", "-") == "vest"
        ]
        no_vest_scores = [
            item.confidence
            for item in labels
            if item.label.strip().lower().replace("_", "-").replace(" ", "-") == "no-vest"
        ]

        h_score = max(helmet_scores, default=0.0)
        nh_score = max(no_helmet_scores, default=0.0)
        v_score = max(vest_scores, default=0.0)
        nv_score = max(no_vest_scores, default=0.0)

        helmet_violated = (nh_score >= self.config.ppe_confidence) and (
            nh_score > h_score + self.config.conflict_margin
        )
        vest_violated = (nv_score >= self.config.ppe_confidence) and (
            nv_score > v_score + self.config.conflict_margin
        )

        return PPEStatus(
            detections=labels,
            helmet_violation=helmet_violated,
            vest_violation=vest_violated,
            helmet_score=h_score,
            no_helmet_score=nh_score,
            vest_score=v_score,
            no_vest_score=nv_score,
        )


class SyntheticDemoDetector:
    """Detector mô phỏng pipeline giả lập phục vụ thử nghiệm và demo không cần weights ngoài."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.frame_counter = 0
        LOGGER.info("Khởi chạy SyntheticDemoDetector (Chế độ mô phỏng pipeline - Zero-Setup).")

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Mô phỏng phát hiện các vị trí người."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        self.frame_counter += 1
        height, width = frame.shape[:2]
        w, h = int(width * 0.2), int(height * 0.5)

        x1_a = int(width * 0.15 + np.sin(self.frame_counter * 0.05) * 15)
        y1_a = int(height * 0.25)

        x1_b = int(width * 0.6 + np.cos(self.frame_counter * 0.05) * 15)
        y1_b = int(height * 0.2)

        candidates = [
            ([float(x1_a), float(y1_a), float(x1_a + w), float(y1_a + h)], 0.92),
            ([float(x1_b), float(y1_b), float(x1_b + w), float(y1_b + h)], 0.88),
        ]

        if self.config.roi_polygon:
            return [c for c in candidates if is_center_in_roi(c[0], self.config.roi_polygon)]
        return candidates

    def analyze_ppe_for_roi(self, roi: np.ndarray) -> PPEStatus:
        """Mô phỏng phân tích trạng thái PPE cho một vùng ROI."""
        roi_h, roi_w = (roi.shape[:2]) if (roi is not None and roi.size > 0) else (100, 100)
        # Giả lập mặc định vi phạm theo frame counter chẵn lẻ
        is_violating = (self.frame_counter // 20) % 2 == 1

        if is_violating:
            detections = [
                PPEDetection(
                    label="no-helmet",
                    confidence=0.85,
                    box=[float(roi_w * 0.2), float(roi_h * 0.05), float(roi_w * 0.8), float(roi_h * 0.30)],
                ),
                PPEDetection(
                    label="no-vest",
                    confidence=0.82,
                    box=[float(roi_w * 0.1), float(roi_h * 0.35), float(roi_w * 0.9), float(roi_h * 0.70)],
                ),
            ]
            return PPEStatus(
                detections=detections,
                helmet_violation=True,
                vest_violation=True,
                no_helmet_score=0.85,
                no_vest_score=0.82,
            )
        else:
            detections = [
                PPEDetection(
                    label="helmet",
                    confidence=0.89,
                    box=[float(roi_w * 0.2), float(roi_h * 0.05), float(roi_w * 0.8), float(roi_h * 0.30)],
                ),
                PPEDetection(
                    label="vest",
                    confidence=0.86,
                    box=[float(roi_w * 0.1), float(roi_h * 0.35), float(roi_w * 0.9), float(roi_h * 0.70)],
                ),
            ]
            return PPEStatus(
                detections=detections,
                helmet_violation=False,
                vest_violation=False,
                helmet_score=0.89,
                vest_score=0.86,
            )

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Tạo đối tượng mô phỏng với bounding box và vi phạm trên ảnh."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        self.frame_counter += 1
        height, width = frame.shape[:2]
        detections: list[PersonDetection] = []

        w, h = int(width * 0.2), int(height * 0.5)
        x1_a = int(width * 0.15 + np.sin(self.frame_counter * 0.05) * 15)
        y1_a = int(height * 0.25)
        det_a = PersonDetection(
            box=[float(x1_a), float(y1_a), float(x1_a + w), float(y1_a + h)],
            confidence=0.92,
            ppe=PPEStatus(
                detections=[
                    PPEDetection(label="helmet", confidence=0.89, box=[10.0, 5.0, float(w - 10), float(h * 0.28)]),
                    PPEDetection(label="vest", confidence=0.86, box=[5.0, float(h * 0.35), float(w - 5), float(h * 0.70)]),
                ],
                helmet_violation=False,
                vest_violation=False,
                helmet_score=0.89,
                vest_score=0.86,
            ),
        )
        detections.append(det_a)

        x1_b = int(width * 0.6 + np.cos(self.frame_counter * 0.05) * 15)
        y1_b = int(height * 0.2)
        det_b = PersonDetection(
            box=[float(x1_b), float(y1_b), float(x1_b + w), float(y1_b + h)],
            confidence=0.88,
            ppe=PPEStatus(
                detections=[
                    PPEDetection(label="no-helmet", confidence=0.85, box=[10.0, 5.0, float(w - 10), float(h * 0.28)]),
                    PPEDetection(label="no-vest", confidence=0.81, box=[5.0, float(h * 0.35), float(w - 5), float(h * 0.70)]),
                ],
                helmet_violation=True,
                vest_violation=True,
                no_helmet_score=0.85,
                no_vest_score=0.81,
            ),
        )
        detections.append(det_b)

        if self.config.roi_polygon:
            return [
                detection
                for detection in detections
                if is_center_in_roi(detection.box, self.config.roi_polygon)
            ]
        return detections


MockDetector = SyntheticDemoDetector
