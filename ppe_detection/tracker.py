"""Module theo dõi đối tượng (Object Tracking) bằng thuật toán IoU Tracker.

Giúp duy trì ID định danh cố định cho từng cá nhân qua các khung hình liên tiếp
bằng cách tính toán chỉ số Intersection over Union (IoU) giữa bounding box mới
và các vết theo dõi (tracks) hiện có.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import PPEStatus, PersonDetection


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Tính tỉ lệ phần giao trên phần hợp (Intersection over Union) của 2 bounding box.

    Args:
        box_a: Tọa độ [x1, y1, x2, y2] của khung A.
        box_b: Tọa độ [x1, y1, x2, y2] của khung B.

    Returns:
        Giá trị IoU nằm trong khoảng [0.0, 1.0].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class Track:
    """Đại diện cho trạng thái của một cá nhân đang được hệ thống theo dõi.

    Attributes:
        track_id: Mã ID định danh duy nhất của người.
        box: Tọa độ bounding box hiện tại [x1, y1, x2, y2].
        ppe: Trạng thái kiểm tra PPE hiện tại.
        confidence: Độ tin cậy phát hiện.
        disappeared: Số khung hình liên tiếp không thấy đối tượng.
        updated: Cờ xác định vết này có được cập nhật ở khung hiện tại không.
    """

    track_id: int
    box: list[float]
    ppe: PPEStatus
    confidence: float
    disappeared: int = 0
    updated: bool = True


class IoUTracker:
    """Bộ theo dõi đối tượng dựa trên thuật toán ghép cặp Greedy IoU."""

    def __init__(self, threshold: float = 0.3, max_disappeared: int = 30) -> None:
        """Khởi tạo IoU Tracker.

        Args:
            threshold: Ngưỡng IoU tối thiểu để coi là cùng một đối tượng.
            max_disappeared: Số lần bỏ lỡ tối đa trước khi xóa ID khỏi bộ nhớ.
        """
        self.threshold = threshold
        self.max_disappeared = max_disappeared
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[PersonDetection]) -> list[Track]:
        """Cập nhật vị trí vết theo dõi dựa trên danh sách detection ở khung hình mới.

        Args:
            detections: Danh sách phát hiện người mới nhất từ detector.

        Returns:
            Danh sách các `Track` đang hoạt động (active).
        """
        if not detections:
            # Nếu không tìm thấy ai, tăng số khung hình bị mất dấu cho mọi track
            for track_id in list(self.tracks):
                self.tracks[track_id].disappeared += 1
                self.tracks[track_id].updated = False
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    del self.tracks[track_id]
            return self.active_tracks()

        track_ids = list(self.tracks)
        if track_ids:
            # Tạo ma trận tỉ lệ IoU giữa các track hiện có và detections mới
            scores = np.array(
                [
                    [iou(self.tracks[tid].box, det.box) for det in detections]
                    for tid in track_ids
                ],
                dtype=float,
            )
        else:
            scores = np.empty((0, len(detections)))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        # Ghép cặp tham lam (Greedy Match) theo giá trị IoU cao nhất
        while scores.size > 0 and scores.max(initial=-1.0) >= self.threshold:
            row, col = np.unravel_index(scores.argmax(), scores.shape)
            tid = track_ids[row]
            det = detections[col]

            # Cập nhật track với thông tin bounding box và PPE mới
            self.tracks[tid] = Track(
                track_id=tid,
                box=det.box,
                ppe=det.ppe,
                confidence=det.confidence,
                disappeared=0,
                updated=True,
            )
            matched_tracks.add(tid)
            matched_detections.add(col)

            # Loại bỏ hàng và cột đã ghép cặp khỏi ma trận
            scores[row, :] = -1.0
            scores[:, col] = -1.0

        # Khởi tạo ID mới cho các detection chưa được ghép cặp
        for col, det in enumerate(detections):
            if col not in matched_detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(
                    track_id=tid,
                    box=det.box,
                    ppe=det.ppe,
                    confidence=det.confidence,
                    disappeared=0,
                    updated=True,
                )
                matched_tracks.add(tid)

        # Xử lý các track không ghép được với detection nào
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.disappeared += 1
                track.updated = False
                if track.disappeared > self.max_disappeared:
                    del self.tracks[tid]

        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        """Trả về danh sách các track còn hiệu lực sắp xếp theo ID tăng dần."""
        return [self.tracks[tid] for tid in sorted(self.tracks)]
