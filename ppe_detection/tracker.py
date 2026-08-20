from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Tính tỉ lệ phần giao trên phần hợp của hai bounding box."""
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """Trạng thái hiện tại của một đối tượng đang được theo dõi."""

    track_id: int
    box: list[float]
    ppe: dict
    confidence: float
    disappeared: int = 0


class IoUTracker:
    """Ghép detection giữa các lần cập nhật dựa trên IoU lớn nhất."""

    def __init__(self, threshold: float = 0.3, max_disappeared: int = 30):
        self.threshold = threshold
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[dict]) -> list[Track]:
        """Cập nhật và trả về toàn bộ track chưa vượt quá thời gian chờ."""
        if not detections:
            for track_id in list(self.tracks):
                self.tracks[track_id].disappeared += 1
                if self.tracks[track_id].disappeared > self.max_disappeared:
                    del self.tracks[track_id]
            return self.active_tracks()

        ids = list(self.tracks)
        scores = np.array(
            [[iou(self.tracks[track_id].box, det["box"]) for det in detections] for track_id in ids],
            dtype=float,
        ) if ids else np.empty((0, len(detections)))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        while scores.size and scores.max(initial=-1) >= self.threshold:
            row, col = np.unravel_index(scores.argmax(), scores.shape)
            track_id = ids[row]
            det = detections[col]
            self.tracks[track_id] = Track(track_id, det["box"], det["ppe"], det["confidence"])
            matched_tracks.add(track_id)
            matched_detections.add(col)
            scores[row, :] = -1
            scores[:, col] = -1

        for index, det in enumerate(detections):
            if index not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = Track(track_id, det["box"], det["ppe"], det["confidence"])
                matched_tracks.add(track_id)

        for track_id in ids:
            if track_id not in matched_tracks:
                track = self.tracks[track_id]
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    del self.tracks[track_id]

        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        """Trả về các track còn hiệu lực theo thứ tự ID."""
        return [self.tracks[track_id] for track_id in sorted(self.tracks)]
