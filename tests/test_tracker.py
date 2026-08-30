"""Kiểm thử tự động cho module ppe_detection.tracker."""

import pytest

from ppe_detection.models import PPEStatus, PersonDetection
from ppe_detection.tracker import IoUTracker, iou


def test_iou_identical_boxes():
    """Kiểm tra IoU của 2 bounding box trùng nhau bằng 1.0."""
    box = [0.0, 0.0, 100.0, 100.0]
    assert pytest.approx(iou(box, box), abs=1e-4) == 1.0


def test_iou_disjoint_boxes():
    """Kiểm tra IoU của 2 bounding box không giao nhau bằng 0.0."""
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    assert iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    """Kiểm tra tính toán IoU khi 2 box giao nhau một phần."""
    box1 = [0.0, 0.0, 10.0, 10.0]  # diện tích 100
    box2 = [5.0, 0.0, 15.0, 10.0]  # diện tích 100, giao 50, hợp 150 -> IoU = 50/150 = 0.3333
    assert round(iou(box1, box2), 3) == 0.333


def test_tracker_assignment_and_new_ids():
    """Kiểm tra tracker gán ID mới và duy trì ID khi di chuyển ít."""
    tracker = IoUTracker(threshold=0.3, max_disappeared=5)

    det1 = PersonDetection(
        box=[10.0, 10.0, 50.0, 100.0],
        confidence=0.9,
        ppe=PPEStatus(),
    )
    tracks_f1 = tracker.update([det1])
    assert len(tracks_f1) == 1
    assigned_id = tracks_f1[0].track_id

    # Khung hình 2: Bounding box dịch chuyển nhẹ
    det2 = PersonDetection(
        box=[12.0, 11.0, 52.0, 101.0],
        confidence=0.91,
        ppe=PPEStatus(),
    )
    tracks_f2 = tracker.update([det2])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == assigned_id  # Phải giữ nguyên ID cũ


def test_tracker_max_disappeared_cleanup():
    """Kiểm tra xóa vết theo dõi khi đối tượng biến mất quá max_disappeared."""
    tracker = IoUTracker(threshold=0.3, max_disappeared=2)

    det = PersonDetection(box=[0.0, 0.0, 10.0, 10.0], confidence=0.8, ppe=PPEStatus())
    tracker.update([det])
    assert len(tracker.active_tracks()) == 1

    # Cập nhật không có detection nào trong 3 frame tiếp theo
    tracker.update([])
    tracker.update([])
    tracker.update([])

    assert len(tracker.active_tracks()) == 0
