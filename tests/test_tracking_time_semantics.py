"""Tests kiểm tra thuật toán ByteTrack và ngữ nghĩa thời gian (Detection Cycles & Motion Prediction)."""

from __future__ import annotations

from ppe_detection.models import PersonDetection
from ppe_detection.tracker import ByteTrack


def test_bytetrack_two_stage_association() -> None:
    """ByteTrack phục hồi được track người bị che khuất một phần nhờ nhóm Low-Confidence Detections."""
    tracker = ByteTrack(high_threshold=0.6, match_threshold=0.5, low_match_threshold=0.3)

    # Frame 1: Người 1 rõ nét (conf 0.90) -> Khởi tạo track ID 1
    det_clear = [PersonDetection(box=[100.0, 100.0, 150.0, 250.0], confidence=0.90)]
    tracks1 = tracker.update(det_clear)
    assert len(tracks1) == 1
    assert tracks1[0].track_id == 1

    # Frame 2: Người 1 bị xe cẩu che khuất 1 phần, confidence giảm xuống 0.45 (nhỏ hơn high_threshold 0.6)
    # Thuật toán IoU thông thường sẽ bỏ qua và sinh ID mới, nhưng ByteTrack ghép cặp ở Stage 2!
    det_occluded = [PersonDetection(box=[105.0, 102.0, 155.0, 252.0], confidence=0.45)]
    tracks2 = tracker.update(det_occluded)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == 1, "ByteTrack phải duy trì đúng ID 1 qua Low-Confidence Stage 2!"


def test_motion_prediction_prevents_box_freeze() -> None:
    """Khi không chạy detector, tracker predict() phải nội suy di chuyển theo vận tốc, không đứng yên."""
    tracker = ByteTrack()

    # Frame 1
    tracker.update([PersonDetection(box=[100.0, 100.0, 150.0, 200.0], confidence=0.9)])
    # Frame 2 (Người di chuyển sang phải 10px)
    tracker.update([PersonDetection(box=[110.0, 100.0, 160.0, 200.0], confidence=0.9)])

    # Frame 3: Non-detection frame -> gọi predict()
    pred_tracks = tracker.predict()
    assert len(pred_tracks) == 1
    # Bounding box phải dịch chuyển theo vận tốc (x1 > 110.0) chứ không bị đóng băng cứng ở 110.0
    assert pred_tracks[0].box[0] > 110.0


def test_max_missed_detections_semantics() -> None:
    """Xác nhận max_missed_detections tính theo chu kỳ detector (update cycles), không xóa track quá sớm."""
    tracker = ByteTrack(max_missed_detections=3)

    tracker.update([PersonDetection(box=[10.0, 10.0, 50.0, 80.0], confidence=0.9)])
    assert len(tracker.active_tracks()) == 1

    # 1st missed cycle
    tracker.update([])
    assert len(tracker.active_tracks()) == 1
    assert tracker.active_tracks()[0].disappeared == 1

    # 2nd missed cycle
    tracker.update([])
    assert len(tracker.active_tracks()) == 1

    # 3rd missed cycle
    tracker.update([])
    assert len(tracker.active_tracks()) == 1

    # 4th missed cycle (> 3) -> Xóa track
    tracker.update([])
    assert len(tracker.active_tracks()) == 0
