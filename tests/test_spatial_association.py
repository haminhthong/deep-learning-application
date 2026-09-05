"""Tests kiểm tra lưu trữ Bounding Box PPE và Liên kết không gian giải phẫu cơ thể (Spatial Body-Zone Association)."""

from __future__ import annotations

from ppe_detection.detector import validate_body_zone
from ppe_detection.models import PPEDetection, PPEStatus


def test_ppe_detection_retains_box() -> None:
    """Đảm bảo PPEDetection lưu trữ đầy đủ tọa độ bounding box."""
    det = PPEDetection(label="helmet", confidence=0.91, box=[10.0, 5.0, 50.0, 45.0])
    assert det.box == [10.0, 5.0, 50.0, 45.0]
    assert det.label == "helmet"
    assert det.confidence == 0.91


def test_helmet_must_match_head_zone() -> None:
    """Mũ bảo hộ ở đỉnh đầu phải được chấp thuận; mũ ở phần dưới cơ thể phải bị loại bỏ."""
    roi_h, roi_w = 200, 100

    # Box 1: Ở vùng đầu (y từ 10 đến 50 -> center y = 30 / 200 = 0.15 <= 0.35) -> HỢP LỆ
    head_box = [10.0, 10.0, 90.0, 50.0]
    assert validate_body_zone("helmet", head_box, roi_h, roi_w) is True
    assert validate_body_zone("no-helmet", head_box, roi_h, roi_w) is True

    # Box 2: Ở chân/đùi do người đứng cạnh bị dính crop (y từ 150 đến 190 -> center y = 170 / 200 = 0.85 > 0.35) -> TỪ CHỐI
    leg_box = [10.0, 150.0, 90.0, 190.0]
    assert validate_body_zone("helmet", leg_box, roi_h, roi_w) is False
    assert validate_body_zone("no-helmet", leg_box, roi_h, roi_w) is False


def test_vest_must_match_torso_zone() -> None:
    """Áo phản quang ở vùng thân phải được chấp thuận; áo ở đỉnh đầu hoặc sát chân phải bị loại bỏ."""
    roi_h, roi_w = 200, 100

    # Box 1: Ở thân (y từ 60 đến 140 -> center y = 100 / 200 = 0.50 trong [0.30, 0.75]) -> HỢP LỆ
    torso_box = [5.0, 60.0, 95.0, 140.0]
    assert validate_body_zone("vest", torso_box, roi_h, roi_w) is True
    assert validate_body_zone("no-vest", torso_box, roi_h, roi_w) is True

    # Box 2: Quá cao ở trên đỉnh đầu (y từ 0 đến 30 -> center y = 15 / 200 = 0.075 < 0.30) -> TỪ CHỐI
    top_box = [5.0, 0.0, 95.0, 30.0]
    assert validate_body_zone("vest", top_box, roi_h, roi_w) is False

    # Box 3: Quá thấp ở mắt cá chân (y từ 170 đến 195 -> center y = 182.5 / 200 = 0.91 > 0.75) -> TỪ CHỐI
    feet_box = [5.0, 170.0, 95.0, 195.0]
    assert validate_body_zone("vest", feet_box, roi_h, roi_w) is False
