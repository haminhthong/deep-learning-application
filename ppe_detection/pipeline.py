from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from .config import DetectionConfig
from .detector import DualModelDetector
from .tracker import IoUTracker
from .visualization import draw_tracks


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def alert() -> None:
    """Phát âm báo; lỗi thiết bị âm thanh không làm dừng chương trình."""
    try:
        if sys.platform == "win32":
            import winsound

            winsound.Beep(1000, 300)
        else:
            print("\a", end="", flush=True)
    except (OSError, RuntimeError):
        print("\a", end="", flush=True)


class PPEPipeline:
    """Điều phối phát hiện, theo dõi, thống kê và hiển thị kết quả."""

    def __init__(self, config: DetectionConfig):
        config.validate()
        self.config = config
        self.detector = DualModelDetector(config)
        self.tracker = IoUTracker(config.tracker_iou, config.max_disappeared)
        self.seen_violations: set[tuple[int, str]] = set()
        self.counts = {"total": 0, "helmet": 0, "vest": 0}

    def _count(self, tracks) -> None:
        """Mỗi loại vi phạm chỉ được tính một lần cho từng track."""
        should_alert = False
        for track in tracks:
            for kind in ("helmet", "vest"):
                key = (track.track_id, kind)
                if track.ppe.get(f"{kind}_violation") and key not in self.seen_violations:
                    self.seen_violations.add(key)
                    self.counts[kind] += 1
                    self.counts["total"] += 1
                    should_alert = True
        if should_alert and self.config.enable_beep:
            alert()

    def run(self, source: int | str) -> None:
        """Chọn chế độ xử lý ảnh đơn hoặc luồng video."""
        if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
            self._run_image(source)
        else:
            self._run_stream(source)

    def _run_image(self, source: str) -> None:
        frame = cv2.imread(source)
        if frame is None:
            raise ValueError(f"Không thể đọc ảnh: {source}")
        tracks = self.tracker.update(self.detector.detect(frame))
        self._count(tracks)
        try:
            cv2.imshow("Phát hiện trang bị bảo hộ", draw_tracks(frame, tracks, self.counts, 0.0))
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

    def _run_stream(self, source: int | str) -> None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Không thể mở nguồn dữ liệu: {source}")
        frame_id, previous_time = 0, time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_id += 1
                if frame_id == 1 or frame_id % self.config.detection_interval == 0:
                    tracks = self.tracker.update(self.detector.detect(frame))
                    self._count(tracks)
                else:
                    tracks = self.tracker.active_tracks()
                now = time.perf_counter()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                cv2.imshow("Phát hiện trang bị bảo hộ", draw_tracks(frame, tracks, self.counts, fps))
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
