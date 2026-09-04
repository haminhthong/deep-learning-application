"""Module điều phối quy trình xử lý chính (PPE Pipeline).

Kết nối các thành phần: Detector, Tracker, Reporter, Visualizer và Sound Alert
để xử lý các nguồn đầu vào từ Ảnh tĩnh, Video file hoặc Luồng Webcam trực tiếp.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig
from .detector import DetectorProtocol, DualModelDetector, SyntheticDemoDetector
from .reporting import SessionReport
from .tracker import IoUTracker, Track
from .visualization import draw_tracks

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def sound_alert() -> None:
    """Phát âm thanh cảnh báo ngắn (Beep) mà không làm dừng chương trình."""
    try:
        if sys.platform == "win32":
            import winsound

            winsound.Beep(1000, 250)
        else:
            print("\a", end="", flush=True)
    except (OSError, RuntimeError):
        LOGGER.warning("Không thể phát âm thanh cảnh báo trên hệ thống này.")


class PPEPipeline:
    """Điều phối toàn bộ quy trình nhận diện, theo dõi, xác nhận vi phạm và lưu báo cáo."""

    def __init__(self, config: DetectionConfig) -> None:
        """Khởi tạo pipeline với cấu hình `DetectionConfig`.

        Args:
            config: Cấu hình hệ thống.
        """
        config.validate()
        self.config = config

        # Chọn detector dựa trên chế độ demo hoặc dual model
        if config.demo_mode:
            LOGGER.info("Đang chạy ở chế độ mô phỏng SyntheticDemoDetector.")
            self.detector: DetectorProtocol = SyntheticDemoDetector(config)
        else:
            self.detector = DualModelDetector(config)


        self.tracker = IoUTracker(config.tracker_iou, config.max_disappeared)
        self.confirmation_streaks: dict[tuple[int, str], int] = {}
        self.confirmed_violations: set[tuple[int, str]] = set()

    def run(self, source: int | str) -> SessionReport:
        """Thực thi luồng xử lý tương ứng với loại nguồn đầu vào.

        Args:
            source: Chỉ số camera (ví dụ 0) hoặc đường dẫn file ảnh/video.

        Returns:
            Đối tượng `SessionReport` chứa toàn bộ số liệu thống kê.
        """
        report = SessionReport(source)
        if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
            self._run_image(Path(source), report)
        else:
            self._run_stream(source, report)
        return report

    def _save_snapshot(self, frame: np.ndarray, track: Track, kind: str, frame_id: int) -> str:
        """Cắt và lưu ảnh snapshot của cá nhân vi phạm làm bằng chứng.

        Args:
            frame: Frame ảnh gốc.
            track: Vết theo dõi của cá nhân vi phạm.
            kind: Loại vi phạm ('helmet' hoặc 'vest').
            frame_id: Số khung hình phát hiện.

        Returns:
            Chuỗi đường dẫn tới file ảnh snapshot đã lưu.
        """
        if not (self.config.save_output and self.config.save_snapshots):
            return ""

        snapshot_dir = self.config.output_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, track.box)
        x1, y1 = max(0, x1 - 15), max(0, y1 - 15)
        x2, y2 = min(w, x2 + 15), min(h, y2 + 15)

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            roi = frame.copy()

        # Thêm thông tin văn bản vi phạm vào snapshot
        cv2.putText(
            roi,
            f"ID:{track.track_id} {kind.upper()} VIOLATION",
            (5, max(15, roi.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

        filename = f"violation_id{track.track_id}_{kind}_frame{frame_id}_{int(time.time())}.jpg"
        filepath = snapshot_dir / filename
        cv2.imwrite(str(filepath), roi)
        return str(filepath)

    def _confirm_violations(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        report: SessionReport,
        frame_id: int,
        source_fps: float,
    ) -> None:
        """Xác nhận vi phạm qua nhiều khung hình liên tiếp để giảm báo động giả.

        Args:
            frame: Frame ảnh hiện tại.
            tracks: Danh sách các track vừa được cập nhật.
            report: Báo cáo phiên làm việc.
            frame_id: Thứ tự frame.
            source_fps: FPS nguồn.
        """
        should_alert = False
        for track in tracks:
            report.unique_track_ids.add(track.track_id)
            if not track.updated:
                continue

            for kind, violated in (
                ("helmet", track.ppe.helmet_violation),
                ("vest", track.ppe.vest_violation),
            ):
                key = (track.track_id, kind)
                if violated:
                    self.confirmation_streaks[key] = self.confirmation_streaks.get(key, 0) + 1
                else:
                    self.confirmation_streaks[key] = 0

                is_confirmed = self.confirmation_streaks[key] >= self.config.violation_confirmations

                if is_confirmed and key not in self.confirmed_violations:
                    self.confirmed_violations.add(key)
                    snapshot_path = self._save_snapshot(frame, track, kind, frame_id)
                    report.add_event(
                        track_id=track.track_id,
                        kind=kind,
                        frame_id=frame_id,
                        fps=source_fps,
                        snapshot_path=snapshot_path,
                    )

                    LOGGER.warning(
                        "XÁC NHẬN VI PHẠM [%s] - Người ID: %d (Frame %d)",
                        kind.upper(),
                        track.track_id,
                        frame_id,
                    )
                    should_alert = True

        if should_alert and self.config.enable_beep:
            sound_alert()

    def _run_image(self, source: Path, report: SessionReport) -> None:
        """Xử lý nguồn dữ liệu dạng file ảnh tĩnh."""
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Không thể đọc file ảnh: {source}")

        tracks = self.tracker.update(self.detector.detect(frame))
        report.total_frames = 1

        for track in tracks:
            report.unique_track_ids.add(track.track_id)
            for kind, violated in (
                ("helmet", track.ppe.helmet_violation),
                ("vest", track.ppe.vest_violation),
            ):
                if violated:
                    snapshot_path = self._save_snapshot(frame, track, kind, 1)
                    report.add_event(
                        track_id=track.track_id,
                        kind=kind,
                        frame_id=1,
                        fps=0.0,
                        snapshot_path=snapshot_path,
                    )

        if report.events and self.config.enable_beep:
            sound_alert()

        annotated = draw_tracks(
            frame,
            tracks,
            report.counts,
            fps=0.0,
            frame_id=1,
            roi_polygon=self.config.roi_polygon,
        )

        if self.config.save_output:
            out_dir = self._prepare_output_dir()
            out_image_path = out_dir / f"{source.stem}_detected{source.suffix}"
            if not cv2.imwrite(str(out_image_path), annotated):
                raise OSError(f"Không thể ghi ảnh đầu ra: {out_image_path}")
            self._save_report(report, source.stem)
            LOGGER.info("Đã lưu ảnh kết quả: %s", out_image_path)

        if self.config.show_window:
            self._show_image(annotated)

    def _run_stream(self, source: int | str, report: SessionReport) -> None:
        """Xử lý nguồn dữ liệu luồng (Video file hoặc Webcam)."""
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Không thể kết nối nguồn dữ liệu: {source}")

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if source_fps > 0 else 30.0
        writer = None
        output_stem = self._source_stem(source)
        prev_time = time.perf_counter()
        smoothed_fps = 0.0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                report.total_frames += 1
                frame_id = report.total_frames

                # Chạy detection định kỳ theo interval để tối ưu FPS
                is_detection_frame = frame_id == 1 or frame_id % self.config.detection_interval == 0
                if is_detection_frame:
                    detections = self.detector.detect(frame)
                    tracks = self.tracker.update(detections)
                    self._confirm_violations(frame, tracks, report, frame_id, source_fps)
                else:
                    tracks = self.tracker.active_tracks()

                now = time.perf_counter()
                curr_fps = 1.0 / max(now - prev_time, 1e-6)
                smoothed_fps = (
                    curr_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * curr_fps
                )
                prev_time = now

                annotated = draw_tracks(
                    frame,
                    tracks,
                    report.counts,
                    smoothed_fps,
                    frame_id,
                    self.config.roi_polygon,
                )

                if self.config.save_output:
                    if writer is None:
                        writer = self._create_video_writer(annotated, source_fps, output_stem)
                    writer.write(annotated)

                if self.config.show_window:
                    cv2.imshow("Hệ thống Giám sát PPE - OpenCV", annotated)
                    # Nhấn phím 'ESC' để dừng
                    if cv2.waitKey(1) & 0xFF == 27:
                        LOGGER.info("Dừng chương trình theo lệnh người dùng (ESC).")
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self.config.show_window:
                cv2.destroyAllWindows()
            if self.config.save_output:
                self._save_report(report, output_stem)

        LOGGER.info(
            "Hoàn tất phiên giám sát: %d khung hình, %d người, %d sự kiện vi phạm.",
            report.total_frames,
            len(report.unique_track_ids),
            report.counts["total"],
        )

    def _prepare_output_dir(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return self.config.output_dir

    def _create_video_writer(self, frame: np.ndarray, fps: float, stem: str):
        out_path = self._prepare_output_dir() / f"{stem}_detected.mp4"
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Không thể tạo VideoWriter tại: {out_path}")
        LOGGER.info("Đang ghi video đầu ra: %s", out_path)
        return writer

    def _save_report(self, report: SessionReport, stem: str) -> None:
        json_p, csv_p = report.save(self._prepare_output_dir(), stem)
        LOGGER.info("Đã xuất báo cáo: JSON (%s), CSV (%s)", json_p, csv_p)

    @staticmethod
    def _source_stem(source: int | str) -> str:
        return f"camera_{source}" if isinstance(source, int) else Path(source).stem

    @staticmethod
    def _show_image(frame: np.ndarray) -> None:
        try:
            cv2.imshow("Kết quả Phát hiện Trang bị Bảo hộ", frame)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
