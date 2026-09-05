"""Module điều phối quy trình xử lý chính (PPE Pipeline - Track-First Architecture).

Kết nối các thành phần theo chuẩn kiến trúc Online Canonical:
1. Frame → Person Detector
2. Multi-Object Tracking (ByteTrack / IoU Tracker kèm Motion Prediction)
3. Trích xuất Person Track ROIs
4. PPE Detector kèm Spatial Body-Zone Association
5. Temporal Violation FSM (Finite State Machine: COMPLIANT → VIOLATING → ALERTED → RESOLVED → VIOLATING)
6. Snapshot bằng chứng, cảnh báo âm thanh và xuất báo cáo đa định dạng JSON/CSV.
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
from .models import PersonDetection
from .reporting import SessionReport
from .tracker import ByteTrack, IoUTracker, Track, TrackerProtocol
from .violation_fsm import TemporalViolationFSM
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
    """Điều phối toàn bộ quy trình nhận diện, theo dõi Track-First và máy trạng thái thời gian."""

    def __init__(self, config: DetectionConfig) -> None:
        """Khởi tạo pipeline với cấu hình `DetectionConfig`."""
        config.validate()
        self.config = config

        if config.demo_mode:
            LOGGER.info("Đang chạy ở chế độ mô phỏng SyntheticDemoDetector.")
            self.detector: DetectorProtocol = SyntheticDemoDetector(config)
        else:
            self.detector = DualModelDetector(config)

        # Khởi tạo tracker theo cấu hình
        if config.tracker_type.lower() == "bytetrack":
            self.tracker: TrackerProtocol = ByteTrack(
                high_threshold=max(0.4, config.person_confidence),
                match_threshold=config.tracker_iou,
                max_missed_detections=config.max_missed_detections,
            )
        else:
            self.tracker = IoUTracker(
                threshold=config.tracker_iou,
                max_disappeared=config.max_missed_detections,
            )

        # Máy trạng thái hữu hạn kiểm soát vi phạm theo thời gian
        self.fsm = TemporalViolationFSM(
            confirm_observations=config.violation_confirmations,
            resolve_observations=config.resolution_confirmations,
        )

    def run(self, source: int | str) -> SessionReport:
        """Thực thi luồng xử lý tương ứng với loại nguồn đầu vào."""
        report = SessionReport(source)
        if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
            self._run_image(Path(source), report)
        else:
            self._run_stream(source, report)
        return report

    def _save_snapshot(self, frame: np.ndarray, track: Track, kind: str, frame_id: int) -> str:
        """Cắt và lưu ảnh snapshot của cá nhân vi phạm làm bằng chứng."""
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

    def _process_track_first_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        source_fps: float,
        report: SessionReport,
    ) -> list[Track]:
        """Quy trình Track-First: Phát hiện người → Cập nhật Track → Cắt ROI theo Track → Nhận diện PPE."""
        height, width = frame.shape[:2]
        person_detections_raw = self.detector.detect_persons(frame)

        person_dets: list[PersonDetection] = []
        for box, conf in person_detections_raw:
            person_dets.append(PersonDetection(box=box, confidence=conf))

        # Bước 1 & 2: Cập nhật vị trí vết theo dõi
        tracks = self.tracker.update(person_dets)

        # Bước 3 & 4: Trích xuất ROI người từ vết theo dõi và nhận diện PPE
        pad = self.config.person_roi_padding
        for track in tracks:
            report.unique_track_ids.add(track.track_id)
            if not track.updated:
                continue

            x1, y1, x2, y2 = map(int, track.box)
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(width, x2 + pad)
            cy2 = min(height, y2 + pad)

            roi = frame[cy1:cy2, cx1:cx2]
            ppe_status = self.detector.analyze_ppe_for_roi(roi)
            track.ppe = ppe_status

        # Bước 5: Đưa vào máy trạng thái Temporal Violation FSM
        timestamp_sec = round((frame_id - 1) / source_fps, 3) if source_fps > 0 else 0.0
        self._evaluate_fsm(frame, tracks, report, frame_id, timestamp_sec)

        return tracks

    def _evaluate_fsm(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        report: SessionReport,
        frame_id: int,
        timestamp_sec: float,
    ) -> None:
        """Đánh giá chuyển trạng thái máy hữu hạn FSM cho các track."""
        should_alert = False

        for track in tracks:
            if not track.updated:
                continue

            for kind, is_violated in (
                ("helmet", track.ppe.helmet_violation),
                ("vest", track.ppe.vest_violation),
            ):
                transition = self.fsm.update(
                    track_id=track.track_id,
                    violation_type=kind,
                    is_violated=is_violated,
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                )

                if transition.should_emit_alert:
                    snapshot_path = self._save_snapshot(frame, track, kind, frame_id)
                    report.add_event(
                        track_id=track.track_id,
                        kind=kind,
                        frame_id=frame_id,
                        fps=(frame_id / timestamp_sec) if timestamp_sec > 0 else 0.0,
                        snapshot_path=snapshot_path,
                    )
                    log_label = "TÁI PHẠM" if transition.is_recurrence else "XÁC NHẬN VI PHẠM"
                    LOGGER.warning(
                        "%s [%s] - Người ID: %d (Frame %d, %.2fs)",
                        log_label,
                        kind.upper(),
                        track.track_id,
                        frame_id,
                        timestamp_sec,
                    )
                    should_alert = True

        if should_alert and self.config.enable_beep:
            sound_alert()

    def _run_image(self, source: Path, report: SessionReport) -> None:
        """Xử lý nguồn dữ liệu dạng file ảnh tĩnh."""
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Không thể đọc file ảnh: {source}")

        report.total_frames = 1
        # Với ảnh đơn, phát hiện trực tiếp
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)

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
        """Xử lý nguồn dữ liệu luồng (Video file hoặc Webcam) với Track-First và Motion Prediction."""
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

                # Chu kỳ chạy detector
                is_detection_frame = frame_id == 1 or frame_id % self.config.detection_interval == 0
                if is_detection_frame:
                    tracks = self._process_track_first_frame(frame, frame_id, source_fps, report)
                else:
                    # Frame trung gian: Dự đoán vị trí chuyển động (Motion Prediction) chống freeze box
                    tracks = self.tracker.predict()

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
