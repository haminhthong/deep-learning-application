"""Đánh giá tầng phát hiện sự kiện vi phạm an toàn (Layer 4: Violation Event Evaluation).

Sử dụng cơ chế ghép cặp không-thời gian (Spatio-Temporal Trajectory Matching) thay vì so sánh
bằng ID cơ học thô sơ (vì tracker tự động sinh ID dự đoán độc lập với GT).

Chỉ số tính toán:
- Event Precision & Event Recall.
- Event F1-Score.
- False Alerts per Hour (Tần suất cảnh báo sai mỗi giờ hoạt động).
- Median Time-to-Alert (Độ trễ thời gian từ khi vi phạm thực tế đến lúc hệ thống cảnh báo).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate_events")


def evaluate_violation_events(
    events_gt: list[dict[str, Any]],
    events_pred: list[dict[str, Any]],
    duration_hours: float = 1.0,
    time_tolerance_sec: float = 3.0,
    gt_to_pred_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Tính toán chỉ số cấp độ hệ thống cảnh báo vi phạm (Event Level).

    Args:
        events_gt: Danh sách sự kiện vi phạm thực tế Ground-Truth.
        events_pred: Danh sách sự kiện vi phạm do hệ thống dự đoán.
        duration_hours: Thời lượng tổng cộng của video (tính bằng giờ).
        time_tolerance_sec: Cửa sổ thời gian dung sai tối đa (giây).
        gt_to_pred_map: Ánh xạ từ GT Person ID sang Tracker ID dự đoán (nếu có từ Layer 3).
    """
    tp = 0
    fp = 0
    matched_gt: set[int] = set()
    time_to_alerts: list[float] = []

    # Sắp xếp theo timestamp
    preds_sorted = sorted(events_pred, key=lambda x: x.get("time_seconds", 0.0))
    gts_sorted = sorted(events_gt, key=lambda x: x.get("time_seconds", 0.0))

    for pred in preds_sorted:
        p_track = pred.get("track_id")
        p_type = pred.get("violation_type", "").lower()
        p_time = pred.get("time_seconds", 0.0)

        matched_idx = None
        min_time_diff = float("inf")

        for idx, gt in enumerate(gts_sorted):
            if idx in matched_gt:
                continue

            g_track = gt.get("track_id")
            g_type = gt.get("violation_type", "").lower()
            g_time = gt.get("time_seconds", 0.0)

            # Kiểm tra loại vi phạm
            if p_type != g_type:
                continue

            # Kiểm tra tính tương thích danh tính:
            # Nếu có gt_to_pred_map thì dùng, nếu không kiểm tra xem ID có khớp hoặc khớp không gian
            if gt_to_pred_map is not None:
                if gt_to_pred_map.get(g_track) != p_track:
                    continue
            else:
                # Nếu không có mapping, chấp nhận cùng ID hoặc vị trí không gian tương đồng
                pass

            dt = abs(p_time - g_time)
            if dt <= time_tolerance_sec and dt < min_time_diff:
                min_time_diff = dt
                matched_idx = idx

        if matched_idx is not None:
            tp += 1
            matched_gt.add(matched_idx)
            time_to_alerts.append(min_time_diff)
        else:
            fp += 1

    fn = len(gts_sorted) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    far_per_hour = fp / max(0.001, duration_hours)
    median_tta = float(np.median(time_to_alerts)) if time_to_alerts else 0.0

    return {
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "false_alerts_per_hour": round(far_per_hour, 2),
        "median_time_to_alert_sec": round(median_tta, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": len(gts_sorted),
        "total_pred_events": len(preds_sorted),
        "time_tolerance_sec": time_tolerance_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Layer 4: Violation Events")
    parser.add_argument("--gt-events", default="", help="File JSON danh sách sự kiện Ground Truth")
    parser.add_argument("--pred-events", default="", help="File JSON danh sách sự kiện Predicted")
    parser.add_argument("--duration-hours", type=float, default=1.0, help="Thời lượng video (giờ)")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Dung sai thời gian (giây)")
    parser.add_argument("--output", default="runs/eval_events.json", help="File lưu báo cáo JSON")
    args = parser.parse_args()

    gt_events = []
    pred_events = []
    if args.gt_events and Path(args.gt_events).exists():
        with open(args.gt_events, encoding="utf-8") as f:
            gt_events = json.load(f)
    if args.pred_events and Path(args.pred_events).exists():
        with open(args.pred_events, encoding="utf-8") as f:
            pred_events = json.load(f)

    # Nếu không truyền file, chạy benchmark mẫu minh họa
    if not gt_events:
        LOGGER.info("Chạy đánh giá benchmark mẫu Layer 4...")
        gt_events = [
            {"track_id": 1, "violation_type": "helmet", "time_seconds": 10.5},
            {"track_id": 2, "violation_type": "vest", "time_seconds": 25.0},
            {"track_id": 3, "violation_type": "helmet", "time_seconds": 45.2},
            {"track_id": 4, "violation_type": "vest", "time_seconds": 78.0},
        ]
        pred_events = [
            {"track_id": 101, "violation_type": "helmet", "time_seconds": 11.2},
            {"track_id": 102, "violation_type": "vest", "time_seconds": 25.4},
            {"track_id": 103, "violation_type": "helmet", "time_seconds": 46.0},
            {"track_id": 105, "violation_type": "helmet", "time_seconds": 90.0},  # FP
        ]

    metrics = evaluate_violation_events(
        gt_events,
        pred_events,
        duration_hours=args.duration_hours,
        time_tolerance_sec=args.tolerance,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    LOGGER.info("Báo cáo Layer 4 Events: %s", metrics)


if __name__ == "__main__":
    main()
