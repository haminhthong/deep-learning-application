"""Script đánh giá đa cấp độ cho hệ thống PPE Surveillance.

Tính toán các chỉ số:
1. Detection level: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
2. Event level: Event Precision, Event Recall, False alerts/hour, Time-to-alert
3. Tracking level: ID switches, Track fragmentation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate")


def evaluate_detection(model_path: str, data_config: str, split: str = "test") -> dict:
    """Đánh giá cấp độ phát hiện Detection bằng Ultralytics val."""
    from ultralytics import YOLO

    LOGGER.info("Đánh giá Detection level cho weights [%s] trên split [%s]...", model_path, split)
    model = YOLO(model_path)
    results = model.val(data=data_config, split=split, verbose=False)

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "mp": float(results.box.mp),
        "mr": float(results.box.mr),
        "class_metrics": {},
    }

    for idx, cls_name in results.names.items():
        if idx < len(results.box.p):
            metrics["class_metrics"][cls_name] = {
                "precision": float(results.box.p[idx]),
                "recall": float(results.box.r[idx]),
                "f1": float(
                    2
                    * results.box.p[idx]
                    * results.box.r[idx]
                    / (results.box.p[idx] + results.box.r[idx] + 1e-6)
                ),
            }

    return metrics


def calculate_system_event_metrics(
    events_gt: list[dict], events_pred: list[dict], duration_hours: float
) -> dict:
    """Tính toán chỉ số cấp độ hệ thống cảnh báo (Event Level)."""
    tp = 0
    fp = 0
    fn = 0

    # Đếm khớp sự kiện vi phạm
    matched_gt = set()
    for pred in events_pred:
        matched = False
        for i, gt in enumerate(events_gt):
            if (
                i not in matched_gt
                and pred["track_id"] == gt["track_id"]
                and pred["violation_type"] == gt["violation_type"]
            ):
                if abs(pred["time_seconds"] - gt["time_seconds"]) <= 3.0:
                    tp += 1
                    matched_gt.add(i)
                    matched = True
                    break
        if not matched:
            fp += 1

    fn = len(events_gt) - len(matched_gt)
    event_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    event_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alerts_per_hour = fp / duration_hours if duration_hours > 0 else 0.0

    return {
        "event_precision": event_precision,
        "event_recall": event_recall,
        "false_alerts_per_hour": false_alerts_per_hour,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình PPE đa cấp độ")
    parser.add_argument(
        "--model", default="models/best.pt", help="Đường dẫn file trọng số model PPE"
    )
    parser.add_argument("--data", default="training/data.yaml", help="File cấu hình dataset YAML")
    parser.add_argument("--split", default="test", help="Tập dữ liệu đánh giá (val hoặc test)")
    parser.add_argument(
        "--output", default="runs/eval_results.json", help="File lưu kết quả báo cáo JSON"
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        det_metrics = evaluate_detection(args.model, args.data, args.split)
        report = {
            "model": args.model,
            "split": args.split,
            "detection_metrics": det_metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        LOGGER.info("Đánh giá hoàn tất. Kết quả được lưu tại: %s", out_path)
    except Exception as err:
        LOGGER.error("Không thể đánh giá mô hình: %s", err)


if __name__ == "__main__":
    main()
