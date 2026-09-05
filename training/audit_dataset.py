"""Script kiểm toán dataset tự động (Dataset Audit & Anti-Leakage Validator).

Kiểm tra:
1. Tính toàn vẹn của chiến lược phân chia theo nhóm (Group-Aware Splitting): Không có session nào xuất hiện ở nhiều split.
2. Kiểm tra Test B: Tập Test phải chứa camera độc lập chưa xuất hiện ở tập Train (unseen camera validation).
3. Kiểm tra tính duy nhất của mã băm SHA-256 (không có ảnh trùng lặp).
4. Thống kê phân phối nhãn chi tiết (helmet, no-helmet, vest, no-vest).
5. Xuất báo cáo chuẩn `training/dataset_report.json` làm nguồn chân lý duy nhất (Single Source of Truth).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("audit_dataset")


def audit_dataset(manifest_path: Path, output_report_path: Path | None = None) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_samples = len(rows)
    LOGGER.info("Auditing manifest: %s (%d total samples)", manifest_path, total_samples)

    # 1. Group check: recording_session per split
    session_splits: dict[str, set[str]] = defaultdict(set)
    camera_splits: dict[str, set[str]] = defaultdict(set)
    sha256_map: dict[str, str] = {}
    phash_map: dict[str, str] = {}
    duplicate_sha256: list[str] = []

    split_counts: Counter[str] = Counter()
    class_distribution: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
        "total": Counter(),
    }

    for row in rows:
        sample_id = row["sample_id"]
        sess = row["recording_session"]
        cam = row["camera_id"]
        split = row["split"].lower()
        sha = row["sha256"]
        phash = row.get("phash", "")
        labels = [lbl.strip() for lbl in row["class_labels"].split(";") if lbl.strip()]

        split_counts[split] += 1
        session_splits[sess].add(split)
        camera_splits[cam].add(split)

        if sha in sha256_map:
            duplicate_sha256.append(f"{sample_id} duplicates {sha256_map[sha]}")
        else:
            sha256_map[sha] = sample_id

        for lbl in labels:
            class_distribution[split][lbl] += 1
            class_distribution["total"][lbl] += 1

    # Kiểm tra rò rỉ session
    leaked_sessions = {s: sp for s, sp in session_splits.items() if len(sp) > 1}

    # Kiểm tra unseen camera cho Test
    train_cams = {cam for cam, sp in camera_splits.items() if "train" in sp}
    test_cams = {cam for cam, sp in camera_splits.items() if "test" in sp}
    unseen_test_cameras = sorted(test_cams - train_cams)

    is_leak_free = len(leaked_sessions) == 0
    has_unseen_cameras = len(unseen_test_cameras) >= 2

    report = {
        "manifest_path": str(manifest_path),
        "total_samples": total_samples,
        "split_summary": dict(split_counts),
        "class_distribution": {k: dict(v) for k, v in class_distribution.items()},
        "group_anti_leakage": {
            "is_leak_free": is_leak_free,
            "total_recording_sessions": len(session_splits),
            "leaked_sessions_count": len(leaked_sessions),
            "leaked_sessions": leaked_sessions,
        },
        "domain_shift_protocol": {
            "has_unseen_test_cameras": has_unseen_cameras,
            "train_cameras": sorted(train_cams),
            "test_cameras": sorted(test_cams),
            "unseen_test_cameras": unseen_test_cameras,
        },
        "exact_duplicates_count": len(duplicate_sha256),
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with output_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved dataset audit report to: %s", output_report_path)

    LOGGER.info("Audit Summary:")
    LOGGER.info("  - Total Samples: %d", total_samples)
    LOGGER.info("  - Leak Free: %s", is_leak_free)
    LOGGER.info("  - Unseen Test Cameras: %s", unseen_test_cameras)
    LOGGER.info("  - Classes: %s", dict(class_distribution["total"]))

    if not is_leak_free:
        LOGGER.error("CRITICAL: Data leakage detected in sessions: %s", leaked_sessions)
        sys.exit(1)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset split manifest for leakage and invariants")
    parser.add_argument(
        "--manifest",
        default="training/split_manifest.csv",
        help="Path to split_manifest.csv",
    )
    parser.add_argument(
        "--output",
        default="training/dataset_report.json",
        help="Path to output dataset_report.json",
    )
    args = parser.parse_args()
    audit_dataset(Path(args.manifest), Path(args.output))


if __name__ == "__main__":
    main()
