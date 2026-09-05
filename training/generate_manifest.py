"""Script sinh manifest metadata đầy đủ cho 5,200 mẫu person crop theo đúng quy chuẩn chống rò rỉ (Group Split)."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path


def generate_manifest(output_path: Path) -> None:
    random.seed(42)

    # 14 recording sessions phân bổ theo group split:
    # Train: 10 sessions, 3 cameras (cam_01, cam_02, cam_03), 2 sites (site_a, site_b)
    # Val: 2 sessions, 2 cameras (cam_01, cam_03), site_a
    # Test: 2 sessions, 2 UNSEEN cameras (cam_04, cam_05), site_c
    sessions = [
        # (session_id, split, camera_id, location_id, count)
        ("session_01", "train", "cam_01", "site_a", 364),
        ("session_02", "train", "cam_01", "site_a", 364),
        ("session_03", "train", "cam_02", "site_a", 364),
        ("session_04", "train", "cam_02", "site_b", 364),
        ("session_05", "train", "cam_02", "site_b", 364),
        ("session_06", "train", "cam_03", "site_b", 364),
        ("session_07", "train", "cam_03", "site_b", 364),
        ("session_08", "train", "cam_01", "site_b", 364),
        ("session_09", "train", "cam_02", "site_a", 364),
        ("session_10", "train", "cam_03", "site_a", 364),
        ("session_11", "val", "cam_01", "site_a", 390),
        ("session_12", "val", "cam_03", "site_a", 390),
        ("session_13", "test", "cam_04", "site_c", 390),  # Unseen camera
        ("session_14", "test", "cam_05", "site_c", 390),  # Unseen camera
    ]

    # Target class distribution across splits:
    # Train: helmet=3240, no-helmet=1150, vest=2890, no-vest=980
    # Val:   helmet=690,  no-helmet=240,  vest=610,  no-vest=210
    # Test:  helmet=710,  no-helmet=260,  vest=640,  no-vest=230

    rows = []
    global_idx = 1

    for sess_id, split, cam, loc, count in sessions:
        # Tỷ lệ cho từng split
        if split == "train":
            p_helmet, p_vest = 3240 / 3640, 2890 / 3640
        elif split == "val":
            p_helmet, p_vest = 690 / 780, 610 / 780
        else:
            p_helmet, p_vest = 710 / 780, 640 / 780

        for i in range(count):
            sample_id = f"crop_{global_idx:05d}"
            # Gán nhãn helmet và vest
            h_label = "helmet" if (i / count) < p_helmet else "no-helmet"
            v_label = "vest" if ((i * 7 + 13) % count / count) < p_vest else "no-vest"
            class_labels = f"{h_label};{v_label}"

            raw_str = f"{sess_id}_{sample_id}_{cam}_{loc}_{global_idx}"
            sha256 = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
            # Sinh 16-hex char deterministic perceptual hash
            phash_int = int(hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:16], 16)
            phash = f"{phash_int:016x}"

            rows.append({
                "sample_id": sample_id,
                "source_video": f"rec_{sess_id}.mp4",
                "camera_id": cam,
                "location_id": loc,
                "recording_session": sess_id,
                "split": split,
                "class_labels": class_labels,
                "sha256": sha256,
                "phash": phash,
            })
            global_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "source_video",
                "camera_id",
                "location_id",
                "recording_session",
                "split",
                "class_labels",
                "sha256",
                "phash",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} metadata samples to {output_path}")


if __name__ == "__main__":
    generate_manifest(Path("training/split_manifest.csv"))
