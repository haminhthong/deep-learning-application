"""Tests kiểm tra tính toàn vẹn của Training Contract và Split Manifest (Anti-Leakage)."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml
from training.train import build_train_kwargs


def test_training_config_fields_are_forwarded() -> None:
    """Đảm bảo các trường siêu tham số trong config được train.py trích xuất và chuyển tiếp đầy đủ."""
    sample_cfg = {
        "experiment_name": "test_exp",
        "data_config": "training/data.yaml",
        "epochs": 50,
        "image_size": 640,
        "batch_size": 8,
        "seed": 123,
        "optimizer": "AdamW",
        "lr0": 0.005,
        "lrf": 0.001,
        "momentum": 0.9,
        "augmentations": {
            "hsv_h": 0.02,
            "mosaic": 0.5,
        },
    }

    train_kwargs = build_train_kwargs(sample_cfg)

    assert train_kwargs["optimizer"] == "AdamW"
    assert train_kwargs["lr0"] == 0.005
    assert train_kwargs["lrf"] == 0.001
    assert train_kwargs["epochs"] == 50
    assert train_kwargs["hsv_h"] == 0.02
    assert train_kwargs["mosaic"] == 0.5


def test_split_manifest_anti_leakage_invariants() -> None:
    """Đảm bảo không có rò rỉ phiên ghi hình (session) giữa các split và tập Test có unseen camera."""
    manifest_file = Path("training/split_manifest.csv")
    assert manifest_file.exists(), "split_manifest.csv phải tồn tại"

    sessions_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    cameras_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    with manifest_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["split"]
            sessions_by_split[sp].add(row["recording_session"])
            cameras_by_split[sp].add(row["camera_id"])

    # 1. Không giao nhau về session (Strict Group Disjointness)
    assert len(sessions_by_split["train"] & sessions_by_split["val"]) == 0
    assert len(sessions_by_split["train"] & sessions_by_split["test"]) == 0
    assert len(sessions_by_split["val"] & sessions_by_split["test"]) == 0

    # 2. Tập Test phải chứa ít nhất 2 camera chưa từng xuất hiện ở tập Train (Domain Shift Protocol)
    unseen_cameras = cameras_by_split["test"] - cameras_by_split["train"]
    assert len(unseen_cameras) >= 2, f"Tập Test phải có ít nhất 2 unseen cameras, thực tế: {unseen_cameras}"
