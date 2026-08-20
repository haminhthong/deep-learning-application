"""Phát hiện người và vi phạm trang bị bảo hộ bằng hai model YOLO."""

from typing import TYPE_CHECKING

from .config import DetectionConfig

if TYPE_CHECKING:
    from .pipeline import PPEPipeline

__all__ = ["DetectionConfig", "PPEPipeline"]


def __getattr__(name: str):
    """Chỉ nạp pipeline khi được sử dụng để tránh import thư viện nặng sớm."""
    if name == "PPEPipeline":
        from .pipeline import PPEPipeline

        return PPEPipeline
    raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")
