"""Phát hiện người và vi phạm trang bị bảo hộ bằng hai model YOLO."""

from typing import TYPE_CHECKING

from .config import DetectionConfig

if TYPE_CHECKING:
    from .pipeline import PPEPipeline
    from .service import DetectionService

__all__ = ["DetectionConfig", "PPEPipeline", "DetectionService"]


def __getattr__(name: str):
    """Chỉ nạp pipeline/service khi được sử dụng để tránh import thư viện nặng sớm."""
    if name == "PPEPipeline":
        from .pipeline import PPEPipeline

        return PPEPipeline
    if name == "DetectionService":
        from .service import DetectionService

        return DetectionService
    raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")
