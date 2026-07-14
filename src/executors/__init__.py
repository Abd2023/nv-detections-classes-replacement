from __future__ import annotations

from .detections_classes_replacement_executor import (
    DetectionsClassesReplacementExecutor,
)


EXECUTORS = {
    "DetectionsClassesReplacementExecutor": DetectionsClassesReplacementExecutor,
}

__all__ = ["DetectionsClassesReplacementExecutor", "EXECUTORS"]
