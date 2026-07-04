"""Object detector protocol aliases and baseline implementations."""

from __future__ import annotations

from collections.abc import Sequence

from supercharged_coral.common.events import Detection, Frame, Region
from supercharged_coral.interfaces import ObjectDetector


class NoOpObjectDetector:
    """Detector implementation that intentionally returns no detections."""

    model_id = "noop-detector"

    async def detect(self, frame: Frame, regions: Sequence[Region] | None = None) -> list[Detection]:
        return []


__all__ = ["NoOpObjectDetector", "ObjectDetector"]
