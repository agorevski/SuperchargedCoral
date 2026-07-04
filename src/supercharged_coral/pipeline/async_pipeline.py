"""Async orchestration for camera frames, motion, detection, tracking, and storage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from supercharged_coral.common.events import Detection, Frame, Region
from supercharged_coral.interfaces import CameraSource, FrameStore, MotionDetector, ObjectDetector, ObjectTracker

__all__ = ["AsyncFramePipeline", "PipelineStats", "RegionEchoDetector"]


@dataclass
class PipelineStats:
    frames_seen: int = 0
    motion_frames: int = 0
    detections: int = 0
    tracks: int = 0


class AsyncFramePipeline:
    """Coordinate camera frames through motion, object detection, tracking, and storage."""

    def __init__(
        self,
        *,
        cameras: list[CameraSource],
        motion_detectors: dict[str, MotionDetector],
        object_detector: ObjectDetector,
        database: FrameStore | None = None,
        tracker: ObjectTracker | None = None,
        max_frames_per_camera: int | None = None,
    ) -> None:
        self.cameras = cameras
        self.motion_detectors = motion_detectors
        self.object_detector = object_detector
        self.database = database
        self.tracker = tracker
        self.max_frames_per_camera = max_frames_per_camera
        self.stats = PipelineStats()

    async def process_frame(self, frame: Frame) -> list[Detection]:
        self.stats.frames_seen += 1
        motion_detector = self.motion_detectors[frame.camera_id]
        motion_result = motion_detector.detect(frame)
        if not motion_result.has_motion:
            if self.database is not None:
                self.database.record_frame(frame)
            return []

        self.stats.motion_frames += 1
        regions = [motion_region.bbox for motion_region in motion_result.regions]
        detections = await self.object_detector.detect(frame, regions)
        self.stats.detections += len(detections)

        frame_id = self.database.record_frame(frame) if self.database is not None else None
        if frame_id is not None:
            for detection in detections:
                self.database.record_detection(frame_id, detection)

        if self.tracker is not None:
            tracks = self.tracker.update(detections)
            self.stats.tracks += len(tracks)
            if frame_id is not None:
                for track in tracks:
                    self.database.record_track(frame_id, track)

        return detections

    async def _run_camera(self, camera: CameraSource) -> None:
        processed = 0
        async for frame in camera.frames():
            await self.process_frame(frame)
            processed += 1
            if self.max_frames_per_camera is not None and processed >= self.max_frames_per_camera:
                break

    async def run(self) -> PipelineStats:
        try:
            await asyncio.gather(*(self._run_camera(camera) for camera in self.cameras))
            return self.stats
        finally:
            await asyncio.gather(*(camera.close() for camera in self.cameras), return_exceptions=True)


class RegionEchoDetector:
    """Debug detector that converts motion regions into synthetic detections."""

    model_id = "region-echo"

    async def detect(self, frame: Frame, regions: list[Region] | None = None) -> list[Detection]:
        return [
            Detection(
                class_name="Unknown",
                confidence=0.1,
                bbox=region,
                model_id=self.model_id,
                metadata={"source": "motion_region"},
            )
            for region in regions or []
        ]
