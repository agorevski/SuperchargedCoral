from collections.abc import AsyncIterator, Sequence

import numpy as np
import pytest

from supercharged_coral.common.events import (
    Detection,
    Frame,
    MotionRegion,
    MotionResult,
    Region,
    Track,
)
from supercharged_coral.interfaces import (
    CameraSource,
    FrameStore,
    MotionDetector,
    ObjectDetector,
    ObjectTracker,
)
from supercharged_coral.pipeline.async_pipeline import AsyncFramePipeline


class CustomCamera:
    camera_id = "custom-camera"

    def __init__(self) -> None:
        self.closed = False

    async def frames(self) -> AsyncIterator[Frame]:
        yield Frame(camera_id=self.camera_id, image=np.zeros((32, 32, 3), dtype=np.uint8), sequence=1)

    async def snapshot(self) -> Frame:
        return Frame(camera_id=self.camera_id, image=np.zeros((32, 32, 3), dtype=np.uint8), sequence=0)

    async def close(self) -> None:
        self.closed = True


class CustomMotionDetector:
    algorithm_name = "custom-motion"

    def detect(self, frame: Frame) -> MotionResult:
        mask = np.ones(frame.image.shape[:2], dtype=np.uint8) * 255
        region = MotionRegion(bbox=Region(1, 1, 10, 10), score=1.0)
        return MotionResult(
            has_motion=True,
            regions=[region],
            mask=mask,
            heatmap=None,
            algorithm=self.algorithm_name,
            latency_ms=0.0,
        )


class CustomDetector:
    model_id = "custom-detector"

    async def detect(self, frame: Frame, regions: Sequence[Region] | None = None) -> list[Detection]:
        return [
            Detection(
                class_name="Person",
                confidence=0.99,
                bbox=region,
                model_id=self.model_id,
            )
            for region in regions or []
        ]


class CustomStore:
    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.detections: list[Detection] = []
        self.tracks: list[Track] = []

    def record_frame(
        self,
        frame: Frame,
        *,
        image_path: str | None = None,
        lighting_estimate: float | None = None,
    ) -> int:
        self.frames.append(frame)
        return len(self.frames)

    def record_detection(self, frame_id: int, detection: Detection) -> int:
        self.detections.append(detection)
        return len(self.detections)

    def record_track(self, frame_id: int, track: Track) -> int:
        self.tracks.append(track)
        return len(self.tracks)


class CustomTracker:
    def update(self, detections: Sequence[Detection]) -> list[Track]:
        return [
            Track(
                track_id=index,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                age=1,
                hits=1,
            )
            for index, detection in enumerate(detections, start=1)
        ]


@pytest.mark.asyncio
async def test_pipeline_accepts_structural_interface_implementations():
    camera = CustomCamera()
    detector = CustomDetector()
    motion = CustomMotionDetector()
    store = CustomStore()
    tracker = CustomTracker()

    assert isinstance(camera, CameraSource)
    assert isinstance(detector, ObjectDetector)
    assert isinstance(motion, MotionDetector)
    assert isinstance(store, FrameStore)
    assert isinstance(tracker, ObjectTracker)

    pipeline = AsyncFramePipeline(
        cameras=[camera],
        motion_detectors={camera.camera_id: motion},
        object_detector=detector,
        database=store,
        tracker=tracker,
    )

    stats = await pipeline.run()

    assert stats.frames_seen == 1
    assert stats.detections == 1
    assert stats.tracks == 1
    assert len(store.frames) == 1
    assert len(store.detections) == 1
    assert len(store.tracks) == 1
    assert camera.closed

