import pytest

from supercharged_coral.camera_service.providers import SyntheticCameraSource
from supercharged_coral.common.config import CameraConfig, MotionConfig
from supercharged_coral.dataset.database import SecurityDatabase
from supercharged_coral.motion_service.algorithms import FrameDifferencingMotionDetector
from supercharged_coral.pipeline.async_pipeline import AsyncFramePipeline, RegionEchoDetector
from supercharged_coral.tracker.sort_like import SortLikeTracker


@pytest.mark.asyncio
async def test_async_pipeline_persists_synthetic_motion_detections():
    config = CameraConfig(
        id="sim",
        source_type="synthetic",
        fps=1000,
        width=160,
        height=120,
    )
    camera = SyntheticCameraSource(config, frame_limit=4)
    database = SecurityDatabase(":memory:")
    database.initialize()
    pipeline = AsyncFramePipeline(
        cameras=[camera],
        motion_detectors={
            "sim": FrameDifferencingMotionDetector(
                MotionConfig(threshold=20, min_area=20, blur_kernel=1, morphology_kernel=1)
            )
        },
        object_detector=RegionEchoDetector(),
        database=database,
        tracker=SortLikeTracker(),
        max_frames_per_camera=4,
    )

    stats = await pipeline.run()

    assert stats.frames_seen == 4
    assert stats.motion_frames >= 1
    assert stats.detections >= 1
    assert database.scalar("SELECT COUNT(*) FROM frames") == 4
    assert database.scalar("SELECT COUNT(*) FROM detections") == stats.detections
    assert database.scalar("SELECT COUNT(*) FROM tracks") == stats.tracks
    database.close()

