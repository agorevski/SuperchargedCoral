from datetime import datetime, timezone

import numpy as np

from supercharged_coral.common.config import MotionConfig
from supercharged_coral.common.events import Frame
from supercharged_coral.motion_service.algorithms import (
    FrameDifferencingMotionDetector,
    RunningGaussianMotionDetector,
    benchmark_motion_algorithms,
)


def _frame(sequence: int, x: int) -> Frame:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[30:70, x : x + 40] = 255
    return Frame(
        camera_id="cam",
        image=image,
        timestamp=datetime.now(timezone.utc),
        sequence=sequence,
    )


def test_frame_differencing_detects_motion_region():
    detector = FrameDifferencingMotionDetector(
        MotionConfig(threshold=20, min_area=100, blur_kernel=1, morphology_kernel=1)
    )

    first = detector.detect(_frame(0, 10))
    second = detector.detect(_frame(1, 50))

    assert not first.has_motion
    assert second.has_motion
    assert second.regions[0].bbox.area >= 100
    assert second.heatmap is not None


def test_motion_benchmark_returns_metrics():
    frames = [_frame(0, 10), _frame(1, 20), _frame(2, 40)]
    results = benchmark_motion_algorithms(
        {
            "frame_differencing": FrameDifferencingMotionDetector(
                MotionConfig(threshold=20, min_area=50, blur_kernel=1, morphology_kernel=1)
            ),
            "running_gaussian": RunningGaussianMotionDetector(
                MotionConfig(threshold=20, min_area=50, blur_kernel=1, morphology_kernel=1)
            ),
        },
        frames,
    )

    assert results["frame_differencing"]["frames"] == 3
    assert results["frame_differencing"]["avg_latency_ms"] >= 0
    assert "motion_frame_ratio" in results["running_gaussian"]

