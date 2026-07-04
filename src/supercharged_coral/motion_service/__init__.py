"""Public motion detection algorithms, factories, and benchmarks."""

from supercharged_coral.motion_service.algorithms import (
    BaseMotionDetector,
    FrameDifferencingMotionDetector,
    KNNMotionDetector,
    MOG2MotionDetector,
    MotionHeatmap,
    RunningGaussianMotionDetector,
    benchmark_motion_algorithms,
    build_motion_detector,
)
from supercharged_coral.interfaces import MotionDetector

__all__ = [
    "BaseMotionDetector",
    "FrameDifferencingMotionDetector",
    "KNNMotionDetector",
    "MOG2MotionDetector",
    "MotionHeatmap",
    "MotionDetector",
    "RunningGaussianMotionDetector",
    "benchmark_motion_algorithms",
    "build_motion_detector",
]
