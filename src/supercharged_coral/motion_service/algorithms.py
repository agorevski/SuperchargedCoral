"""Motion detection baselines and utilities for frame streams."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean

import numpy as np

from supercharged_coral.common.config import MotionConfig
from supercharged_coral.common.events import Frame, MotionRegion, MotionResult, Region
from supercharged_coral.interfaces import MotionDetector

__all__ = [
    "BaseMotionDetector",
    "FrameDifferencingMotionDetector",
    "KNNMotionDetector",
    "MOG2MotionDetector",
    "MotionHeatmap",
    "RunningGaussianMotionDetector",
    "benchmark_motion_algorithms",
    "build_motion_detector",
]


def _cv2():
    import cv2

    return cv2


def _odd_kernel(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _to_gray(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _blur(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return gray
    cv2 = _cv2()
    kernel = _odd_kernel(kernel_size)
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def _cleanup(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    cv2 = _cv2()
    kernel = np.ones((_odd_kernel(kernel_size), _odd_kernel(kernel_size)), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


def _apply_ignored_regions(mask: np.ndarray, ignored_regions: list[Region]) -> np.ndarray:
    if not ignored_regions:
        return mask
    filtered = mask.copy()
    for region in ignored_regions:
        rows, cols = region.to_slice(filtered.shape)
        filtered[rows, cols] = 0
    return filtered


def _extract_regions(mask: np.ndarray, min_area: int) -> list[MotionRegion]:
    cv2 = _cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[MotionRegion] = []
    frame_area = max(1, mask.shape[0] * mask.shape[1])
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append(MotionRegion(bbox=Region(x, y, width, height), score=area / frame_area))
    regions.sort(key=lambda region: region.bbox.area, reverse=True)
    return regions


@dataclass
class MotionHeatmap:
    decay: float = 0.95
    accumulator: np.ndarray | None = None

    def update(self, mask: np.ndarray) -> np.ndarray:
        normalized = mask.astype(np.float32) / 255.0
        if self.accumulator is None or self.accumulator.shape != normalized.shape:
            self.accumulator = normalized
        else:
            self.accumulator = self.accumulator * self.decay + normalized * (1.0 - self.decay)
        return self.accumulator.copy()


class BaseMotionDetector:
    algorithm_name = "base"

    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self.heatmap = MotionHeatmap()

    def detect(self, frame: Frame) -> MotionResult:
        raise NotImplementedError

    def _threshold(self, diff: np.ndarray) -> np.ndarray:
        cv2 = _cv2()
        if self.config.threshold is None:
            threshold = max(15, min(255, int(float(diff.mean()) + float(diff.std()) * 2.0)))
        else:
            threshold = self.config.threshold
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)

    def _finalize(self, mask: np.ndarray, started_at: float) -> MotionResult:
        mask = _cleanup(mask, self.config.morphology_kernel)
        mask = _apply_ignored_regions(mask, self.config.ignore_regions)
        regions = _extract_regions(mask, self.config.min_area)
        heatmap = self.heatmap.update(mask)
        return MotionResult(
            has_motion=bool(regions),
            regions=regions,
            mask=mask,
            heatmap=heatmap,
            algorithm=self.algorithm_name,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    def _empty_result(self, frame: Frame, started_at: float) -> MotionResult:
        mask = np.zeros(frame.image.shape[:2], dtype=np.uint8)
        return MotionResult(
            has_motion=False,
            regions=[],
            mask=mask,
            heatmap=self.heatmap.update(mask),
            algorithm=self.algorithm_name,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )


class FrameDifferencingMotionDetector(BaseMotionDetector):
    algorithm_name = "frame_differencing"

    def __init__(self, config: MotionConfig | None = None) -> None:
        super().__init__(config)
        self.previous_gray: np.ndarray | None = None

    def detect(self, frame: Frame) -> MotionResult:
        started_at = time.perf_counter()
        cv2 = _cv2()
        gray = _blur(_to_gray(frame.image), self.config.blur_kernel)
        if self.previous_gray is None:
            self.previous_gray = gray
            return self._empty_result(frame, started_at)
        diff = cv2.absdiff(self.previous_gray, gray)
        self.previous_gray = gray
        return self._finalize(self._threshold(diff), started_at)


class RunningGaussianMotionDetector(BaseMotionDetector):
    algorithm_name = "running_gaussian"

    def __init__(self, config: MotionConfig | None = None) -> None:
        super().__init__(config)
        self.background: np.ndarray | None = None

    def detect(self, frame: Frame) -> MotionResult:
        started_at = time.perf_counter()
        cv2 = _cv2()
        gray = _blur(_to_gray(frame.image), self.config.blur_kernel)
        if self.background is None:
            self.background = gray.astype(np.float32)
            return self._empty_result(frame, started_at)
        background_u8 = cv2.convertScaleAbs(self.background)
        diff = cv2.absdiff(background_u8, gray)
        cv2.accumulateWeighted(gray, self.background, self.config.running_alpha)
        return self._finalize(self._threshold(diff), started_at)


class _BackgroundSubtractorMotionDetector(BaseMotionDetector):
    algorithm_name = "background_subtractor"

    def __init__(self, config: MotionConfig | None = None) -> None:
        super().__init__(config)
        self._subtractor = self._create_subtractor()

    def _create_subtractor(self):
        raise NotImplementedError

    def detect(self, frame: Frame) -> MotionResult:
        started_at = time.perf_counter()
        gray = _blur(_to_gray(frame.image), self.config.blur_kernel)
        mask = self._subtractor.apply(gray)
        if self.config.threshold is not None:
            mask = self._threshold(mask)
        return self._finalize(mask, started_at)


class MOG2MotionDetector(_BackgroundSubtractorMotionDetector):
    algorithm_name = "mog2"

    def _create_subtractor(self):
        cv2 = _cv2()
        return cv2.createBackgroundSubtractorMOG2(detectShadows=True)


class KNNMotionDetector(_BackgroundSubtractorMotionDetector):
    algorithm_name = "knn"

    def _create_subtractor(self):
        cv2 = _cv2()
        return cv2.createBackgroundSubtractorKNN(detectShadows=True)


def build_motion_detector(config: MotionConfig) -> MotionDetector:
    detectors = {
        "frame_differencing": FrameDifferencingMotionDetector,
        "running_gaussian": RunningGaussianMotionDetector,
        "mog2": MOG2MotionDetector,
        "knn": KNNMotionDetector,
    }
    return detectors[config.algorithm](config)


def benchmark_motion_algorithms(
    detectors: Mapping[str, MotionDetector], frames: list[Frame]
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, detector in detectors.items():
        latencies: list[float] = []
        motion_frames = 0
        region_counts: list[int] = []
        for frame in frames:
            result = detector.detect(frame)
            latencies.append(result.latency_ms)
            motion_frames += int(result.has_motion)
            region_counts.append(len(result.regions))
        frame_count = max(1, len(frames))
        results[name] = {
            "frames": float(len(frames)),
            "avg_latency_ms": mean(latencies) if latencies else 0.0,
            "motion_frame_ratio": motion_frames / frame_count,
            "avg_regions": mean(region_counts) if region_counts else 0.0,
        }
    return results
