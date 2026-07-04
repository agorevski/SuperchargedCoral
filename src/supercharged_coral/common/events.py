"""Serializable event and data-transfer records shared across services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

__all__ = [
    "DatasetVersion",
    "DeploymentRecord",
    "Detection",
    "EvaluationReport",
    "Frame",
    "HardwareHealth",
    "ModelArtifact",
    "MotionRegion",
    "MotionResult",
    "QualityResult",
    "Region",
    "TeacherOutput",
    "Track",
    "TrainingRequest",
    "utc_now",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("Region width and height must be non-negative")

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> Region:
        return cls(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))

    def clamp(self, frame_shape: tuple[int, ...]) -> Region:
        frame_height, frame_width = frame_shape[:2]
        x1 = min(max(self.x, 0), frame_width)
        y1 = min(max(self.y, 0), frame_height)
        x2 = min(max(self.x2, 0), frame_width)
        y2 = min(max(self.y2, 0), frame_height)
        return Region.from_xyxy(x1, y1, x2, y2)

    def to_slice(self, frame_shape: tuple[int, ...]) -> tuple[slice, slice]:
        clamped = self.clamp(frame_shape)
        return slice(clamped.y, clamped.y2), slice(clamped.x, clamped.x2)

    def intersects(self, other: Region) -> bool:
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )

    def iou(self, other: Region) -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        return 0.0 if union == 0 else intersection / union


@dataclass(frozen=True)
class Frame:
    camera_id: str
    image: np.ndarray
    timestamp: datetime = field(default_factory=utc_now)
    sequence: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(frozen=True)
class MotionRegion:
    bbox: Region
    score: float


@dataclass(frozen=True)
class MotionResult:
    has_motion: bool
    regions: list[MotionRegion]
    mask: np.ndarray
    heatmap: np.ndarray | None
    algorithm: str
    latency_ms: float


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: Region
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox: Region
    age: int
    hits: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    score: float
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherOutput:
    teacher_model: str
    labels: list[str]
    boxes: list[Region]
    confidences: list[float]
    caption: str | None = None
    explanation: str | None = None
    segmentation_mask_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    version: str
    frame_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRequest:
    dataset: DatasetVersion
    base_model_id: str
    model_family: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    family: str
    version: str
    path: str
    compiled_edge_tpu_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    model_id: str
    metrics: dict[str, float]
    slice_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    passed_regression_gates: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentRecord:
    model_id: str
    deployed_at: datetime
    deployed_by: str
    previous_model_id: str | None
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareHealth:
    healthy: bool
    device_count: int
    devices: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
