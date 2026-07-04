"""Runtime-checkable protocols that define Supercharged Coral service seams."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from supercharged_coral.common.events import (
    DatasetVersion,
    DeploymentRecord,
    Detection,
    EvaluationReport,
    Frame,
    HardwareHealth,
    ModelArtifact,
    MotionResult,
    QualityResult,
    Region,
    TeacherOutput,
    Track,
    TrainingRequest,
)

__all__ = [
    "ActiveLearningSampler",
    "CameraSource",
    "DashboardBackend",
    "DataAugmenter",
    "DatasetBuilder",
    "DeploymentStrategy",
    "DetectionVerifier",
    "Distiller",
    "EdgeTPUBackend",
    "Evaluator",
    "FrameQualityFilter",
    "FrameStore",
    "MetricsSink",
    "ModelCompiler",
    "ModelRegistry",
    "MotionDetector",
    "ObjectDetector",
    "ObjectTracker",
    "TeacherModel",
    "TeacherOutputStore",
    "Trainer",
]


@runtime_checkable
class CameraSource(Protocol):
    """Frame source interface for RTSP, ONVIF, USB, clips, files, and simulators."""

    @property
    def camera_id(self) -> str: ...

    def frames(self) -> AsyncIterator[Frame]: ...

    async def snapshot(self) -> Frame: ...

    async def close(self) -> None: ...


@runtime_checkable
class MotionDetector(Protocol):
    """Motion detector interface for frame differencing, MOG2, KNN, and custom models."""

    @property
    def algorithm_name(self) -> str: ...

    def detect(self, frame: Frame) -> MotionResult: ...


@runtime_checkable
class ObjectDetector(Protocol):
    """Object detector interface for Edge TPU, CPU, GPU, remote, and ensemble detectors."""

    @property
    def model_id(self) -> str: ...

    async def detect(self, frame: Frame, regions: Sequence[Region] | None = None) -> list[Detection]: ...


@runtime_checkable
class ObjectTracker(Protocol):
    """Tracker interface for SORT, ByteTrack, DeepSORT, and re-identification trackers."""

    def update(self, detections: Sequence[Detection]) -> list[Track]: ...


@runtime_checkable
class DetectionVerifier(Protocol):
    """Optional second-pass verifier for GPU VLMs, remote services, or ensembles."""

    @property
    def model_id(self) -> str: ...

    async def verify(self, frame: Frame, detections: Sequence[Detection]) -> list[Detection]: ...


@runtime_checkable
class FrameStore(Protocol):
    """Persistence boundary used by the online inference pipeline."""

    def record_frame(
        self,
        frame: Frame,
        *,
        image_path: str | None = None,
        lighting_estimate: float | None = None,
    ) -> int: ...

    def record_detection(self, frame_id: int, detection: Detection) -> int: ...

    def record_track(self, frame_id: int, track: Track) -> int: ...


@runtime_checkable
class TeacherOutputStore(Protocol):
    """Persistence boundary for VLM teacher outputs."""

    def record_teacher_output(
        self,
        frame_id: int,
        *,
        teacher_model: str,
        labels: list[str],
        boxes: list[dict[str, Any]],
        confidence: float | None = None,
        caption: str | None = None,
        explanation: str | None = None,
        segmentation_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int: ...


@runtime_checkable
class TeacherModel(Protocol):
    """GPU teacher interface for Florence, Grounding DINO, Qwen-VL, InternVL, Molmo, etc."""

    @property
    def model_id(self) -> str: ...

    async def label(self, frame: Frame, regions: Sequence[Region] | None = None) -> TeacherOutput: ...


@runtime_checkable
class FrameQualityFilter(Protocol):
    """Dataset quality gate for blur, bad crops, duplicates, and corrupt frames."""

    def evaluate(self, frame: Frame, regions: Sequence[Region] | None = None) -> QualityResult: ...


@runtime_checkable
class ActiveLearningSampler(Protocol):
    """Scores frames for teacher labeling or training priority."""

    def score(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        teacher_outputs: Sequence[TeacherOutput] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> float: ...


@runtime_checkable
class DatasetBuilder(Protocol):
    """Dataset construction interface for versioned, de-duplicated training sets."""

    def add_frame(
        self,
        frame_id: int,
        *,
        split: str,
        priority: float = 0.0,
        reason: str | None = None,
    ) -> None: ...

    def build(self, name: str, version: str, metadata: Mapping[str, Any] | None = None) -> DatasetVersion: ...


@runtime_checkable
class DataAugmenter(Protocol):
    """Image augmentation interface for weather, lighting, compression, and camera artifacts."""

    def augment(self, frame: Frame, annotations: Sequence[Detection]) -> list[tuple[Frame, list[Detection]]]: ...


@runtime_checkable
class Trainer(Protocol):
    """Training interface for fine-tuning EfficientDet-Lite and other student detectors."""

    def train(self, request: TrainingRequest) -> ModelArtifact: ...


@runtime_checkable
class Distiller(Protocol):
    """Knowledge-distillation interface for teacher-to-student training."""

    def distill(self, request: TrainingRequest, teacher_outputs: Sequence[TeacherOutput]) -> ModelArtifact: ...


@runtime_checkable
class Evaluator(Protocol):
    """Evaluation interface for model quality, latency, hardware, and regression gates."""

    def evaluate(self, model: ModelArtifact, dataset: DatasetVersion) -> EvaluationReport: ...


@runtime_checkable
class ModelCompiler(Protocol):
    """Compiler interface for TensorFlow Lite and Edge TPU model artifacts."""

    def compile(self, model: ModelArtifact) -> ModelArtifact: ...


@runtime_checkable
class ModelRegistry(Protocol):
    """Model registry interface for leaderboard, production, and archived artifacts."""

    def register(self, model: ModelArtifact, report: EvaluationReport | None = None) -> None: ...

    def get(self, model_id: str) -> ModelArtifact: ...

    def production_model(self) -> ModelArtifact | None: ...

    def promote(self, model_id: str, reason: str) -> DeploymentRecord: ...


@runtime_checkable
class DeploymentStrategy(Protocol):
    """Deployment interface for hot-swapping models and rolling back safely."""

    def deploy(self, model: ModelArtifact, reason: str) -> DeploymentRecord: ...

    def rollback(self, target_model_id: str, reason: str) -> DeploymentRecord: ...


@runtime_checkable
class EdgeTPUBackend(Protocol):
    """Hardware runtime interface for Coral discovery, health, and telemetry."""

    def list_devices(self) -> list[Any]: ...

    def health(self) -> HardwareHealth | dict[str, Any]: ...


@runtime_checkable
class MetricsSink(Protocol):
    """Metrics sink interface for experiment tracking, dashboards, and telemetry backends."""

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        model_id: str | None = None,
        slice_name: str = "global",
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...


@runtime_checkable
class DashboardBackend(Protocol):
    """Dashboard data boundary so the UI can swap FastAPI, GraphQL, or local adapters."""

    async def status(self) -> Mapping[str, Any]: ...

    async def recent_detections(self, *, limit: int = 100) -> Sequence[Detection]: ...
