# Interface-First Architecture

[Documentation index](README.md) · [Project specification](PROJECT_SPEC.md) · [Repository README](../README.md)

Supercharged Coral uses structural interfaces (`typing.Protocol`) so every major component can be swapped without changing pipeline code. Implementations do not need to inherit from a base class; they only need to provide the methods and properties defined in `supercharged_coral.interfaces` and use shared datatypes from `supercharged_coral.common.events`.

## Design rules

- Depend on protocols and shared event datatypes at service boundaries.
- Inject concrete implementations through orchestration code instead of importing sibling service implementations directly.
- Keep hardware, storage, model, and dashboard choices behind replaceable adapters.
- Make test doubles ordinary Python objects that satisfy the same structural contracts.

## Core contract map

| Interface | Purpose | Example implementations |
| --- | --- | --- |
| `CameraSource` | Produces timestamped frames and snapshots. | RTSP, ONVIF, USB, video file, synthetic simulator |
| `MotionDetector` | Converts a frame into motion regions and masks. | Frame differencing, running Gaussian, MOG2, KNN |
| `ObjectDetector` | Detects configured classes inside a full frame or motion crops. | Edge TPU EfficientDet-Lite, GPU detector, remote detector, ensemble |
| `ObjectTracker` | Converts detections into stable tracks. | SORT-like baseline, ByteTrack, DeepSORT |
| `DetectionVerifier` | Optional second-pass verification. | GPU VLM, remote verification service, detector ensemble |
| `FrameStore` | Persists frames, detections, and tracks. | SQLite, PostgreSQL, object-store-backed repository |
| `TeacherModel` | Produces teacher labels, boxes, captions, and explanations. | Florence-2, Grounding DINO, Qwen-VL, InternVL, Molmo |
| `TeacherOutputStore` | Persists teacher outputs. | SQLite, PostgreSQL, lakehouse |
| `FrameQualityFilter` | Accepts or rejects training candidates. | Blur filter, crop-quality filter, duplicate filter |
| `ActiveLearningSampler` | Scores samples for labeling or training priority. | Uncertainty, disagreement, novelty, rare-weather samplers |
| `DatasetBuilder` | Creates versioned train/validation/test datasets. | Local filesystem builder, cloud/object-store builder |
| `DataAugmenter` | Applies training augmentations. | Weather, lighting, compression, camera-shake augmentation |
| `Trainer` | Fine-tunes student detectors. | TensorFlow EfficientDet-Lite trainer, distributed trainer |
| `Distiller` | Trains students from teacher outputs. | Soft-label distillation, feature distillation |
| `Evaluator` | Produces quality, slice, and hardware reports. | mAP evaluator, latency evaluator, regression gate evaluator |
| `ModelCompiler` | Converts or compiles model artifacts. | TFLite converter, Edge TPU compiler |
| `ModelRegistry` | Tracks model artifacts and promotions. | SQLite registry, MLflow adapter, filesystem registry |
| `DeploymentStrategy` | Hot-swaps or rolls back production models. | Local deployment, multi-TPU deployment, remote deployment |
| `EdgeTPUBackend` | Reports Coral devices and hardware health. | PyCoral runtime, mocked test runtime |
| `MetricsSink` | Records metrics for dashboards and experiments. | SQLite, TensorBoard, Prometheus, MLflow |
| `DashboardBackend` | Provides data to the UI. | FastAPI backend, GraphQL backend, local read-only adapter |

## Plug-in rule

New components should depend on datatypes from `supercharged_coral.common.events` and protocols from `supercharged_coral.interfaces`. They should not import concrete implementations from sibling services unless they are intentionally composing a known implementation.

## Minimal detector plug-in

```python
from collections.abc import Sequence

from supercharged_coral.common.events import Detection, Frame, Region


class MyDetector:
    model_id = "my-detector-v1"

    async def detect(
        self,
        frame: Frame,
        regions: Sequence[Region] | None = None,
    ) -> list[Detection]:
        # Run any local, remote, TPU, GPU, or ensemble model here.
        return []
```

Because interfaces are structural, `MyDetector` can be injected wherever an `ObjectDetector` is expected without subclassing.

## Pipeline injection example

```python
from supercharged_coral.pipeline import AsyncFramePipeline

pipeline = AsyncFramePipeline(
    cameras=[my_camera_source],
    motion_detectors={my_camera_source.camera_id: my_motion_detector},
    object_detector=my_object_detector,
    database=my_frame_store,
    tracker=my_tracker,
)
```

This is the standard extension seam for swapping camera providers, motion algorithms, detectors, stores, and trackers.
