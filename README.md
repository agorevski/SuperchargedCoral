# Supercharged Coral

Supercharged Coral is a modular, self-hosted home security AI platform optimized for Google Coral Edge TPU inference and GPU-assisted continuous learning. The project is intentionally interface-first: every major runtime and training component is replaceable behind a structural Python protocol.

## Current implementation

The initial slice is an installable Python package with:

- Structural component interfaces for cameras, motion detection, detection, tracking, persistence, training, evaluation, deployment, hardware, telemetry, and dashboards.
- Camera provider abstractions, synthetic camera simulation, motion detection algorithms, and a baseline async frame pipeline.
- SQLite persistence schemas, evaluation primitives, Coral runtime discovery hooks, and CLI smoke-test/database commands.
- Unit and integration tests covering the implemented foundations.

The complete product vision is tracked in [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

## Quick start

Requires Python 3.12+.

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m supercharged_coral.cli smoke-test --frames 5
python3 -m supercharged_coral.cli init-db --database supercharged-coral.db
python3 -m supercharged_coral.cli download-frigate-events --base-url http://frigate:5000 --output-dir data/frigate-events
python3 -m supercharged_coral.cli download-frigate-events --base-url http://frigate:5000 --output-dir data/frigate-snapshots --media snapshot
```

After installation, the packaged console script is also available as `supercharged-coral`.

The Frigate downloader pages through `/api/events` and downloads each available
event clip from `/api/events/{event_id}/clip.mp4`, or each best event snapshot
from `/api/events/{event_id}/snapshot.jpg` with `--media snapshot`, using 10
parallel workers. It skips any non-empty MP4/JPG already present in the output
directory. Set `FRIGATE_API_KEY` or pass `--api-key` for bearer-token
authentication, and repeat `--header "Name: value"` for other Frigate proxy
headers.

## Documentation

| Document | Purpose |
| --- | --- |
| [`docs/README.md`](docs/README.md) | Documentation index and recommended reading order. |
| [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) | Product requirements, target architecture, and long-term roadmap. |
| [`docs/INTERFACES.md`](docs/INTERFACES.md) | Protocol contract map and plug-in implementation guidance. |
| [`RUNBOOK.md`](RUNBOOK.md) | End-to-end model training workflow from zero training data through deployment. |

## Architecture at a glance

The runtime path is intentionally small and composable:

```text
Camera source
  -> motion detection
  -> object detection
  -> object tracking
  -> optional verification
  -> frame/event persistence
  -> dataset and training feedback loop
```

The near-term implementation focuses on the camera, motion, detection, tracking, persistence, and Coral runtime seams. The longer-term architecture adds teacher models, active learning, model evaluation gates, Edge TPU compilation, safe deployment, and dashboard integrations without changing the orchestration contracts.

## Pluggable interfaces

Supercharged Coral uses structural interfaces defined in `src/supercharged_coral/interfaces.py`. These interfaces use `typing.Protocol`, so plug-ins do not need to inherit from framework base classes. Any object can be injected as long as it exposes the expected methods, properties, and shared datatypes from `supercharged_coral.common.events`.

| Interface | What it lets you swap |
| --- | --- |
| `CameraSource` | Frame producers such as RTSP, ONVIF, USB, file, replay, or synthetic cameras. |
| `MotionDetector` | Motion algorithms such as frame differencing, MOG2, KNN, or custom models. |
| `ObjectDetector` | Edge TPU, CPU, GPU, remote, or ensemble object detectors. |
| `ObjectTracker` | Tracking implementations such as SORT, ByteTrack, DeepSORT, or re-identification trackers. |
| `DetectionVerifier` | Optional second-pass verification with VLMs, remote services, or ensembles. |
| `FrameStore` and `TeacherOutputStore` | Persistence backends such as SQLite, PostgreSQL, object storage, or lakehouse adapters. |
| `TeacherModel`, `FrameQualityFilter`, `ActiveLearningSampler`, `DatasetBuilder`, and `DataAugmenter` | Continuous-learning components for labeling, filtering, sampling, dataset construction, and augmentation. |
| `Trainer`, `Distiller`, `Evaluator`, `ModelCompiler`, `ModelRegistry`, and `DeploymentStrategy` | Training, evaluation, compilation, model promotion, hot-swap, and rollback backends. |
| `EdgeTPUBackend`, `MetricsSink`, and `DashboardBackend` | Hardware runtime, telemetry, experiment tracking, and dashboard adapters. |

A detector plug-in only needs to expose `model_id` and an async `detect` method with the expected signature:

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
        return []
```

Because the contract is structural, `MyDetector` can be passed anywhere an `ObjectDetector` is expected. The async pipeline uses the same pattern for runtime composition:

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

See [`docs/INTERFACES.md`](docs/INTERFACES.md) for the full contract map and additional examples.

## Project layout

```text
src/supercharged_coral/
  cli.py              CLI entry points for smoke tests and database initialization.
  interfaces.py       Structural protocols for every pluggable component.
  common/             Shared configuration, events, and datatypes.
  camera_service/     Camera source abstractions and providers.
  motion_service/     Motion detection algorithms and benchmarking seams.
  object_detector/    Object detector contracts and adapters.
  edge_tpu/           Coral discovery, health, and model runtime scaffolding.
  tracker/            Tracking contracts and a SORT-like baseline.
  dataset/            SQLite schemas and persistence helpers.
  evaluation/         Metric primitives and evaluation seams.
  teacher/            Teacher model interfaces.
  training/           Training and continuous-learning interfaces.
  deployment/         Model registry, compiler, and deployment interfaces.
  dashboard/          Dashboard backend interfaces.
  pipeline/           Async camera -> motion -> detection -> tracking pipeline.
config/example.yaml   Example platform configuration.
docs/                 Product, architecture, and interface documentation.
tests/                Unit and integration tests for implemented components.
pyproject.toml        Package metadata, dependencies, scripts, and test settings.
```
