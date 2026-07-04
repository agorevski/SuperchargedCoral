# Supercharged Coral: Self-Improving Home Security AI Platform

[Documentation index](README.md) · [Interface guide](INTERFACES.md) · [Repository README](../README.md)

## Document status

This document captures the target product requirements and roadmap-level architecture. For the current implementation status and repository layout, see the top-level [`README.md`](../README.md).

## Contents

- [Mission](#mission)
- [High-Level Goals](#high-level-goals)
- [Target Hardware and Runtime](#target-hardware-and-runtime)
- [Service Architecture](#service-architecture)
- [Interface-First Architecture](#interface-first-architecture)
- [Camera Service](#camera-service)
- [Motion Detection](#motion-detection)
- [Frame Pipeline](#frame-pipeline)
- [Google Coral Edge TPU](#google-coral-edge-tpu)
- [EfficientDet-Lite Tooling](#efficientdet-lite-tooling)
- [Initial Object Classes](#initial-object-classes)
- [Object Tracking](#object-tracking)
- [Teacher Models](#teacher-models)
- [Automatic Dataset Builder](#automatic-dataset-builder)
- [Active Learning](#active-learning)
- [Fine-Tuning Pipeline](#fine-tuning-pipeline)
- [Knowledge Distillation](#knowledge-distillation)
- [Data Augmentation](#data-augmentation)
- [Evaluation Framework](#evaluation-framework)
- [Regression Testing and Deployment Gates](#regression-testing-and-deployment-gates)
- [Continuous Learning Loop](#continuous-learning-loop)
- [Dashboard](#dashboard)
- [Database Schemas](#database-schemas)
- [Testing Goals](#testing-goals)
- [Documentation Goals](#documentation-goals)
- [Stretch Goals](#stretch-goals)
- [Success Criteria](#success-criteria)

## Mission

Build a production-quality, modular, well-tested home security AI platform optimized for Google Coral Edge TPU hardware. The system should operate as an open-source, self-hosted AI security platform that can continuously collect data, improve its own models, prove that new models are better, and deploy only non-regressing model versions.

## High-Level Goals

- Connect to multiple 4K IP cameras.
- Continuously capture frames.
- Use motion detection to reduce inference cost.
- Perform real-time object detection on Google Coral Edge TPU.
- Use GPU-based Vision Language Models as teacher models.
- Automatically create and version training datasets.
- Continuously fine-tune EfficientDet-Lite.
- Measure whether retrained models improve over production.
- Automatically deploy better models.
- Prevent quality regressions.

The initial target is 4 cameras, with the architecture scaling to dozens of cameras, multiple Coral accelerators, and multiple GPUs.

## Target Hardware and Runtime

- Google Coral USB Accelerator.
- 4 NVIDIA GPUs with 48 GB VRAM each.
- Linux.
- Python 3.12+.
- CUDA.
- TensorFlow Lite.
- TensorFlow.
- PyTorch.
- ONNX.
- Docker.

## Service Architecture

The system is designed as independent services with clean interfaces:

- `camera-service`
- `motion-service`
- `object-detector`
- `edge-tpu`
- `tracker`
- `teacher`
- `dataset`
- `training`
- `evaluation`
- `deployment`
- `dashboard`
- `common`
- `config`
- `tests`

## Interface-First Architecture

Every major component must sit behind a protocol in `supercharged_coral.interfaces` so implementations can be swapped without changing orchestration code. These are structural interfaces, so plug-ins do not need to inherit from framework base classes; they only need to expose the expected methods.

Current and planned extension points:

- `CameraSource` for RTSP, ONVIF, USB, video files, clips, and simulators.
- `MotionDetector` for frame differencing, running Gaussian, MOG2, KNN, and learned motion filters.
- `ObjectDetector` for Edge TPU EfficientDet-Lite, GPU detectors, remote detectors, and ensembles.
- `ObjectTracker` for SORT, ByteTrack, DeepSORT, and re-identification trackers.
- `DetectionVerifier` and `TeacherModel` for GPU VLM verification and teacher labeling.
- `FrameStore` and `TeacherOutputStore` for SQLite, PostgreSQL, and object-store-backed persistence.
- `FrameQualityFilter`, `ActiveLearningSampler`, `DatasetBuilder`, and `DataAugmenter` for dataset generation.
- `Trainer` and `Distiller` for fine-tuning and teacher-to-student distillation.
- `Evaluator`, `MetricsSink`, `ModelCompiler`, `ModelRegistry`, and `DeploymentStrategy` for quality gates and safe deployment.
- `EdgeTPUBackend` for Coral discovery, telemetry, and mocked hardware tests.
- `DashboardBackend` for interchangeable dashboard APIs.

The implementation guide for these contracts is maintained in [`INTERFACES.md`](INTERFACES.md).

## Camera Service

Supported sources:

- RTSP streams.
- ONVIF cameras.
- USB cameras.
- Video files.
- Recorded clips.

Required capabilities:

- Frame capture.
- Reconnect logic.
- Dropped-frame handling.
- Timestamp synchronization.
- Configurable FPS.
- Per-camera settings.
- Snapshot API.

## Motion Detection

Because cameras are static, motion detection should reduce downstream inference costs.

Algorithms:

- Background subtraction.
- Running Gaussian background model.
- MOG2.
- KNN.
- Frame differencing.
- Morphological cleanup.
- Adaptive thresholds.
- ROI masking.
- Configurable ignored regions.
- Motion heatmaps.

Each algorithm must be benchmarked for accuracy and performance before choosing the default.

## Frame Pipeline

The asynchronous pipeline is:

```text
Camera
  -> Motion Detection
  -> Crop Motion Regions
  -> Object Detection
  -> Object Tracking
  -> Optional GPU Verification
  -> Database
  -> Training Dataset
```

## Google Coral Edge TPU

Required capabilities:

- Device detection.
- Health monitoring.
- Model loading.
- Hot swapping.
- Multiple TPU support.
- Inference benchmarking.
- Automatic batching where appropriate.
- EfficientDet-Lite model support.

Benchmarks:

- Latency.
- FPS.
- CPU utilization.
- Edge TPU utilization.

## EfficientDet-Lite Tooling

Support:

- Lite0.
- Lite1.
- Lite2.
- Lite3.
- Lite4.

Tooling:

- Download pretrained models.
- Train.
- Fine-tune.
- Evaluate.
- Export.
- Convert to TensorFlow Lite.
- Compile for Edge TPU.
- Deploy automatically only after passing regression gates.
- Benchmark every model version.

## Initial Object Classes

The class list must be configurable. Initial classes:

- Person
- Package
- Dog
- Cat
- Car
- Truck
- Motorcycle
- Bicycle
- Deer
- Bear
- Rabbit
- Raccoon
- Coyote
- Bird

Users must be able to extend this list easily.

## Object Tracking

Implement and benchmark:

- ByteTrack.
- SORT.
- DeepSORT.

Evaluate:

- Track stability.
- ID switching.
- Speed.

## Teacher Models

GPU teacher inference should support modern vision models such as:

- Florence-2.
- Grounding DINO.
- Qwen2.5-VL.
- InternVL.
- Molmo.

Teacher outputs should include:

- Improved bounding boxes.
- Improved labels.
- Confidence estimates.
- Image captions.
- Explanations.
- Segmentation masks when available.

All teacher outputs must be stored.

## Automatic Dataset Builder

Continuously build datasets and store:

- Original frame.
- Motion crop.
- Bounding boxes.
- Teacher labels.
- Confidence.
- Timestamp.
- Optional weather metadata.
- Lighting estimate.
- Camera ID.
- Track ID.

Dataset rules:

- Every image is versioned.
- Duplicated images are rejected.
- Blurry frames are rejected.
- Poor crops are rejected.

## Active Learning

Prioritize training samples from:

- Low-confidence detections.
- Teacher disagreement.
- Multiple-detector disagreement.
- Novel scenes.
- Unknown objects.
- Rare weather.
- Night.
- Snow.
- Rain.
- Fog.

## Fine-Tuning Pipeline

The retraining system must include:

- Data cleaning.
- Augmentation.
- Train/validation/test split.
- Hyperparameter tuning.
- Mixed precision.
- Checkpointing.
- Resume support.
- TensorBoard integration.
- Early stopping.
- Automatic experiment tracking.
- Incremental fine tuning.

## Knowledge Distillation

Implement teacher-to-student distillation where EfficientDet-Lite learns from teacher outputs.

Targets:

- Bounding boxes.
- Class probabilities.
- Confidence.
- Objectness scores.

Optional targets:

- Intermediate feature matching.
- Soft labels.
- Temperature scaling.

Distillation must be compared against ordinary fine tuning.

## Data Augmentation

Augmentations should be configurable:

- Brightness.
- Contrast.
- Rain.
- Snow.
- Fog.
- Motion blur.
- JPEG artifacts.
- Compression.
- Noise.
- Gamma.
- Perspective.
- Occlusion.
- IR simulation.
- Night simulation.
- Shadow simulation.
- Camera shake.

The evaluation framework must measure which augmentations improve performance.

## Evaluation Framework

Compute:

- mAP.
- mAP50.
- mAP75.
- Precision.
- Recall.
- F1.
- ROC.
- PR curves.
- Confusion matrix.
- False positives.
- False negatives.
- Latency.
- FPS.
- Memory usage.
- Power consumption.
- Edge TPU utilization.
- GPU utilization.
- Per-class metrics.
- Per-camera metrics.
- Per-time-of-day metrics.
- Per-weather metrics.
- Night-only metrics.
- Motion-only metrics.

Every run should produce graphs and reproducible reports.

## Regression Testing and Deployment Gates

A newly trained model must never automatically deploy without validation.

Run each candidate model against historical validation videos and compare it to production. Reject a model if:

- Precision decreases.
- Recall decreases.
- False positives increase.
- Latency exceeds threshold.
- Edge TPU compilation fails.

Maintain a leaderboard of every model ever trained.

## Continuous Learning Loop

```text
Collect data
  -> Teacher labels
  -> Review dataset quality
  -> Fine tune
  -> Evaluate
  -> Benchmark
  -> Regression tests
  -> Compile Edge TPU model
  -> Deploy if better
  -> Archive previous version
```

Everything must be reproducible.

## Dashboard

The web UI should include:

- Camera status.
- Live detections.
- Historical detections.
- Training progress.
- Dataset browser.
- Teacher predictions.
- False positives.
- Model leaderboard.
- Performance charts.
- Hardware monitoring.
- Inference latency.
- GPU usage.
- TPU usage.
- Model version history.
- Deployment history.

## Database Schemas

Design schemas for:

- Frames.
- Detections.
- Tracks.
- Training datasets.
- Teacher outputs.
- Experiments.
- Metrics.
- Model registry.
- Deployment history.

## Testing Goals

Test categories:

- Unit tests.
- Integration tests.
- Performance tests.
- Stress tests.
- Regression tests.
- Edge TPU tests.
- Training pipeline tests.
- Camera simulator tests.
- Synthetic dataset tests.

Target code coverage: greater than 90%.

## Documentation Goals

Produce documentation for:

- Architecture.
- Training guide.
- Deployment guide.
- Edge TPU guide.
- Fine-tuning guide.
- Adding new cameras.
- Adding new object classes.
- Dataset format.
- Model registry.
- Benchmark methodology.
- Troubleshooting.

## Stretch Goals

- Multi-camera object re-identification.
- Cross-camera tracking.
- Face recognition as an optional plugin.
- License plate recognition plugin.
- Pose estimation.
- Segmentation.
- Foundation-model-assisted search.
- Natural-language event search.
- RAG over historical events.
- Automatic anomaly detection.
- Open-set object detection.
- Unknown-object clustering.
- Event summarization.
- Notification prioritization.
- Remote inference.
- Distributed training.
- Federated learning.

## Success Criteria

The completed system should:

- Run in real time on Google Coral.
- Continuously improve through self-training.
- Demonstrate measurable improvements over the baseline EfficientDet-Lite model.
- Produce reproducible evaluation reports proving gains in precision, recall, mAP, latency, and false-positive reduction.
- Be modular, extensible, well-tested, and documented well enough for external contributors to build upon.
