# Supercharged Coral Training Runbook

This runbook explains the end-to-end path for training a Supercharged Coral object-detection model when you do not have any training data yet. It covers the practical workflow from first data collection through teacher labeling, human review, dataset versioning, fine-tuning, evaluation, Edge TPU compilation, deployment, and the next continuous-learning cycle.

## Current repository status

The repository currently provides the foundation for this workflow:

- Runtime interfaces for cameras, motion detection, object detection, tracking, teacher labeling, dataset building, training, evaluation, model compilation, registry, and deployment in `src/supercharged_coral/interfaces.py`.
- Shared data records such as `Frame`, `Detection`, `TeacherOutput`, `DatasetVersion`, `TrainingRequest`, `ModelArtifact`, and `EvaluationReport` in `src/supercharged_coral/common/events.py`.
- SQLite schemas for frames, detections, tracks, teacher outputs, training datasets, experiments, metrics, model registry, and deployment history in `src/supercharged_coral/dataset/database.py`.
- A synthetic local pipeline smoke test in `src/supercharged_coral/cli.py`.

Concrete teacher-model adapters, dataset builders, EfficientDet-Lite trainers, Edge TPU compilers, registries, and deployment strategies still need to be implemented behind those interfaces. Treat this runbook as the operating blueprint and acceptance checklist for those adapters.

## End-to-end workflow

```text
Set up environment
  -> define classes and acceptance gates
  -> collect raw camera data
  -> run baseline inference and teacher labeling
  -> review and correct labels
  -> build a frozen dataset version
  -> train or fine-tune a student detector
  -> evaluate against baseline and production
  -> convert and compile for Edge TPU
  -> register and deploy only if gates pass
  -> collect errors for the next dataset version
```

## 1. Set up the local environment

Requires Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Validate the current repository foundation:

```bash
python -m pytest
python -m supercharged_coral.cli smoke-test --frames 25 --database data/smoke-test.db
python -m supercharged_coral.cli init-db --database data/supercharged-coral.db
```

Install training dependencies when you are ready to run real model training:

```bash
python -m pip install -e ".[training]"
```

If TensorFlow, PyTorch, or CUDA wheels fail to install, install the versions recommended for your GPU and driver first, then re-run the editable project install.

## 2. Create the training workspace

Use a predictable local layout and keep raw data, reviewed labels, frozen datasets, model artifacts, and reports separate.

```text
data/
  raw/
    front-door/
    driveway/
  labels/
    teacher/
    reviewed/
  datasets/
    security-v001/
      images/
        train/
        validation/
        test/
      annotations/
        instances_train.json
        instances_validation.json
        instances_test.json
      metadata.json
  experiments/
    security-v001/
  models/
    base/
    candidates/
    compiled/
    production/
    archive/
  reports/
```

Keep the following artifacts immutable once created:

- Raw captured frames and clips.
- Human-reviewed annotation files.
- Frozen dataset versions such as `security-v001`.
- Candidate model artifacts.
- Evaluation reports.
- Deployment records.

## 3. Define the detection problem before collecting data

Start with a small, stable class list. The example config currently lists:

```text
Person, Package, Dog, Cat, Car, Truck, Motorcycle, Bicycle, Deer, Bear,
Rabbit, Raccoon, Coyote, Bird
```

Before labeling, decide:

- Which classes are in scope for the first model version.
- Which classes should be ignored until later because they are too rare.
- Whether ambiguous objects should be labeled as `Unknown`, ignored, or mapped to a broader class.
- Whether you need separate labels for delivery vehicles, people, pets, wildlife, or packages.
- Which camera regions should never produce detections, such as roads, trees, timestamp overlays, or neighbor property.

Recommended first version:

- Keep `Person`, `Package`, `Dog`, `Cat`, `Car`, and one or two local wildlife classes.
- Add more rare classes only after you have enough reviewed examples.
- Keep negative images with no target objects; they are important for reducing false positives.

## 4. Define acceptance gates before training

Write the gates down before the first run so you do not promote a model just because it looks better on a few examples.

Use these as initial gates:

| Gate | Initial rule |
| --- | --- |
| Validation mAP50 | Must be no worse than the baseline model, preferably at least 2 percentage points better. |
| Precision | Must not decrease versus baseline or production. |
| Recall | Must not decrease versus baseline or production unless precision improves by an explicitly accepted amount. |
| False positives | Must not increase on validation clips and negative images. |
| Per-class health | No important class can regress badly even if global mAP improves. |
| Per-camera health | No camera can regress badly due to lighting, angle, or distance. |
| Latency | Must meet the Edge TPU latency/FPS target. |
| Compilation | Must convert to full-integer TFLite and compile with the Edge TPU compiler. |
| Rollback | Previous production model must remain available. |

Do not deploy a candidate model if the regression gates fail.

## 5. Bootstrap data from zero

You need real images from the deployment environment before a model can become useful. Synthetic frames and public datasets can help with plumbing and rare classes, but the first production-quality dataset should include your own camera angles, lighting, compression, motion blur, weather, and backgrounds.

### 5.1 Validate the pipeline with synthetic frames

This proves the local package, database, motion detector, tracker, and pipeline wiring work. It does not produce a useful training dataset.

```bash
python -m supercharged_coral.cli smoke-test --frames 100 --database data/synthetic-smoke.db
```

### 5.2 Collect short camera clips

Collect a small but diverse seed corpus:

- Day, dusk, night, and infrared/night-vision frames.
- Clear, rain, snow, fog, shadows, glare, and windy scenes when available.
- Empty-scene negatives.
- Motion events with people, animals, vehicles, packages, and common false-positive objects.
- Multiple distances and object sizes.
- Each camera that will run in production.

If you have RTSP cameras and `ffmpeg` installed, capture one frame per second into the raw-data area:

```bash
mkdir -p data/raw/front-door/$(date +%Y-%m-%d)
export CAMERA_RTSP_URL="rtsp://<username>:<password>@<camera-host>/stream1"
ffmpeg -rtsp_transport tcp \
  -i "$CAMERA_RTSP_URL" \
  -vf fps=1 \
  "data/raw/front-door/$(date +%Y-%m-%d)/%06d.jpg"
```

Use a lower FPS for long captures to avoid thousands of near-duplicate frames. For a first dataset, a better target is diversity, not volume.

### 5.3 Use public data carefully

Public datasets can help bootstrap common classes such as person, car, dog, cat, and bicycle. Only use data with licenses compatible with your intended use, and record the source in dataset metadata. Do not mix public images into the test set that is meant to represent your cameras.

### 5.4 Seed data targets

These are practical starting points, not hard rules:

| Stage | Data target |
| --- | --- |
| Pipeline sanity | Synthetic frames from the smoke test. |
| First label review | 200 to 500 diverse real frames. |
| First fine-tune | 500 to 1,500 reviewed real frames, including negatives. |
| First serious candidate | 2,000+ reviewed real frames with coverage across cameras, lighting, and common classes. |
| Rare classes | Prefer at least 100 reviewed boxes per class before trusting per-class metrics. |

If a class has very few examples, keep it in evaluation reports but do not let it dominate promotion decisions.

## 6. Pre-label with baseline and teacher models

The intended Supercharged Coral loop uses two label sources:

- A fast student detector, such as EfficientDet-Lite, to produce baseline detections.
- A stronger GPU teacher model, such as Grounding DINO, Florence-2, Qwen-VL, InternVL, or Molmo, to produce better labels, boxes, captions, and explanations.

Every teacher adapter should satisfy the `TeacherModel` protocol:

```python
async def label(frame: Frame, regions: Sequence[Region] | None = None) -> TeacherOutput:
    ...
```

Every teacher output should be persisted through `TeacherOutputStore.record_teacher_output` so labels can be audited and reproduced.

Recommended teacher-labeling rules:

- Use the project class list as the prompt vocabulary.
- Ask the teacher for tight bounding boxes, class names, confidence, and explanation.
- Preserve the raw teacher output before human edits.
- Store the teacher model ID and version.
- Track the camera ID, timestamp, frame hash, weather/light metadata when available, and source path.
- Keep low-confidence or conflicting labels for review instead of silently dropping them.

## 7. Review and correct labels

Human review is required for the first useful dataset. Teacher labels are bootstrapping material, not ground truth.

Use a local or self-hosted labeling tool when camera footage is sensitive. Export reviewed labels to COCO detection format and keep COCO as the source-of-truth annotation format for dataset versions.

Labeling standards:

- Draw tight boxes around visible object pixels.
- Label partially occluded objects if enough of the object is visible to identify it.
- Mark severe truncation, blur, glare, or tiny objects in metadata when the tool supports it.
- Ignore objects outside the configured region of interest.
- Add negative images that contain no target object.
- Keep labels consistent across cameras and reviewers.
- Prefer a broad class over a guessed fine-grained class when uncertain.

Do not train directly on unreviewed teacher labels for the first production candidate. After the system matures, unreviewed high-confidence teacher labels can be introduced behind a separate experiment flag and measured against reviewed-only training.

## 8. Build a frozen dataset version

Dataset versions should be reproducible and immutable. For `security-v001`, create train, validation, and test splits that avoid leakage.

Recommended split policy:

- Split by time window or clip, not random adjacent frames.
- Keep frames from the same motion event in only one split.
- Keep the test set untouched after creation.
- Balance cameras across splits when possible.
- Ensure negative frames are present in all splits.
- Preserve class distribution, but prefer leakage prevention over perfect balance.

Recommended split ratio:

```text
70% train
15% validation
15% test
```

Write dataset metadata:

```json
{
  "name": "security",
  "version": "v001",
  "created_by": "manual-review",
  "source_cameras": ["front-door", "driveway"],
  "annotation_format": "coco-detection",
  "classes": ["Person", "Package", "Dog", "Cat", "Car"],
  "split_policy": "by-time-window",
  "notes": "First reviewed local-camera dataset."
}
```

When a concrete `DatasetBuilder` adapter exists, it should return a `DatasetVersion` with the dataset name, version, frame IDs, and metadata.

## 9. Run baseline evaluation

Before fine-tuning, evaluate the base model on `security-v001`. This gives you the comparison point for every candidate.

Baseline artifacts to save:

- Base model ID and source.
- Dataset version.
- Global mAP, mAP50, mAP75, precision, recall, and F1.
- Per-class metrics.
- Per-camera metrics.
- False-positive and false-negative examples.
- Latency and FPS on the target inference hardware when available.
- Evaluation script version or adapter version.

Store the report under:

```text
data/reports/security-v001/baseline/
```

Do not skip this step. Without a baseline, you cannot prove the trained model improved.

## 10. Train or fine-tune the student detector

Supercharged Coral targets EfficientDet-Lite for Edge TPU inference. The first trainer adapter should satisfy the `Trainer` protocol:

```python
def train(request: TrainingRequest) -> ModelArtifact:
    ...
```

The `TrainingRequest` should include:

- Frozen dataset version.
- Base model ID, such as `efficientdet-lite0-coco`.
- Model family, such as `efficientdet-lite`.
- Hyperparameters.
- Dataset, augmentation, and experiment metadata.

Recommended first training strategy:

- Start with EfficientDet-Lite0 or Lite1.
- Fine-tune from a pretrained checkpoint instead of training from scratch.
- Use conservative augmentations first: brightness, contrast, blur, JPEG artifacts, crop/scale, and horizontal flip only if it is valid for your camera geometry.
- Add weather and night simulation after you have a baseline.
- Use early stopping based on validation mAP50 and false positives.
- Save checkpoints and support resume.
- Track every run in TensorBoard or another metrics sink.

Example hyperparameters for a small first run:

```json
{
  "input_size": 320,
  "batch_size": 8,
  "epochs": 50,
  "initial_learning_rate": 0.001,
  "warmup_epochs": 3,
  "early_stopping_patience": 8,
  "mixed_precision": true,
  "augmentations": {
    "brightness": true,
    "contrast": true,
    "jpeg_artifacts": true,
    "motion_blur": true,
    "rain": false,
    "snow": false,
    "fog": false
  }
}
```

Until a project-specific trainer CLI exists, run the backend-specific training command for your chosen trainer and write outputs to:

```text
data/experiments/security-v001/<run-id>/
data/models/candidates/<model-id>/
```

Expected candidate outputs:

```text
data/models/candidates/efficientdet-lite0-security-v001-r001/
  saved_model/
  model.tflite
  checkpoints/
  training_config.json
  model_card.json
```

The future project CLI should wrap this as something equivalent to:

```bash
supercharged-coral train \
  --dataset data/datasets/security-v001 \
  --base-model efficientdet-lite0-coco \
  --model-family efficientdet-lite \
  --output data/models/candidates/efficientdet-lite0-security-v001-r001
```

That command is a target workflow shape and is not currently implemented.

## 11. Evaluate the candidate model

Evaluate every candidate on validation and test data, then compare it to the baseline and current production model.

Required candidate reports:

- Global metrics: mAP, mAP50, mAP75, precision, recall, F1.
- Per-class metrics.
- Per-camera metrics.
- Per-time-of-day metrics when metadata exists.
- Negative-image false positives.
- False negatives for safety-critical classes.
- Latency and FPS on CPU/GPU for development.
- Latency and FPS on Edge TPU after compilation.
- Confusion matrix.
- Example images for the most important false positives and false negatives.

Promotion decision:

```text
candidate passes only if:
  quality gates pass
  latency gates pass
  Edge TPU compilation succeeds
  no important slice regresses
  rollback target is available
```

Store reports under:

```text
data/reports/security-v001/<model-id>/
```

When a concrete evaluator exists, it should return an `EvaluationReport` with `passed_regression_gates` set to `true` only when all gates pass.

## 12. Convert and compile for Edge TPU

Edge TPU deployment requires a fully quantized TFLite model. Use a representative dataset from the training split or a separate calibration subset. Do not use the test set for calibration.

Expected conversion outputs:

```text
data/models/candidates/<model-id>/model.tflite
data/models/candidates/<model-id>/quantization_report.json
```

Compile with the Coral Edge TPU compiler:

```bash
mkdir -p data/models/compiled/<model-id>
edgetpu_compiler \
  data/models/candidates/<model-id>/model.tflite \
  -o data/models/compiled/<model-id>
```

Expected compiled output:

```text
data/models/compiled/<model-id>/model_edgetpu.tflite
```

Compilation failures usually mean unsupported operations, missing full-integer quantization, unsupported tensor shapes, or a model variant that is too large for the Edge TPU. Fix those before deployment.

## 13. Register the model

After successful evaluation and compilation, register the model artifact with:

- Model ID.
- Family and version.
- Source dataset version.
- Candidate artifact path.
- Compiled Edge TPU artifact path.
- Evaluation report path.
- Gate status.
- Training code version.
- Base model ID.
- Hyperparameters.

The current SQLite schema has a `model_registry` table for this purpose. A concrete `ModelRegistry` adapter should write to that table and expose `register`, `get`, `production_model`, and `promote`.

Example model metadata:

```json
{
  "model_id": "efficientdet-lite0-security-v001-r001",
  "family": "efficientdet-lite",
  "version": "v001-r001",
  "base_model_id": "efficientdet-lite0-coco",
  "dataset": "security-v001",
  "candidate_path": "data/models/candidates/efficientdet-lite0-security-v001-r001",
  "compiled_edge_tpu_path": "data/models/compiled/efficientdet-lite0-security-v001-r001/model_edgetpu.tflite",
  "passed_regression_gates": true
}
```

## 14. Deploy safely

Only deploy after registration and passing gates.

Deployment rules:

- Keep the previous production model in `data/models/archive/` or `data/models/production/previous/`.
- Record who or what promoted the model and why.
- Deploy to one camera first when possible.
- Monitor false positives, false negatives, latency, and TPU health.
- Roll back immediately if safety-critical detections regress.

The deployment adapter should satisfy:

```python
def deploy(model: ModelArtifact, reason: str) -> DeploymentRecord:
    ...

def rollback(target_model_id: str, reason: str) -> DeploymentRecord:
    ...
```

## 15. Start the continuous-learning loop

After deployment, collect the mistakes that matter most:

- False positives.
- False negatives.
- Low-confidence detections.
- Teacher/student disagreements.
- New weather, night, glare, or seasonal conditions.
- New camera angles.
- Rare classes.
- Unknown objects in motion regions.

Use active learning to prioritize review, then create the next frozen dataset:

```text
security-v001 -> security-v002 -> security-v003
```

Rules for every new version:

- Never mutate old dataset versions.
- Never train on the test split.
- Keep a stable long-term test set for production comparisons.
- Add new reviewed data to train/validation first.
- Refresh the test set only through an explicit dataset-version decision.
- Compare each model to both the original baseline and current production.

## 16. Operational checklist

Use this checklist for every model candidate.

```text
[ ] Local package installs.
[ ] Tests pass.
[ ] Database initialized.
[ ] Class list frozen for this dataset version.
[ ] Acceptance gates written before training.
[ ] Raw data captured and source metadata recorded.
[ ] Teacher labels generated and stored.
[ ] Human-reviewed labels exported.
[ ] Dataset version built with train/validation/test splits.
[ ] Baseline model evaluated on the dataset.
[ ] Candidate model trained from a known base model.
[ ] Training config and hyperparameters saved.
[ ] Candidate evaluated on validation and test splits.
[ ] Candidate compared against baseline and production.
[ ] Full-integer TFLite conversion succeeds.
[ ] Edge TPU compilation succeeds.
[ ] Edge TPU latency/FPS measured.
[ ] Model registered with report paths.
[ ] Deployment record created.
[ ] Previous production model remains available for rollback.
[ ] Post-deployment monitoring started.
```

## 17. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Model looks good in review but bad in production | Train/test leakage or non-representative validation data | Split by event/time, add production-like validation clips, keep test set untouched. |
| Many false positives on trees, shadows, rain, or headlights | Too few hard negatives | Add reviewed negative frames and hard false positives to the next dataset version. |
| Good global mAP but one camera is bad | Camera-specific lighting or angle gap | Add per-camera slice metrics and collect more data for that camera. |
| Rare class metrics are unstable | Too few examples | Do not optimize for that class yet; collect and review more examples. |
| Training overfits quickly | Dataset too small or too duplicated | Deduplicate adjacent frames, add diversity, reduce epochs, and increase augmentation carefully. |
| Edge TPU compiler fails | Unsupported ops or incomplete quantization | Use an Edge TPU-compatible EfficientDet-Lite variant and full-integer quantization. |
| Edge TPU latency is too high | Model too large or input too high resolution | Try Lite0, lower input size, crop motion regions, or reduce post-processing overhead. |
| Teacher labels are inconsistent | Prompt or class taxonomy is ambiguous | Tighten class definitions and require human review before training. |

## 18. First implementation milestones

To make this runbook executable inside the project, implement these adapters behind the existing protocols:

1. `DatasetBuilder`: import reviewed COCO annotations, de-duplicate frames, write dataset metadata, and return `DatasetVersion`.
2. `TeacherModel` and `TeacherOutputStore`: run a selected local teacher model and persist outputs.
3. `Trainer`: fine-tune EfficientDet-Lite from a frozen dataset and return `ModelArtifact`.
4. `Evaluator`: compute quality, slice, latency, and regression-gate reports.
5. `ModelCompiler`: convert to TFLite and compile with `edgetpu_compiler`.
6. `ModelRegistry`: persist model artifacts and promotion state in SQLite.
7. `DeploymentStrategy`: atomically switch production model paths and support rollback.

The most useful first vertical slice is:

```text
reviewed COCO dataset
  -> EfficientDet-Lite0 fine-tune
  -> validation report
  -> TFLite export
  -> Edge TPU compile
  -> SQLite model registry entry
```
