# Copilot instructions for Supercharged Coral

Supercharged Coral is an interface-first Python 3.12+ package for a self-hosted home security AI platform optimized for Google Coral Edge TPU inference and GPU-assisted continuous learning.

## Repository priorities

- Preserve the interface-first architecture. Major runtime, training, persistence, hardware, telemetry, deployment, and dashboard components should sit behind structural `typing.Protocol` contracts in `src/supercharged_coral/interfaces.py`.
- Depend on shared datatypes from `src/supercharged_coral/common/events.py` at service boundaries. Prefer immutable dataclasses for event/data-transfer records when adding new shared records.
- Inject concrete implementations through orchestration code instead of importing sibling service implementations directly. Test doubles should be ordinary Python objects that satisfy the same protocols.
- Keep hardware-specific, model-specific, storage-specific, and dashboard-specific behavior behind replaceable adapters. Avoid hard dependencies on Coral, CUDA, RTSP cameras, or external services in core tests.
- Keep current implementation docs separate from roadmap/product requirements. Update `README.md`, `docs/INTERFACES.md`, `docs/PROJECT_SPEC.md`, or `RUNBOOK.md` when changing behavior documented there.

## Project layout

- `src/supercharged_coral/cli.py` contains CLI entry points for local smoke tests and database initialization.
- `src/supercharged_coral/interfaces.py` defines the main protocol contracts.
- `src/supercharged_coral/common/` contains configuration, events, and shared datatypes.
- `src/supercharged_coral/camera_service/`, `motion_service/`, `object_detector/`, `tracker/`, `dataset/`, `edge_tpu/`, `pipeline/`, `teacher/`, `training/`, `evaluation/`, `deployment/`, and `dashboard/` contain replaceable component seams and baseline implementations.
- `tests/unit/` and `tests/integration/` cover implemented foundations.
- `config/example.yaml` is the example platform configuration.

## Python conventions

- Use modern typed Python with `from __future__ import annotations`.
- Prefer explicit dataclasses, protocols, and small focused classes over framework-heavy abstractions.
- Keep async interfaces async where the contract already uses async methods, especially camera sources, object detectors, teacher models, and the frame pipeline.
- Avoid broad exception swallowing and silent fallbacks. Surface errors unless an existing pattern intentionally handles them.
- Keep local smoke-test paths runnable without real cameras, Coral hardware, GPUs, or external services.

## Development commands

Install for development:

```bash
python3 -m pip install -e ".[dev]"
```

Run tests:

```bash
python3 -m pytest
```

Run the local synthetic pipeline smoke test:

```bash
python3 -m supercharged_coral.cli smoke-test --frames 5
```

Initialize a SQLite database:

```bash
python3 -m supercharged_coral.cli init-db --database supercharged-coral.db
```

Install optional extras only when working on those areas:

```bash
python3 -m pip install -e ".[edge-tpu]"
python3 -m pip install -e ".[training]"
python3 -m pip install -e ".[dashboard]"
```

## Testing guidance

- Add or update targeted pytest coverage for changed behavior.
- Prefer synthetic cameras, in-memory SQLite databases, mock hardware runtimes, and simple protocol-satisfying test doubles.
- Do not require real RTSP cameras, Coral devices, CUDA GPUs, cloud services, or dashboard servers for default tests.
- For pipeline changes, verify frame, motion, detection, tracking, and persistence stats remain consistent.

## Documentation guidance

- Keep relative links in Markdown.
- Update `docs/INTERFACES.md` whenever protocol contracts, shared event types, or injection patterns change.
- Update `RUNBOOK.md` when changing the training/evaluation/deployment workflow.
- Keep the top-level `README.md` concise and focused on current implementation status and quick start.
