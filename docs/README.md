# Supercharged Coral Documentation

This directory contains the product and architecture documents for Supercharged Coral. Start with the top-level [`README.md`](../README.md) for current implementation status, then use the documents below for deeper planning and extension guidance.

## Documentation map

| Document | Audience | Use it for |
| --- | --- | --- |
| [`../README.md`](../README.md) | Contributors and evaluators | Project overview, quick start, architecture summary, and repository layout. |
| [`PROJECT_SPEC.md`](PROJECT_SPEC.md) | Product, architecture, and implementation owners | Full product requirements, target capabilities, and roadmap-level acceptance criteria. |
| [`INTERFACES.md`](INTERFACES.md) | Engineers building integrations | Protocol contracts, plug-in boundaries, and examples for replacing runtime components. |
| [`../RUNBOOK.md`](../RUNBOOK.md) | Operators and training pipeline implementers | End-to-end model training workflow from zero data through evaluation, Edge TPU compilation, and deployment. |

## Recommended reading order

1. [`../README.md`](../README.md) — understand what exists today and how the repository is organized.
2. [`PROJECT_SPEC.md`](PROJECT_SPEC.md) — review the long-term target architecture and product requirements.
3. [`INTERFACES.md`](INTERFACES.md) — implement or replace components behind stable protocol boundaries.
4. [`../RUNBOOK.md`](../RUNBOOK.md) — operate the full training workflow once data and trainer adapters are available.

## Documentation standards

- Keep links relative so they render correctly in GitHub and local editors.
- Separate current implementation status from future product requirements.
- Update the interface guide whenever a protocol contract, shared event type, or injection pattern changes.
- Keep the top-level README concise; put detailed roadmap material in the project specification.
