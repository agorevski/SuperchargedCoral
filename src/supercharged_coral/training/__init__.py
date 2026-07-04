"""Training, data-quality, distillation, and continuous-learning protocols."""

from supercharged_coral.interfaces import (
    ActiveLearningSampler,
    DataAugmenter,
    DatasetBuilder,
    Distiller,
    FrameQualityFilter,
    Trainer,
)

__all__ = [
    "ActiveLearningSampler",
    "DataAugmenter",
    "DatasetBuilder",
    "Distiller",
    "FrameQualityFilter",
    "Trainer",
]
