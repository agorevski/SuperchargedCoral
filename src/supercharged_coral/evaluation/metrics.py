"""Small metric primitives used by evaluation adapters."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BinaryClassificationCounts", "precision_recall_f1"]


@dataclass(frozen=True)
class BinaryClassificationCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0


def precision_recall_f1(counts: BinaryClassificationCounts) -> dict[str, float]:
    precision_denominator = counts.true_positives + counts.false_positives
    recall_denominator = counts.true_positives + counts.false_negatives
    precision = 0.0 if precision_denominator == 0 else counts.true_positives / precision_denominator
    recall = 0.0 if recall_denominator == 0 else counts.true_positives / recall_denominator
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
