"""Small IoU-based tracking baseline inspired by SORT."""

from __future__ import annotations

from dataclasses import dataclass

from supercharged_coral.common.events import Detection, Track

__all__ = ["SortLikeTracker"]


@dataclass
class _TrackState:
    track: Track
    missed: int = 0


class SortLikeTracker:
    """Small IoU-based tracker baseline used until ByteTrack/DeepSORT integrations land."""

    def __init__(self, *, iou_threshold: float = 0.3, max_age: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_track_ids = set(self._tracks)
        output: list[Track] = []

        for detection in detections:
            best_id: int | None = None
            best_iou = 0.0
            for track_id in list(unmatched_track_ids):
                state = self._tracks[track_id]
                if state.track.class_name != detection.class_name:
                    continue
                score = state.track.bbox.iou(detection.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_id is not None and best_iou >= self.iou_threshold:
                previous = self._tracks[best_id].track
                track = Track(
                    track_id=best_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    age=previous.age + 1,
                    hits=previous.hits + 1,
                    metadata={"matched_iou": best_iou},
                )
                self._tracks[best_id] = _TrackState(track=track)
                unmatched_track_ids.remove(best_id)
            else:
                track = Track(
                    track_id=self._next_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    age=1,
                    hits=1,
                )
                self._tracks[self._next_id] = _TrackState(track=track)
                self._next_id += 1
            output.append(track)

        for track_id in unmatched_track_ids:
            state = self._tracks[track_id]
            state.missed += 1
            if state.missed > self.max_age:
                del self._tracks[track_id]

        return output
