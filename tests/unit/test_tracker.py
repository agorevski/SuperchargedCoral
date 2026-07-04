from supercharged_coral.common.events import Detection, Region
from supercharged_coral.tracker.sort_like import SortLikeTracker


def test_sort_like_tracker_preserves_id_for_overlapping_detections():
    tracker = SortLikeTracker(iou_threshold=0.1)

    first = tracker.update(
        [Detection("Person", 0.9, Region(10, 10, 20, 20), model_id="detector")]
    )
    second = tracker.update(
        [Detection("Person", 0.8, Region(12, 12, 20, 20), model_id="detector")]
    )

    assert first[0].track_id == second[0].track_id
    assert second[0].hits == 2

