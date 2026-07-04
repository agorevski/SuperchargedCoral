import numpy as np

from supercharged_coral.common.events import Detection, Frame, Region, Track
from supercharged_coral.dataset.database import SecurityDatabase


def test_database_records_frames_detections_tracks_and_teacher_outputs():
    database = SecurityDatabase(":memory:")
    database.initialize()
    frame = Frame(camera_id="cam", sequence=1, image=np.zeros((10, 20, 3), dtype=np.uint8))

    frame_id = database.record_frame(frame)
    duplicate_frame_id = database.record_frame(frame)
    detection_id = database.record_detection(
        frame_id,
        Detection(
            class_name="Person",
            confidence=0.9,
            bbox=Region(1, 2, 3, 4),
            model_id="test-model",
        ),
    )
    track_id = database.record_track(
        frame_id,
        Track(
            track_id=7,
            class_name="Person",
            confidence=0.9,
            bbox=Region(1, 2, 3, 4),
            age=1,
            hits=1,
        ),
    )
    teacher_output_id = database.record_teacher_output(
        frame_id,
        teacher_model="teacher",
        labels=["Person"],
        boxes=[{"x": 1, "y": 2, "width": 3, "height": 4}],
        confidence=0.95,
        caption="A person at the door.",
    )

    assert frame_id == duplicate_frame_id
    assert detection_id == 1
    assert track_id == 1
    assert teacher_output_id == 1
    assert database.scalar("SELECT COUNT(*) FROM frames") == 1
    assert database.scalar("SELECT COUNT(*) FROM detections") == 1
    database.close()


def test_database_creates_parent_directory(tmp_path):
    database_path = tmp_path / "nested" / "security.db"
    database = SecurityDatabase(database_path)
    database.initialize()

    assert database_path.exists()
    database.close()
