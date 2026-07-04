"""SQLite persistence helpers for frames, detections, tracks, and datasets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from supercharged_coral.common.events import Detection, Frame, Track

__all__ = ["SecurityDatabase", "hash_frame"]


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    image_hash TEXT NOT NULL UNIQUE,
    image_path TEXT,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    lighting_estimate REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_frames_camera_timestamp ON frames(camera_id, timestamp);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_detections_frame_id ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_class_name ON detections(class_name);

CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    age INTEGER NOT NULL,
    hits INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tracks_track_id ON tracks(track_id);

CREATE TABLE IF NOT EXISTS teacher_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    teacher_model TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    boxes_json TEXT NOT NULL,
    confidence REAL,
    caption TEXT,
    explanation TEXT,
    segmentation_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS dataset_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES training_datasets(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    split TEXT NOT NULL CHECK(split IN ('train', 'validation', 'test')),
    priority REAL NOT NULL DEFAULT 0.0,
    reason TEXT,
    UNIQUE(dataset_id, frame_id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model_family TEXT NOT NULL,
    base_model_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    model_id TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    slice_name TEXT NOT NULL DEFAULT 'global',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL UNIQUE,
    family TEXT NOT NULL,
    version TEXT NOT NULL,
    path TEXT NOT NULL,
    compiled_edge_tpu_path TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS deployment_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    deployed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deployed_by TEXT NOT NULL,
    previous_model_id TEXT,
    reason TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""


def hash_frame(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json(value: dict[str, Any] | list[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class SecurityDatabase:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

    def initialize(self) -> None:
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def record_frame(
        self,
        frame: Frame,
        *,
        image_path: str | None = None,
        lighting_estimate: float | None = None,
    ) -> int:
        image_hash = hash_frame(frame.image)
        self.connection.execute(
            """
            INSERT OR IGNORE INTO frames (
                camera_id, sequence, timestamp, image_hash, image_path, width, height,
                lighting_estimate, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame.camera_id,
                frame.sequence,
                frame.timestamp.isoformat(),
                image_hash,
                image_path,
                frame.width,
                frame.height,
                lighting_estimate,
                _json(frame.metadata),
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM frames WHERE image_hash = ?",
            (image_hash,),
        ).fetchone()
        self.connection.commit()
        return int(row["id"])

    def record_detection(self, frame_id: int, detection: Detection) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO detections (
                frame_id, class_name, confidence, x, y, width, height, model_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                detection.class_name,
                detection.confidence,
                detection.bbox.x,
                detection.bbox.y,
                detection.bbox.width,
                detection.bbox.height,
                detection.model_id,
                _json(detection.metadata),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_track(self, frame_id: int, track: Track) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO tracks (
                frame_id, track_id, class_name, confidence, x, y, width, height, age, hits,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                track.track_id,
                track.class_name,
                track.confidence,
                track.bbox.x,
                track.bbox.y,
                track.bbox.width,
                track.bbox.height,
                track.age,
                track.hits,
                _json(track.metadata),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_teacher_output(
        self,
        frame_id: int,
        *,
        teacher_model: str,
        labels: list[str],
        boxes: list[dict[str, Any]],
        confidence: float | None = None,
        caption: str | None = None,
        explanation: str | None = None,
        segmentation_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO teacher_outputs (
                frame_id, teacher_model, labels_json, boxes_json, confidence, caption,
                explanation, segmentation_path, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                teacher_model,
                _json(labels),
                _json(boxes),
                confidence,
                caption,
                explanation,
                segmentation_path,
                _json(metadata or {}),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def scalar(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        row = self.connection.execute(query, params).fetchone()
        return None if row is None else row[0]
