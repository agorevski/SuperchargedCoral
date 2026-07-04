"""Configuration objects and mapping loaders for platform settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from supercharged_coral.common.events import Region

__all__ = [
    "CAMERA_SOURCE_TYPES",
    "DEFAULT_OBJECT_CLASSES",
    "MOTION_ALGORITHMS",
    "CameraConfig",
    "MotionConfig",
    "PlatformConfig",
    "camera_config_from_mapping",
    "load_config",
    "motion_config_from_mapping",
    "platform_config_from_mapping",
]


DEFAULT_OBJECT_CLASSES = [
    "Person",
    "Package",
    "Dog",
    "Cat",
    "Car",
    "Truck",
    "Motorcycle",
    "Bicycle",
    "Deer",
    "Bear",
    "Rabbit",
    "Raccoon",
    "Coyote",
    "Bird",
]

CAMERA_SOURCE_TYPES = {"rtsp", "onvif", "usb", "video_file", "clip", "synthetic"}
MOTION_ALGORITHMS = {"frame_differencing", "running_gaussian", "mog2", "knn"}


@dataclass(frozen=True)
class CameraConfig:
    id: str
    source_type: str
    uri: str | None = None
    fps: float = 5.0
    width: int | None = None
    height: int | None = None
    reconnect_attempts: int = 5
    dropped_frame_tolerance: int = 30
    ignore_regions: list[Region] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Camera id is required")
        if self.source_type not in CAMERA_SOURCE_TYPES:
            raise ValueError(f"Unsupported camera source type: {self.source_type}")
        if self.fps <= 0:
            raise ValueError("Camera FPS must be greater than zero")
        if self.reconnect_attempts < 0:
            raise ValueError("Reconnect attempts cannot be negative")
        if self.dropped_frame_tolerance < 0:
            raise ValueError("Dropped frame tolerance cannot be negative")


@dataclass(frozen=True)
class MotionConfig:
    algorithm: str = "frame_differencing"
    threshold: int | None = 30
    min_area: int = 500
    blur_kernel: int = 5
    morphology_kernel: int = 5
    running_alpha: float = 0.05
    ignore_regions: list[Region] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.algorithm not in MOTION_ALGORITHMS:
            raise ValueError(f"Unsupported motion algorithm: {self.algorithm}")
        if self.threshold is not None and not 0 <= self.threshold <= 255:
            raise ValueError("Motion threshold must be between 0 and 255")
        if self.min_area < 0:
            raise ValueError("Motion min_area cannot be negative")
        if self.running_alpha <= 0 or self.running_alpha > 1:
            raise ValueError("running_alpha must be in the range (0, 1]")


@dataclass(frozen=True)
class PlatformConfig:
    cameras: list[CameraConfig]
    database_path: str = "data/supercharged-coral.db"
    object_classes: list[str] = field(default_factory=lambda: list(DEFAULT_OBJECT_CLASSES))
    motion: MotionConfig = field(default_factory=MotionConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("At least one camera must be configured")
        if len(set(self.object_classes)) != len(self.object_classes):
            raise ValueError("Object classes must be unique")
        if len({camera.id for camera in self.cameras}) != len(self.cameras):
            raise ValueError("Camera ids must be unique")


def _coerce_region(value: Region | dict[str, Any]) -> Region:
    if isinstance(value, Region):
        return value
    return Region(
        x=int(value["x"]),
        y=int(value["y"]),
        width=int(value["width"]),
        height=int(value["height"]),
    )


def camera_config_from_mapping(value: dict[str, Any]) -> CameraConfig:
    data = dict(value)
    data["ignore_regions"] = [_coerce_region(region) for region in data.get("ignore_regions", [])]
    return CameraConfig(**data)


def motion_config_from_mapping(value: dict[str, Any] | None) -> MotionConfig:
    data = dict(value or {})
    data["ignore_regions"] = [_coerce_region(region) for region in data.get("ignore_regions", [])]
    return MotionConfig(**data)


def platform_config_from_mapping(value: dict[str, Any]) -> PlatformConfig:
    cameras = [camera_config_from_mapping(camera) for camera in value.get("cameras", [])]
    return PlatformConfig(
        cameras=cameras,
        database_path=value.get("database_path", "data/supercharged-coral.db"),
        object_classes=list(value.get("object_classes", DEFAULT_OBJECT_CLASSES)),
        motion=motion_config_from_mapping(value.get("motion")),
        metadata=dict(value.get("metadata", {})),
    )


def load_config(path: str | Path) -> PlatformConfig:
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
            raise RuntimeError("YAML config files require PyYAML to be installed") from exc
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Platform config must be a mapping")
    return platform_config_from_mapping(data)
