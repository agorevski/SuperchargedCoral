"""Camera source providers for synthetic, local, and stream-backed inputs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from supercharged_coral.common.config import CameraConfig
from supercharged_coral.common.events import Frame
from supercharged_coral.interfaces import CameraSource

__all__ = [
    "OpenCVCameraSource",
    "SyntheticCameraSource",
    "SyntheticScene",
    "create_camera_source",
]


@dataclass
class SyntheticScene:
    width: int = 640
    height: int = 360
    object_size: int = 48
    velocity_px: int = 7

    def render(self, sequence: int) -> np.ndarray:
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        max_x = max(1, self.width - self.object_size)
        x = (sequence * self.velocity_px) % max_x
        y = self.height // 2 - self.object_size // 2
        image[y : y + self.object_size, x : x + self.object_size] = (255, 255, 255)
        return image


class SyntheticCameraSource:
    """Deterministic moving-object simulator used by tests and local development."""

    def __init__(self, config: CameraConfig, *, frame_limit: int | None = None) -> None:
        self.config = config
        self.scene = SyntheticScene(
            width=config.width or 640,
            height=config.height or 360,
        )
        self.frame_limit = frame_limit
        self._sequence = 0
        self._closed = False

    @property
    def camera_id(self) -> str:
        return self.config.id

    async def frames(self) -> AsyncIterator[Frame]:
        interval = 1.0 / self.config.fps
        while not self._closed:
            if self.frame_limit is not None and self._sequence >= self.frame_limit:
                break
            yield await self.snapshot()
            await asyncio.sleep(interval)

    async def snapshot(self) -> Frame:
        frame = Frame(
            camera_id=self.config.id,
            image=self.scene.render(self._sequence),
            timestamp=datetime.now(timezone.utc),
            sequence=self._sequence,
            metadata={"source_type": "synthetic"},
        )
        self._sequence += 1
        return frame

    async def close(self) -> None:
        self._closed = True


class OpenCVCameraSource:
    """OpenCV-backed source for RTSP, ONVIF stream URLs, USB cameras, and clips."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: Any | None = None
        self._sequence = 0
        self._closed = False
        self._dropped_frames = 0

    @property
    def camera_id(self) -> str:
        return self.config.id

    def _source(self) -> str | int:
        if self.config.source_type == "usb":
            return int(self.config.uri or 0)
        if not self.config.uri:
            raise ValueError(f"Camera {self.config.id} requires a URI")
        return self.config.uri

    def _open(self) -> None:
        import cv2

        self._capture = cv2.VideoCapture(self._source())
        if self.config.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps:
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera source {self.config.id}")

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    async def _reconnect(self) -> bool:
        self._release()
        for attempt in range(self.config.reconnect_attempts + 1):
            try:
                self._open()
                self._dropped_frames = 0
                return True
            except RuntimeError:
                if attempt >= self.config.reconnect_attempts:
                    return False
                await asyncio.sleep(min(5.0, 0.25 * (attempt + 1)))
        return False

    async def frames(self) -> AsyncIterator[Frame]:
        if self._capture is None:
            self._open()
        interval = 1.0 / self.config.fps
        while not self._closed:
            try:
                frame = await self.snapshot()
            except RuntimeError as exc:
                if "dropped frame" not in str(exc):
                    raise
                if self.config.source_type in {"video_file", "clip"}:
                    break
                await asyncio.sleep(interval)
                continue
            yield frame
            await asyncio.sleep(interval)

    async def snapshot(self) -> Frame:
        if self._capture is None:
            self._open()
        assert self._capture is not None
        ok, image = self._capture.read()
        if not ok:
            self._dropped_frames += 1
            if self._dropped_frames > self.config.dropped_frame_tolerance:
                reconnected = await self._reconnect()
                if not reconnected:
                    raise RuntimeError(f"Camera {self.config.id} exceeded dropped-frame tolerance")
            raise RuntimeError(f"Camera {self.config.id} dropped frame {self._dropped_frames}")
        self._dropped_frames = 0
        frame = Frame(
            camera_id=self.config.id,
            image=image,
            timestamp=datetime.now(timezone.utc),
            sequence=self._sequence,
            metadata={"source_type": self.config.source_type},
        )
        self._sequence += 1
        return frame

    async def close(self) -> None:
        self._closed = True
        self._release()


def create_camera_source(config: CameraConfig) -> CameraSource:
    if config.source_type == "synthetic":
        return SyntheticCameraSource(config)
    return OpenCVCameraSource(config)
