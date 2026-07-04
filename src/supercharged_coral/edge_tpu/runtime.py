"""Optional Coral Edge TPU device discovery and health reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["CoralDevice", "EdgeTPURuntime"]


@dataclass(frozen=True)
class CoralDevice:
    device_type: str
    path: str
    healthy: bool
    metadata: dict[str, str]


class EdgeTPURuntime:
    """Discovery and health hooks for Coral devices.

    PyCoral is optional so the platform can run tests and CPU-only development workflows
    without a connected accelerator.
    """

    def list_devices(self) -> list[CoralDevice]:
        devices: list[CoralDevice] = []
        try:
            from pycoral.utils.edgetpu import list_edge_tpus
        except ImportError:
            list_edge_tpus = None

        if list_edge_tpus is not None:
            for device in list_edge_tpus():
                devices.append(
                    CoralDevice(
                        device_type=str(device.get("type", "unknown")),
                        path=str(device.get("path", "")),
                        healthy=True,
                        metadata={str(key): str(value) for key, value in device.items()},
                    )
                )

        for apex in sorted(Path("/dev").glob("apex_*")):
            devices.append(
                CoralDevice(
                    device_type="pci",
                    path=str(apex),
                    healthy=apex.exists(),
                    metadata={"source": "devfs"},
                )
            )

        return devices

    def health(self) -> dict[str, int | bool]:
        devices = self.list_devices()
        return {"healthy": bool(devices), "device_count": len(devices)}
