"""Google Coral Edge TPU runtime interfaces and discovery scaffolding."""

from supercharged_coral.edge_tpu.runtime import CoralDevice, EdgeTPURuntime
from supercharged_coral.interfaces import EdgeTPUBackend

__all__ = ["CoralDevice", "EdgeTPUBackend", "EdgeTPURuntime"]
