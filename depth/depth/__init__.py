"""DA3 depth estimation for a single image or a whole video."""

__version__ = "0.1.0"

from depth.device import resolve_torch_device, set_cuda_device_if_indexed

__all__ = [
    "resolve_torch_device",
    "set_cuda_device_if_indexed",
]
