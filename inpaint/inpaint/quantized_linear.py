"""Comfy-kitchen backed FP8 and NVFP4 linear layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

QUANT_FORMAT_LAYOUTS = {
    "float8_e4m3fn": "TensorCoreFP8Layout",
    "nvfp4": "TensorCoreNVFP4Layout",
}


def _quantized_types() -> tuple[Any, Any, Any]:
    try:
        from comfy_kitchen.tensor import (
            QuantizedTensor,
            TensorCoreFP8Layout,
            TensorCoreNVFP4Layout,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Loading a ComfyUI FP8/NVFP4 checkpoint requires comfy-kitchen. "
            "Run `uv sync` in 02_inpaint."
        ) from exc
    return QuantizedTensor, TensorCoreFP8Layout, TensorCoreNVFP4Layout


def validate_nvfp4_runtime(device: torch.device) -> dict[str, Any]:
    """Validate the Blackwell/CUDA runtime required by the native NVFP4 kernel."""
    if device.type != "cuda":
        raise RuntimeError("NVFP4 inference requires an NVIDIA CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    cuda_version = tuple(int(part) for part in (torch.version.cuda or "0").split("."))
    if cuda_version < (12, 8):
        raise RuntimeError(
            "Native NVFP4 inference requires PyTorch built with CUDA >= 12.8; "
            f"the current build is torch {torch.__version__} / CUDA {torch.version.cuda}."
        )

    capability = torch.cuda.get_device_capability(device)
    if capability < (10, 0):
        raise RuntimeError(
            "Native NVFP4 inference requires a Blackwell GPU (SM >= 10.0); "
            f"the selected device is SM {capability[0]}.{capability[1]}."
        )

    import comfy_kitchen as ck

    backends = ck.list_backends()
    cuda_backend = backends.get("cuda", {})
    capabilities = cuda_backend.get("capabilities", [])
    if (
        not cuda_backend.get("available", False)
        or cuda_backend.get("disabled", False)
        or "scaled_mm_nvfp4" not in capabilities
    ):
        raise RuntimeError(
            "comfy-kitchen's CUDA backend is unavailable. Install a binary wheel "
            "compatible with Python and the selected CUDA runtime."
        )
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": capability,
        "backends": backends,
    }


class QuantizedLinear(nn.Module):
    """Inference-only Linear that keeps ComfyUI weights in quantized storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "meta",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.quant_format: str | None = None
        self.layout_type: str | None = None
        self.register_parameter("weight", None)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=compute_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer("input_scale", None, persistent=False)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> QuantizedLinear:
        return cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            compute_dtype=torch.bfloat16,
            device="meta",
        )

    def set_quantized_weight(
        self,
        quant_format: str,
        qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor | None,
        input_scale: torch.Tensor,
    ) -> None:
        QuantizedTensor, FP8Layout, NVFP4Layout = _quantized_types()
        orig_shape = (self.out_features, self.in_features)

        if quant_format == "nvfp4":
            if weight_scale_2 is None:
                raise ValueError("NVFP4 weight is missing weight_scale_2.")
            layout_cls = NVFP4Layout
            params = layout_cls.Params(
                scale=weight_scale_2.float(),
                block_scale=weight_scale,
                orig_dtype=self.compute_dtype,
                orig_shape=orig_shape,
            )
            storage_dtype = torch.uint8
        elif quant_format == "float8_e4m3fn":
            layout_cls = FP8Layout
            params = layout_cls.Params(
                scale=weight_scale.float(),
                orig_dtype=self.compute_dtype,
                orig_shape=orig_shape,
            )
            storage_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(f"Unsupported quantization format: {quant_format}")

        self.quant_format = quant_format
        self.layout_type = QUANT_FORMAT_LAYOUTS[quant_format]
        qweight = QuantizedTensor(
            qdata.to(dtype=storage_dtype),
            self.layout_type,
            params,
        )
        self.weight = nn.Parameter(qweight, requires_grad=False)
        self.input_scale = input_scale.float()

    def _apply(self, fn, recurse: bool = True):
        """Move QuantizedTensor storage and scales without materializing BF16."""
        if recurse:
            for child in self.children():
                child._apply(fn)
        for name, parameter in self._parameters.items():
            if parameter is None:
                continue
            moved = fn(parameter)
            self.register_parameter(name, nn.Parameter(moved, requires_grad=False))
        for name, buffer in self._buffers.items():
            if buffer is not None:
                self._buffers[name] = fn(buffer)
        return self

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.layout_type is None or self.input_scale is None:
            raise RuntimeError("QuantizedLinear has not been loaded.")

        QuantizedTensor, _, _ = _quantized_types()
        input_shape = input_tensor.shape
        flat_input = input_tensor.flatten(0, -2).contiguous()
        quantized_input = QuantizedTensor.from_float(
            flat_input,
            self.layout_type,
            scale=self.input_scale,
        )
        output = F.linear(quantized_input, self.weight, self.bias)
        if input_tensor.ndim > 2:
            output = output.unflatten(0, input_shape[:-1])
        return output
