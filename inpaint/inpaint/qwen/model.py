"""Qwen-Image-Edit loading for diffusers and ComfyUI quantized checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors import safe_open

from inpaint.quantized_linear import (
    QUANT_FORMAT_LAYOUTS,
    QuantizedLinear,
    validate_nvfp4_runtime,
)


class QwenEditModelLoader:
    """Build a Qwen pipeline without materializing quantized weights as BF16."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        base_model_path: Path | None = None,
        cpu_offload: bool = True,
    ) -> None:
        self.model_path = model_path.expanduser()
        self.device = device
        self.base_model_path = (
            base_model_path.expanduser() if base_model_path is not None else None
        )
        self.cpu_offload = cpu_offload
        self._single_file = self.model_path.is_file()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Inpaint model not found: {self.model_path}")
        if self._single_file and self.base_model_path is None:
            raise ValueError(
                "Single-file checkpoint requires --qwen.base-model-path for the "
                "pipeline config, text encoder, processor, scheduler, and VAE."
            )
        if self._single_file and not self.base_model_path.exists():
            raise FileNotFoundError(
                f"Base model path not found: {self.base_model_path}"
            )

    @property
    def checkpoint_name(self) -> str:
        return self.model_path.name

    @staticmethod
    def _resolve_weight_file(index_json: Path) -> str | None:
        try:
            data = json.loads(index_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        shards = sorted(set(data.get("weight_map", {}).values()))
        return shards[0] if shards else None

    def _check_diffusers_model_ready(self, model_dir: Path) -> None:
        transformer_dir = model_dir / "transformer"
        index_json = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
        first_shard = self._resolve_weight_file(index_json)
        if first_shard is not None and not (transformer_dir / first_shard).exists():
            raise FileNotFoundError(
                "Diffusers transformer weights are missing: "
                f"{transformer_dir / first_shard}. Use the NVFP4 single file or "
                "download every transformer shard."
            )

    def _read_quantization_metadata(self) -> dict[str, dict[str, Any]]:
        with safe_open(self.model_path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
        raw = metadata.get("_quantization_metadata")
        if raw is None:
            raise ValueError(
                f"Single-file checkpoint has no _quantization_metadata: {self.model_path}"
            )
        parsed = json.loads(raw)
        layers = parsed.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError("Quantization metadata contains no layer definitions.")
        unsupported = {
            conf.get("format")
            for conf in layers.values()
            if conf.get("format") not in QUANT_FORMAT_LAYOUTS
        }
        if unsupported:
            raise ValueError(f"Unsupported quantization formats: {sorted(unsupported)}")
        return layers

    @staticmethod
    def _replace_quantized_linears(
        transformer: torch.nn.Module,
        layers: dict[str, dict[str, Any]],
    ) -> dict[str, QuantizedLinear]:
        replacements: dict[str, QuantizedLinear] = {}
        for layer_name in layers:
            old_module = transformer.get_submodule(layer_name)
            if not isinstance(old_module, torch.nn.Linear):
                raise TypeError(
                    f"Quantized layer {layer_name} is {type(old_module).__name__}, "
                    "expected torch.nn.Linear."
                )
            parent_name, _, child_name = layer_name.rpartition(".")
            parent = (
                transformer.get_submodule(parent_name) if parent_name else transformer
            )
            replacement = QuantizedLinear.from_linear(old_module)
            parent._modules[child_name] = replacement
            replacements[layer_name] = replacement
        return replacements

    def _load_quantized_transformer(self) -> torch.nn.Module:
        from accelerate import init_empty_weights
        from diffusers import QwenImageTransformer2DModel

        runtime = validate_nvfp4_runtime(self.device)
        logger.info(
            "[Qwen] NVFP4 runtime: torch={}, CUDA={}, GPU={}, backends={}",
            runtime["torch"],
            runtime["cuda"],
            runtime["device"],
            "cuda/scaled_mm_nvfp4",
        )

        layers = self._read_quantization_metadata()
        config_path = self.base_model_path / "transformer" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # Runtime RoPE caches are non-persistent buffers and are not in the checkpoint.
        with init_empty_weights(include_buffers=False):
            transformer = QwenImageTransformer2DModel.from_config(config)
        replacements = self._replace_quantized_linears(transformer, layers)

        load_device = torch.device("cpu") if self.cpu_offload else self.device
        regular_state: dict[str, torch.Tensor] = {}
        auxiliary_suffixes = (".weight_scale", ".weight_scale_2", ".input_scale")
        quantized_weight_keys = {f"{name}.weight" for name in layers}

        logger.info(
            "[Qwen] Loading {} quantized layers from {} to {}",
            len(layers),
            self.model_path,
            load_device,
        )
        with safe_open(
            self.model_path,
            framework="pt",
            device=str(load_device),
        ) as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            for key in checkpoint_keys:
                if (
                    key.startswith("__")
                    or key in quantized_weight_keys
                    or key.endswith(auxiliary_suffixes)
                ):
                    continue
                regular_state[key] = checkpoint.get_tensor(key)

            incompatible = transformer.load_state_dict(
                regular_state,
                strict=False,
                assign=True,
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected transformer keys: {incompatible.unexpected_keys[:20]}"
                )

            for layer_name, module in replacements.items():
                quant_format = layers[layer_name]["format"]
                prefix = f"{layer_name}."
                required = ["weight", "weight_scale", "input_scale"]
                if quant_format == "nvfp4":
                    required.append("weight_scale_2")
                missing = [
                    name for name in required if prefix + name not in checkpoint_keys
                ]
                if missing:
                    raise ValueError(f"{layer_name} is missing tensors: {missing}")
                module.set_quantized_weight(
                    quant_format=quant_format,
                    qdata=checkpoint.get_tensor(prefix + "weight"),
                    weight_scale=checkpoint.get_tensor(prefix + "weight_scale"),
                    weight_scale_2=(
                        checkpoint.get_tensor(prefix + "weight_scale_2")
                        if quant_format == "nvfp4"
                        else None
                    ),
                    input_scale=checkpoint.get_tensor(prefix + "input_scale"),
                )

        del regular_state
        meta_parameters = [
            name
            for name, parameter in transformer.named_parameters()
            if parameter.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                f"Transformer parameters were not loaded: {meta_parameters[:20]}"
            )
        meta_buffers = [
            name for name, buffer in transformer.named_buffers() if buffer.is_meta
        ]
        if meta_buffers:
            raise RuntimeError(
                f"Transformer runtime buffers are still on meta: {meta_buffers[:20]}"
            )
        transformer.eval()
        logger.info(
            "[Qwen] Quantized transformer ready: {} NVFP4, {} FP8 layers",
            sum(conf["format"] == "nvfp4" for conf in layers.values()),
            sum(conf["format"] == "float8_e4m3fn" for conf in layers.values()),
        )
        return transformer

    def load(self) -> Any:
        from diffusers import QwenImageEditPlusPipeline

        if not self._single_file:
            self._check_diffusers_model_ready(self.model_path)
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        else:
            transformer = self._load_quantized_transformer()
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.base_model_path,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )

        if self.cpu_offload:
            pipeline.enable_model_cpu_offload(device=self.device)
        else:
            pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline
