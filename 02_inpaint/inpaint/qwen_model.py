"""Qwen-Image-Edit model loading (diffusers dir or single-file checkpoint).

Qwen-Image-Edit runs through the diffusers ``QwenImageEditPlusPipeline``. Its
checkpoints come in two layouts:

- A **diffusers directory** (``model_index.json``, ``transformer/``, ``vae/``,
  ``text_encoder/``, ...). Loaded with ``from_pretrained`` and used directly.
- A **single-file ComfyUI checkpoint** (a ``.safetensors`` state dict of a
  *base* ``QwenImageEditPlusPipeline`` transformer), e.g. the quantized FP8 /
  NVFP4 releases. Verified against the reference base model, these files carry
  the *same* 1933 state-dict keys as the diffusers ``QwenImageTransformer2DModel``
  (``transformer_blocks.*``, ``img_in``, ``txt_in``, ``time_text_embed``, ...),
  just with quantized dtypes. We load the diffusers base model first, then
  ``load_state_dict`` the single file onto its transformer — torch's
  ``copy_`` converts the quantized dtype (``float8_e4m3fn``, ``torch.uint8``
  NVFP4 packing) into the base model's ``bfloat16`` in place.

Only the transformer differs between the two paths; the VAE / text encoder /
scheduler all come from the base model (in the single-file case the base is
``--qwen.base-model-path``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger


class QwenEditModelLoader:
    """Build a ``QwenImageEditPlusPipeline`` from either checkpoint layout."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        base_model_path: Path | None = None,
        cpu_offload: bool = False,
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
                "Single-file checkpoint requires --qwen.base-model-path (the "
                "diffusers base model whose VAE/text encoder/transformer layout "
                "the single file builds on)."
            )
        if self._single_file and not self.base_model_path.exists():
            raise FileNotFoundError(
                f"Base model path not found: {self.base_model_path}"
            )

    @property
    def checkpoint_name(self) -> str:
        return self.model_path.name if self._single_file else self.model_path.name

    def load(self) -> Any:
        """Load the pipeline (heavy: downloads/allocates the base model weights)."""

        from diffusers import QwenImageEditPlusPipeline

        base_dir = self.base_model_path if self._single_file else self.model_path
        logger.info(
            "[Qwen] Loading base model: device={}, path={}", self.device, base_dir
        )
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            base_dir,
            torch_dtype=torch.bfloat16,
        )

        if self._single_file:
            self._load_single_file(pipeline)

        if self.cpu_offload:
            pipeline.enable_sequential_cpu_offload(device=self.device)
        else:
            pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def _load_single_file(self, pipeline: Any) -> None:
        """Replace the pipeline transformer's weights with the single file's.

        The single file is a state dict of the same ``QwenImageTransformer2DModel``
        layout (verified: identical key sets), so a strict ``load_state_dict`` is
        the whole load — torch's in-place ``copy_`` converts the quantized dtype
        to the base model's ``bfloat16``.
        """
        from safetensors.torch import load_file

        transformer = pipeline.transformer
        logger.info("[Qwen] Loading single-file transformer: {}", self.model_path)
        state_dict = load_file(str(self.model_path))

        incompatible = transformer.load_state_dict(state_dict, strict=True)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
        if missing or unexpected:
            logger.warning(
                "[Qwen] single-file state dict mismatch: missing={} unexpected={}",
                missing,
                unexpected,
            )
        logger.info(
            "[Qwen] Loaded {} transformer parameters from single file.", len(state_dict)
        )
