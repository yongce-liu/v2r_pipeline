"""Argument dataclasses for the MiniMax-Remover backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

UPSTREAM_REVISION = "28e12b450d8a72a7547b86940a4985e6ad90d75b"
DEFAULT_MODEL_PATH = Path(__file__).parents[3] / "ckpts" / "minimax-remover"


@dataclass
class MiniMaxRemoverInpaintArgs:
    """Arguments for whole-video MiniMax-Remover inference.

    The upstream project is a checkpoint-only diffusers pipeline rather than
    a pip package. Its model and pipeline implementations are vendored into
    this package at ``UPSTREAM_REVISION``; only the Hugging Face checkpoint
    remains external.
    """

    model_path: Path | None = DEFAULT_MODEL_PATH
    """Local ``zibojia/minimax-remover`` snapshot. No runtime download is
    performed; the directory must contain ``vae``, ``transformer`` and
    ``scheduler`` subdirectories."""

    device: str = "auto"
    """Torch device: ``auto`` or ``cuda[:N]``. The vendored pipeline runs in
    fp16 and does not support CPU inference."""

    height: int = 480
    """Network inference height. Must be divisible by 16."""

    width: int = 832
    """Network inference width. Must be divisible by 16."""

    auto_orientation: bool = True
    """Swap the configured inference size for portrait clips, matching the
    upstream demo's orientation policy."""

    steps: int = 12
    """Number of scheduler steps. The upstream quick start uses 12; the paper
    reports that the distilled model can operate with as few as 6."""

    mask_dilation: int = 12
    """Iterations of binary dilation applied inside the upstream pipeline to
    the mask condition."""

    post_mask_dilation: int = 24
    """Dilation in SOURCE-frame pixels used when compositing the generated
    result. Outside this region the original frame is copied exactly."""

    seed: int = 42
    """Random seed for the initial latent noise."""

    overwrite: bool = True
    """Clear existing MiniMax outputs and recompute. With it off, a run whose
    outputs are all present is reused without model inference."""
