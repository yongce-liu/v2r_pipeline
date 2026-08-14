"""Argument dataclasses for the LaMa inpainting backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LamaInpaintArgs:
    """Arguments for LaMa (big-lama) inpainting of one masked frame.

    The backend is ``simple-lama-inpainting`` (a TorchScript big-lama wrapper).
    ``model_path`` is a local ``.pt`` in the simple-lama format (preferred for
    offline runs); when unset the package auto-downloads ``big-lama.pt`` from the
    author's GitHub release on first use.
    """

    model_path: Path | None = None
    """Local TorchScript ``big-lama.pt`` in the ``simple-lama`` format (e.g.
    ``ckpts/big-lama/big-lama.pt``). Loaded via the ``LAMA_MODEL`` env var the
    package reads, so this works fully offline. None falls back to the
    package's auto-download from GitHub."""

    device: str = "auto"
    """Torch device: ``auto``, ``cuda[:N]``, or ``cpu``."""

    overwrite: bool = True
    """Clear existing lama outputs and recompute. With it off, prior per-frame
    outputs are reused (idempotent re-runs)."""

    dilate_ratio: float = 0.05
    """Morphologically dilate the mask outward by this fraction of the frame's
    shorter side before inpainting. Segment masks hug the hand/arm contour
    tightly, which leaves a faint original-edge halo in the fill; a small
    dilation makes LaMa erase that halo. Being a ratio of the image size (not a
    fixed pixel count) it stays consistent across source resolutions: at 1080p
    ``0.05`` ≈ 54 px, close to the old fixed-pixel default of 50; a 720p frame
    gets half that, a 4K frame twice that. ``0`` disables dilation."""

    repeat: int = 1
    """Number of LaMa passes per masked frame. Each pass re-feeds the previous
    output with the SAME (dilated) mask — the hole is re-zeroed inside the
    network, so the boundary context is the previous fill and large holes
    converge instead of being a single-shot fill. ``1`` = single pass (current
    behavior); 2–3 typically cleans up residual boundary artifacts at ≈N× the
    per-frame cost."""
