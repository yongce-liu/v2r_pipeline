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

    dilate_px: int = 100
    """Morphologically dilate the mask outward by this many pixels before
    inpainting. Segment masks hug the hand/arm contour tightly, which leaves a
    faint original-edge halo in the fill; a small dilation makes LaMa erase
    that halo. The effective expansion is area-aware: at a reference mask of
    ~1e6 px (a mid-size hand at 1080p) it equals ``dilate_px``, growing
    sub-linearly with mask area. ``0`` disables dilation."""
