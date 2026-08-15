"""Argument dataclasses for the ProPainter inpainting backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_RAFT_FILENAME = "raft-things.pth"
DEFAULT_PPRFC_FILENAME = "recurrent_flow_completion.pth"
DEFAULT_PP_FILENAME = "ProPainter.pth"


@dataclass
class ProPainterInpaintArgs:
    """Arguments for ProPainter video inpainting.

    Drives the ``propainter`` pip package (the osmr streaming reimplementation
    built on pytorchcv), which needs three sub-network checkpoints: RAFT optical
    flow, recurrent flow completion (RFC) and the ProPainter transformer.

    Local checkpoints are preferred for offline runs. When a configured file is
    missing or not in the package's pytorchcv checkpoint format (e.g. the
    original ProPainter-repo ``.pth`` files), the inpainter logs a warning and
    falls back to the package's own pretrained weights (auto-downloaded on
    first use).
    """

    model_dir: Path | None = Path(__file__).parents[3] / "ckpts" / "propainter"
    """Directory holding the three sub-network checkpoints (defaults to the
    repo's ``ckpts/propainter``). None disables local checkpoints entirely."""

    raft_model_path: Path | None = None
    """RAFT optical-flow checkpoint (pytorchcv format). Defaults to
    ``<model_dir>/raft-things.pth``."""

    pprfc_model_path: Path | None = None
    """Recurrent flow completion checkpoint. Defaults to
    ``<model_dir>/recurrent_flow_completion.pth``."""

    pp_model_path: Path | None = None
    """ProPainter transformer checkpoint. Defaults to
    ``<model_dir>/ProPainter.pth``."""

    device: str = "auto"
    """Torch device: ``auto``, ``cuda[:N]``, or ``cpu``."""

    resize_ratio: float = 0.5
    """Scale applied to frames/masks before the network (dimensions are floored
    to multiples of 8). 1.0 keeps the source resolution."""

    mask_dilation: int = 10
    """Dilation (px) of the network-input mask. Must be > 0 (the streaming
    package asserts this)."""

    post_raw_mask_dilation: int = 5
    """Extra dilation (px) used when blending the output: pixels outside this
    dilated raw mask are copied back from the original frame, so non-masked
    regions stay pixel-exact. 0 returns the raw network output everywhere."""

    raft_iters: int = 20
    """Number of RAFT refinement iterations."""

    raft_window_size: int | None = None
    """RAFT sequencer window size (None = package default)."""

    pp_window_size: int = 80
    """ProPainter temporal attention window size."""

    pp_stride: int = 5
    """ProPainter sliding-window stride."""

    step: int = 10
    """Frames per streaming iteration (the chunk size used for saving)."""

    overwrite: bool = True
    """Clear existing propainter outputs and recompute. With it off, a run whose
    outputs are all present is skipped (idempotent re-runs)."""

    def resolve_model_paths(self) -> tuple[Path | None, Path | None, Path | None]:
        """Resolve the three checkpoint paths (explicit path or ``model_dir``).

        Returns ``(raft, pprfc, propainter)``; a None entry means "use the
        package's pretrained weights".
        """

        model_dir = self.model_dir.expanduser() if self.model_dir is not None else None

        def resolve(path: Path | None, filename: str) -> Path | None:
            if path is not None:
                return path.expanduser()
            if model_dir is None:
                return None
            return model_dir / filename

        return (
            resolve(self.raft_model_path, DEFAULT_RAFT_FILENAME),
            resolve(self.pprfc_model_path, DEFAULT_PPRFC_FILENAME),
            resolve(self.pp_model_path, DEFAULT_PP_FILENAME),
        )
