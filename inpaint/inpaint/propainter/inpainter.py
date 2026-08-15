"""ProPainter (streaming) video inpainting backend.

Drives the third-party ``propainter`` pip package (osmr streaming
reimplementation built on pytorchcv): RAFT optical flow, recurrent flow
completion and the ProPainter transformer are chained through
``ScaledProPainterIterator`` to inpaint a whole video with temporal
propagation, behind the same lazy-model pattern as
:class:`inpaint.lama.simple.SimpleLamaInpainter`.

Input convention (per the streaming package):

- ``frame_paths[i]``: RGB frame file (PNG).
- ``mask_paths[i]``: 1-channel binary mask file where nonzero pixels mark the
  region to inpaint. Frames without a mask pass a zero mask so ProPainter runs
  over the full sequence with complete temporal context.
- The network fills the mask region using flow-warped appearance from
  neighboring frames; with ``post_raw_mask_dilation > 0`` pixels outside the
  dilated mask are copied back from the original frames (pixel-exact
  background).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger
from PIL import Image

from inpaint.device import resolve_torch_device, set_cuda_device_if_indexed
from inpaint.propainter.args import ProPainterInpaintArgs

if TYPE_CHECKING:
    from torch import nn


class PathListSequencer:
    """Minimal sequence adapter over explicit file paths.

    The streaming package indexes its ``data`` with ``len()`` / slicing, so a
    plain list does not work (a list is treated as the raw-data list itself);
    this mirrors the package's own ``FilePathDirSequencer`` but for an explicit
    path list (frame/mask order is controlled by the caller).
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = [str(path) for path in paths]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int | slice):
        return self._paths[index]


class ProPainterInpainter:
    """Reusable streaming ProPainter inpainter for a whole video."""

    def __init__(self, args: ProPainterInpaintArgs, device: str | None = None) -> None:
        if args.mask_dilation <= 0:
            raise ValueError("--propainter.mask-dilation must be > 0.")
        if args.post_raw_mask_dilation < 0:
            raise ValueError("--propainter.post-raw-mask-dilation must be >= 0.")
        if args.resize_ratio <= 0:
            raise ValueError("--propainter.resize-ratio must be > 0.")
        if args.step <= 0:
            raise ValueError("--propainter.step must be > 0.")
        if args.raft_iters <= 0:
            raise ValueError("--propainter.raft-iters must be > 0.")
        if args.pp_window_size <= 0 or args.pp_stride <= 0:
            raise ValueError("--propainter window size / stride must be > 0.")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The 'propainter' streaming backend requires CUDA: the third-party "
                "package moves its frame/mask tensors to cuda unconditionally. Use "
                "--backend qwen or --backend lama on CPU-only machines."
            )
        self.device = resolve_torch_device(device or args.device)
        if self.device.type != "cuda":
            raise ValueError(
                "--propainter.device must be a CUDA device (the streaming "
                "package is CUDA-only)."
            )
        set_cuda_device_if_indexed(self.device)
        self.args = args
        self._models: tuple[nn.Module, nn.Module, nn.Module] | None = None

    @property
    def models(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        """Lazily built (RAFT, RFC, ProPainter) models."""

        if self._models is None:
            self._models = self._build_models()
        return self._models

    def _build_models(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        raft_path, pprfc_path, pp_path = self.args.resolve_model_paths()
        return (
            self._load_raft(raft_path),
            self._load_pprfc(pprfc_path),
            self._load_propainter(pp_path),
        )

    def _load_raft(self, path: Path | None) -> nn.Module:
        from pytorchcv.models.raft import raft_things

        return self._load_net(
            build=lambda pretrained: raft_things(
                pretrained=pretrained,
                in_normalize=False,
                iters=self.args.raft_iters,
            ),
            path=path,
            label="RAFT",
        )

    def _load_pprfc(self, path: Path | None) -> nn.Module:
        from pytorchcv.models.propainter_rfc import propainter_rfc

        return self._load_net(
            build=lambda pretrained: propainter_rfc(pretrained=pretrained),
            path=path,
            label="RFC",
        )

    def _load_propainter(self, path: Path | None) -> nn.Module:
        from pytorchcv.models.propainter import propainter

        return self._load_net(
            build=lambda pretrained: propainter(pretrained=pretrained),
            path=path,
            label="ProPainter",
        )

    def _load_net(
        self,
        build: Callable[[bool], nn.Module],
        path: Path | None,
        label: str,
    ) -> nn.Module:
        """Build one sub-network, preferring a local checkpoint.

        A missing local file falls back to the package's pretrained weights; a
        present-but-incompatible checkpoint (e.g. the original ProPainter-repo
        format) is rejected with a warning and the same pretrained fallback.
        """

        if path is None:
            logger.info(
                "[propainter] {}: no local checkpoint, using pretrained weights "
                "(auto-download on first use)",
                label,
            )
            return build(pretrained=True).to(self.device)

        path = path.expanduser()
        if not path.exists():
            logger.warning(
                "[propainter] {} checkpoint not found: {}; falling back to "
                "pretrained weights",
                label,
                path,
            )
            return build(pretrained=True).to(self.device)

        net = build(pretrained=False)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        try:
            net.load_state_dict(state_dict)
        except RuntimeError as err:
            reason = str(err).splitlines()[0] if str(err) else str(err)
            logger.warning(
                "[propainter] {} checkpoint {} is not in the streaming package's "
                "checkpoint format ({}...); falling back to pretrained weights. "
                "The files in ckpts/propainter are the original ProPainter-repo "
                "checkpoints; the 'propainter' pip package needs its own "
                "pytorchcv-format weights.",
                label,
                path,
                reason,
            )
            return build(pretrained=True).to(self.device)
        logger.info("[propainter] {} loaded from: {}", label, path)
        return net.to(self.device)

    def inpaint_video(
        self,
        frame_paths: list[Path],
        mask_paths: list[Path],
        output_paths: list[Path],
        args: ProPainterInpaintArgs,
    ) -> None:
        """Inpaint a whole video and save every frame.

        The three path lists must be non-empty and aligned: ``frame_paths[i]`` /
        ``mask_paths[i]`` / ``output_paths[i]`` describe one video position.
        Frames are processed in list order (the caller controls the windowing,
        e.g. ``--max-frames``). CUDA is required (see :class:`ProPainterInpainter`).
        """
        if not (len(frame_paths) == len(mask_paths) == len(output_paths) > 0):
            raise ValueError(
                "frame/mask/output path lists must be non-empty and aligned."
            )

        from propainter.propainter_video import (
            RawFrameSequencer,
            RawMaskSequencer,
            ScaledProPainterIterator,
        )

        raft_model, pprfc_model, pp_model = self.models
        raw_frames = RawFrameSequencer(data=PathListSequencer(frame_paths))
        raw_masks = RawMaskSequencer(data=PathListSequencer(mask_paths))
        iterator = ScaledProPainterIterator(
            raw_frames=raw_frames,
            raw_masks=raw_masks,
            image_resize_ratio=args.resize_ratio,
            mask_dilation=args.mask_dilation,
            post_raw_mask_dilation=args.post_raw_mask_dilation,
            raft_model=raft_model,
            pprfc_model=pprfc_model,
            pp_model=pp_model,
            use_cuda=True,
            raft_window_size=args.raft_window_size,
            pp_window_size=args.pp_window_size,
            pp_stride=args.pp_stride,
            step=args.step,
        )

        started_at = time.perf_counter()
        chunk_start = 0
        try:
            with torch.inference_mode():
                for chunk in iterator:
                    chunk_end = chunk_start + len(chunk)
                    for offset, frame_rgb in enumerate(chunk):
                        Image.fromarray(frame_rgb).save(
                            output_paths[chunk_start + offset]
                        )
                    chunk_start = chunk_end
        except torch.OutOfMemoryError as err:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory during ProPainter inference. The model "
                "weights are small (~200 MB), but the transformer processes a "
                "whole frame window at once and 1080p intermediate activations "
                "can exceed 30 GB. Lower --video.propainter.resize-ratio (e.g. "
                "0.5 keeps the 1080p output via full-resolution blending) and "
                "keep the default --video.propainter.pp-window-size."
            ) from err
        logger.info(
            "[propainter] Inference complete: frames={} elapsed={:.2f}s, out={}",
            len(output_paths),
            time.perf_counter() - started_at,
            output_paths[0].parent,
        )
