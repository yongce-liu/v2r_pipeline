"""Video-mode ProPainter arm-removal inpainting.

Drives the ``propainter`` pip package (streaming RAFT + flow completion +
transformer). Unlike the per-frame Qwen/LaMa backends, ProPainter inpaits the
whole video at once so it can propagate appearance across frames. Every frame of
the selected range is fed to the network (frames without a mask pass a zero
mask), and the results are written to the same ``outputs/<clip>/inpaint/``
layout as the other backends — the output does not differ by backend.

Output layout (identical to the Qwen / LaMa backends):

.. code-block:: text

    outputs/<clip>/inpaint/
    ├── config.json     # effective run config (same style as process)
    ├── inpainted.json  # per-frame inpaint manifest (index / paths / backend)
    ├── inpainted/      # edited frames (000000.png, ...)
    └── inpainted_vis/  # original + edited side-by-side (000000.png, ...),
                        # only when vis=True
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from inpaint.frames import load_frame_manifest
from inpaint.masks import load_mask_manifest
from inpaint.media import load_rgb_image, save_image, save_side_by_side
from inpaint.propainter import __version__ as _propainter_version
from inpaint.propainter.args import ProPainterInpaintArgs
from inpaint.propainter.inpainter import ProPainterInpainter

INPAINT_FILENAME_PATTERN = "{:06d}.png"
VIS_FILENAME_PATTERN = "{:06d}.png"
ZERO_MASK_DIRNAME = ".propainter_masks"


@dataclass
class ProPainterVideoArgs:
    """Arguments for whole-video ProPainter inpainting."""

    masks_json: Path | None = None
    """Path to the ``segment`` stage's ``masks.json`` (required for video mode)."""

    output_root: Path = Path(__file__).parents[3] / "outputs"
    """Root under which ``<clip_stem>/inpaint/`` is created."""

    vis: bool = True
    """Write an original + edited side-by-side image for every processed frame."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames in the manifest)."""

    propainter: ProPainterInpaintArgs = field(default_factory=ProPainterInpaintArgs)
    """ProPainter settings (checkpoints, device, windowing, ...)."""


@dataclass(frozen=True)
class ProPainterEntry:
    """Per-frame inpaint record written into ``inpainted.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    has_mask: bool
    mask_filename: str | None
    inpainted_filename: str
    vis_filename: str | None
    height: int
    width: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "timestamp_sec": self.timestamp_sec,
            "has_mask": self.has_mask,
            "mask_filename": self.mask_filename,
            "inpainted_filename": self.inpainted_filename,
            "vis_filename": self.vis_filename,
            "height": self.height,
            "width": self.width,
        }


@dataclass(frozen=True)
class ProPainterVideoOutputs:
    """Everything produced by one ProPainter video inpainting run."""

    clip_root: Path
    stage_dir: Path
    inpainted_dir: Path
    inpainted_vis_dir: Path | None
    inpainted_json_path: Path
    config_json_path: Path
    entries: list[ProPainterEntry]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_existing_entries(inpainted_json: Path) -> dict[int, ProPainterEntry]:
    """Reuse previously written inpaint entries (idempotent non-overwrite runs)."""

    if not inpainted_json.exists():
        return {}
    try:
        data = json.loads(inpainted_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    existing: dict[int, ProPainterEntry] = {}
    for raw in data.get("entries", []):
        try:
            existing[int(raw["index"])] = ProPainterEntry(
                index=int(raw["index"]),
                frame_filename=str(raw.get("frame_filename", "")),
                timestamp_sec=raw.get("timestamp_sec"),
                has_mask=bool(raw.get("has_mask")),
                mask_filename=raw.get("mask_filename"),
                inpainted_filename=str(raw.get("inpainted_filename", "")),
                vis_filename=raw.get("vis_filename"),
                height=int(raw.get("height", 0)),
                width=int(raw.get("width", 0)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return existing


def _inpaint_manifest_dict(
    args: ProPainterVideoArgs,
    mask_manifest,
    inpainted_dir: Path,
    inpainted_vis_dir: Path | None,
    entries: list[ProPainterEntry],
) -> dict:
    return {
        "schema_version": "1.0",
        "stage": "inpaint",
        "backend": "propainter",
        "source_masks_json": str(args.masks_json.expanduser().resolve()),
        "source_frames_json": mask_manifest.source_frames_json,
        "source_video": mask_manifest.source_video,
        "fps": mask_manifest.fps,
        "width": mask_manifest.width,
        "height": mask_manifest.height,
        "frame_format": mask_manifest.frame_format,
        "frame_count": mask_manifest.frame_count,
        "frames_dir": str(inpainted_dir),
        "processed_count": len(entries),
        "masked_count": sum(1 for entry in entries if entry.has_mask),
        "inpaint_format": "png",
        "inpainted_dir": str(inpainted_dir),
        "inpainted_vis_dir": (
            str(inpainted_vis_dir) if inpainted_vis_dir is not None else None
        ),
        "vis_enabled": args.vis,
        "entries": [entry.to_dict() for entry in entries],
    }


def _config_dict(args: ProPainterVideoArgs) -> dict:
    raft_path, pprfc_path, pp_path = args.propainter.resolve_model_paths()
    return {
        "package": {"name": "inpaint.propainter", "version": _propainter_version},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "propainter",
        "inpaint": {
            "backend": "streaming",
            "raft_model_path": str(raft_path) if raft_path is not None else None,
            "pprfc_model_path": str(pprfc_path) if pprfc_path is not None else None,
            "pp_model_path": str(pp_path) if pp_path is not None else None,
            "device": args.propainter.device,
            "resize_ratio": args.propainter.resize_ratio,
            "mask_dilation": args.propainter.mask_dilation,
            "post_raw_mask_dilation": args.propainter.post_raw_mask_dilation,
            "raft_iters": args.propainter.raft_iters,
            "raft_window_size": args.propainter.raft_window_size,
            "pp_window_size": args.propainter.pp_window_size,
            "pp_stride": args.propainter.pp_stride,
            "step": args.propainter.step,
            "overwrite": args.propainter.overwrite,
            "vis": args.vis,
            "max_frames": args.max_frames,
        },
        "software": {},
    }


def _write_zero_mask(path: Path, height: int, width: int) -> None:
    """Write an all-zero binary mask (frames without a hand/arm mask)."""

    Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(path)


def run_propainter_video_inpaint(
    args: ProPainterVideoArgs,
    inpainter: ProPainterInpainter | None = None,
) -> ProPainterVideoOutputs:
    """Remove the hand/arm from every frame of a segment output with ProPainter.

    ``args.masks_json`` is read from the ``segment`` stage; its referenced frame
    manifest is resolved to the original frames. The three sub-networks are
    loaded once and the whole selected range is inpainted in one pass (temporal
    propagation), then every frame is written to the shared
    ``<clip>/inpaint/`` layout. When ``inpainter`` is provided (e.g. a test
    double) it must expose ``inpaint_video(frame_paths, mask_paths,
    output_paths, args)``.
    """
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")
    if args.masks_json is None:
        raise ValueError("--propainter-video.masks-json is required for video mode.")

    masks_json = args.masks_json.expanduser().resolve()
    mask_manifest = load_mask_manifest(masks_json)

    frames_json = Path(mask_manifest.source_frames_json)
    manifest = load_frame_manifest(frames_json)
    frame_by_index = {entry.index: entry for entry in manifest.entries}

    clip_stem = masks_json.parent.parent.name  # outputs/<clip>/segment/masks.json
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "inpaint"
    inpainted_dir = stage_dir / "inpainted"
    inpainted_vis_dir = stage_dir / "inpainted_vis" if args.vis else None

    if args.propainter.overwrite:
        if inpainted_dir.exists():
            shutil.rmtree(inpainted_dir)
        if inpainted_vis_dir is not None and inpainted_vis_dir.exists():
            shutil.rmtree(inpainted_vis_dir)
    inpainted_dir.mkdir(parents=True, exist_ok=True)
    if inpainted_vis_dir is not None:
        inpainted_vis_dir.mkdir(parents=True, exist_ok=True)

    selected = mask_manifest.entries
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    # Build aligned frame / mask / output path lists over the selected range.
    # Frames without a mask get a generated zero mask in a scratch dir so the
    # whole sequence keeps its temporal context.
    frame_paths: list[Path] = []
    mask_paths: list[Path] = []
    output_paths: list[Path] = []
    frame_rgb_by_index: dict[int, np.ndarray] = {}
    selected_indices: list[int] = []
    scratch_mask_dir: Path | None = None
    for mask_entry in selected:
        frame_entry = frame_by_index.get(mask_entry.index)
        if frame_entry is None:
            logger.warning(
                "[propainter] skip frame {}: not in the frame manifest",
                mask_entry.index,
            )
            continue

        frame_rgb = load_rgb_image(frame_entry.path)
        frame_rgb_by_index[mask_entry.index] = frame_rgb
        selected_indices.append(mask_entry.index)
        frame_paths.append(frame_entry.path)
        output_paths.append(
            inpainted_dir / INPAINT_FILENAME_PATTERN.format(mask_entry.index)
        )

        if mask_entry.has_mask and mask_entry.mask_path is not None:
            mask_paths.append(mask_entry.mask_path)
        else:
            if scratch_mask_dir is None:
                scratch_mask_dir = stage_dir / ZERO_MASK_DIRNAME
                scratch_mask_dir.mkdir(parents=True, exist_ok=True)
            zero_mask_path = scratch_mask_dir / (
                INPAINT_FILENAME_PATTERN.format(mask_entry.index)
            )
            if not zero_mask_path.exists():
                _write_zero_mask(
                    zero_mask_path,
                    frame_rgb.shape[0],
                    frame_rgb.shape[1],
                )
            mask_paths.append(zero_mask_path)

    if not frame_paths:
        raise ValueError("No frames selected for propainter inpainting.")

    prior_entries = _load_existing_entries(stage_dir / "inpainted.json")
    indices = selected_indices
    can_reuse = (
        not args.propainter.overwrite
        and all(prior_entries.get(i) is not None for i in indices)
        and all(
            (inpainted_dir / INPAINT_FILENAME_PATTERN.format(i)).exists()
            for i in indices
        )
        and (
            not args.vis
            or all(
                (inpainted_vis_dir / VIS_FILENAME_PATTERN.format(i)).exists()
                for i in indices
            )
        )
    )

    try:
        if can_reuse:
            entries = [prior_entries[i] for i in indices]
        else:
            active_inpainter = inpainter or ProPainterInpainter(args.propainter)
            active_inpainter.inpaint_video(
                frame_paths,
                mask_paths,
                output_paths,
                args.propainter,
            )

            entries: list[ProPainterEntry] = []
            for mask_entry in selected:
                frame_entry = frame_by_index.get(mask_entry.index)
                if frame_entry is None:
                    continue
                inpaint_path = inpainted_dir / INPAINT_FILENAME_PATTERN.format(
                    mask_entry.index
                )
                vis_path = (
                    inpainted_vis_dir / VIS_FILENAME_PATTERN.format(mask_entry.index)
                    if inpainted_vis_dir is not None
                    else None
                )
                frame_rgb = frame_rgb_by_index[mask_entry.index]
                frame_h, frame_w = frame_rgb.shape[:2]

                edited_rgb = load_rgb_image(inpaint_path)
                if edited_rgb.shape[:2] != (frame_h, frame_w):
                    edited_rgb = _resize_rgb(edited_rgb, frame_w, frame_h)
                    save_image(edited_rgb, inpaint_path, True)

                if vis_path is not None:
                    save_side_by_side(
                        frame_rgb,
                        edited_rgb,
                        vis_path,
                        overwrite=args.propainter.overwrite,
                    )

                entries.append(
                    ProPainterEntry(
                        index=mask_entry.index,
                        frame_filename=frame_entry.frame_filename,
                        timestamp_sec=frame_entry.timestamp_sec,
                        has_mask=mask_entry.has_mask,
                        mask_filename=mask_entry.mask_filename,
                        inpainted_filename=inpaint_path.name,
                        vis_filename=vis_path.name if vis_path is not None else None,
                        height=frame_h,
                        width=frame_w,
                    )
                )
    finally:
        if scratch_mask_dir is not None:
            shutil.rmtree(scratch_mask_dir, ignore_errors=True)

    _write_json(
        stage_dir / "inpainted.json",
        _inpaint_manifest_dict(
            args, mask_manifest, inpainted_dir, inpainted_vis_dir, entries
        ),
    )
    _write_json(stage_dir / "config.json", _config_dict(args))

    masked_count = sum(1 for entry in entries if entry.has_mask)
    logger.info(
        "[propainter] Done: processed={} masked={} vis={} out={}",
        len(entries),
        masked_count,
        args.vis,
        stage_dir,
    )

    return ProPainterVideoOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        inpainted_dir=inpainted_dir,
        inpainted_vis_dir=inpainted_vis_dir,
        inpainted_json_path=stage_dir / "inpainted.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
    )


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB array to the given resolution (LANCZOS, matching the
    reference EgoDataPipe behavior)."""

    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.LANCZOS))
