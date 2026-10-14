"""Video-mode MiniMax-Remover object-removal inpainting.

Reads the ``segment`` stage's ``masks.json`` and writes the same
``outputs/<clip>/inpaint/`` contract as the Qwen, LaMa and ProPainter backends.
The model is video-based: all selected frames are passed in temporal order,
frames without a mask receive an all-zero mask, and each result is composited
back at the source resolution.
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
from inpaint.minimax import __version__ as _minimax_version
from inpaint.minimax.args import UPSTREAM_REVISION, MiniMaxRemoverInpaintArgs
from inpaint.minimax.inpainter import MiniMaxRemoverInpainter

INPAINT_FILENAME_PATTERN = "{:06d}.png"
VIS_FILENAME_PATTERN = "{:06d}.png"
ZERO_MASK_DIRNAME = ".minimax_masks"


@dataclass
class MiniMaxVideoArgs:
    """Arguments for whole-video MiniMax-Remover inpainting."""

    masks_json: Path | None = None
    """Path to the ``segment`` stage's ``masks.json`` (required for video mode)."""

    output_root: Path = Path(__file__).parents[3] / "outputs"
    """Root under which ``<clip_stem>/inpaint/`` is created."""

    vis: bool = True
    """Write an original + edited side-by-side image for every processed frame."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames in the manifest)."""

    minimax: MiniMaxRemoverInpaintArgs = field(
        default_factory=MiniMaxRemoverInpaintArgs
    )
    """MiniMax-Remover checkpoint and inference settings."""


@dataclass(frozen=True)
class MiniMaxEntry:
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
class MiniMaxVideoOutputs:
    """Everything produced by one MiniMax-Remover video run."""

    clip_root: Path
    stage_dir: Path
    inpainted_dir: Path
    inpainted_vis_dir: Path | None
    inpainted_json_path: Path
    config_json_path: Path
    entries: list[MiniMaxEntry]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_existing_entries(inpainted_json: Path) -> dict[int, MiniMaxEntry]:
    """Reuse entries only when the prior manifest is from this backend."""

    if not inpainted_json.exists():
        return {}
    try:
        data = json.loads(inpainted_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if data.get("backend") != "minimax":
        return {}

    existing: dict[int, MiniMaxEntry] = {}
    for raw in data.get("entries", []):
        try:
            existing[int(raw["index"])] = MiniMaxEntry(
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
    args: MiniMaxVideoArgs,
    mask_manifest,
    inpainted_dir: Path,
    inpainted_vis_dir: Path | None,
    entries: list[MiniMaxEntry],
) -> dict:
    return {
        "schema_version": "1.0",
        "stage": "inpaint",
        "backend": "minimax",
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


def _config_dict(args: MiniMaxVideoArgs) -> dict:
    model_path = (
        str(args.minimax.model_path.expanduser())
        if args.minimax.model_path is not None
        else None
    )
    return {
        "package": {"name": "inpaint.minimax", "version": _minimax_version},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "minimax",
        "inpaint": {
            "model_path": model_path,
            "upstream_revision": UPSTREAM_REVISION,
            "device": args.minimax.device,
            "height": args.minimax.height,
            "width": args.minimax.width,
            "auto_orientation": args.minimax.auto_orientation,
            "steps": args.minimax.steps,
            "mask_dilation": args.minimax.mask_dilation,
            "post_mask_dilation": args.minimax.post_mask_dilation,
            "seed": args.minimax.seed,
            "overwrite": args.minimax.overwrite,
            "vis": args.vis,
            "max_frames": args.max_frames,
            "dtype": "float16",
        },
        "software": {},
    }


def _write_zero_mask(path: Path, height: int, width: int) -> None:
    """Write an all-zero mask for a frame with no detected object."""

    Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(path)


def run_minimax_video_inpaint(
    args: MiniMaxVideoArgs,
    inpainter: MiniMaxRemoverInpainter | None = None,
) -> MiniMaxVideoOutputs:
    """Remove masked objects from every selected frame with MiniMax-Remover.

    When ``inpainter`` is provided (for tests) it must expose
    ``inpaint_video(frame_paths, mask_paths, output_paths, args)``.
    """

    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")
    if args.masks_json is None:
        raise ValueError("--minimax-video.masks-json is required for video mode.")

    masks_json = args.masks_json.expanduser().resolve()
    mask_manifest = load_mask_manifest(masks_json)

    frames_json = Path(mask_manifest.source_frames_json)
    manifest = load_frame_manifest(frames_json)
    frame_by_index = {entry.index: entry for entry in manifest.entries}

    clip_stem = masks_json.parent.parent.name
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "inpaint"
    inpainted_dir = stage_dir / "inpainted"
    inpainted_vis_dir = stage_dir / "inpainted_vis" if args.vis else None

    if args.minimax.overwrite:
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

    selected_entries = [
        mask_entry
        for mask_entry in selected
        if frame_by_index.get(mask_entry.index) is not None
    ]
    if not selected_entries:
        raise ValueError("No frames selected for MiniMax-Remover inpainting.")

    frame_paths: list[Path] = []
    mask_paths: list[Path] = []
    output_paths: list[Path] = []
    indices: list[int] = []
    frame_rgb_by_index: dict[int, np.ndarray] = {}
    scratch_mask_dir: Path | None = None

    for mask_entry in selected_entries:
        frame_entry = frame_by_index[mask_entry.index]
        frame_rgb = load_rgb_image(frame_entry.path)
        frame_rgb_by_index[mask_entry.index] = frame_rgb
        frame_paths.append(frame_entry.path)
        output_paths.append(
            inpainted_dir / INPAINT_FILENAME_PATTERN.format(mask_entry.index)
        )
        indices.append(mask_entry.index)

        if mask_entry.has_mask and mask_entry.mask_path is not None:
            mask_paths.append(mask_entry.mask_path)
        else:
            if scratch_mask_dir is None:
                scratch_mask_dir = stage_dir / ZERO_MASK_DIRNAME
                scratch_mask_dir.mkdir(parents=True, exist_ok=True)
            zero_mask_path = scratch_mask_dir / f"{mask_entry.index:06d}.png"
            _write_zero_mask(
                zero_mask_path,
                frame_rgb.shape[0],
                frame_rgb.shape[1],
            )
            mask_paths.append(zero_mask_path)

    prior_entries = _load_existing_entries(stage_dir / "inpainted.json")
    can_reuse = (
        not args.minimax.overwrite
        and all(prior_entries.get(index) is not None for index in indices)
        and all(path.exists() for path in output_paths)
        and (
            not args.vis
            or all(
                (inpainted_vis_dir / VIS_FILENAME_PATTERN.format(index)).exists()
                for index in indices
            )
        )
    )

    try:
        if can_reuse:
            entries = [prior_entries[index] for index in indices]
        else:
            active_inpainter = inpainter or MiniMaxRemoverInpainter(args.minimax)
            active_inpainter.inpaint_video(
                frame_paths,
                mask_paths,
                output_paths,
                args.minimax,
            )

            entries = []
            for mask_entry in selected_entries:
                frame_entry = frame_by_index[mask_entry.index]
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
                        overwrite=args.minimax.overwrite,
                    )

                entries.append(
                    MiniMaxEntry(
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
            args,
            mask_manifest,
            inpainted_dir,
            inpainted_vis_dir,
            entries,
        ),
    )
    _write_json(stage_dir / "config.json", _config_dict(args))

    masked_count = sum(1 for entry in entries if entry.has_mask)
    logger.info(
        "[minimax] Done: processed={} masked={} vis={} out={}",
        len(entries),
        masked_count,
        args.vis,
        stage_dir,
    )

    return MiniMaxVideoOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        inpainted_dir=inpainted_dir,
        inpainted_vis_dir=inpainted_vis_dir,
        inpainted_json_path=stage_dir / "inpainted.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
    )


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB array to the source resolution."""

    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.LANCZOS))
