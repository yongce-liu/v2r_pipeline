"""Video-mode (frame-by-frame) LaMa arm-removal inpainting.

Drives the ``simple-lama-inpainting`` TorchScript big-lama wrapper. For every
frame with a hand/arm mask the frame + the real segment mask are fed to LaMa,
which fills the masked region from context; the result is written to
``outputs/<clip>/lama/``. Frames without a mask are copied through unchanged, so
the output always has one image per source frame.

The input frame is the *clean* frame and the mask is binary (nonzero = inpaint
region); the hole is zeroed inside the network, never a colored overlay.

Output layout (parallel to ``outputs/<clip>/inpaint``):

.. code-block:: text

    outputs/<clip>/lama/
    ├── config.json     # effective run config (same style as process)
    ├── lama.json       # per-frame inpaint manifest (index / paths)
    ├── lama/           # edited frames (lama_000000.png, ...)
    └── lama_vis/       # original + edited side-by-side (vis_000000.png, ...),
                        # only when vis=True
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from inpaint.frames import load_frame_manifest
from inpaint.lama import __version__ as _lama_version
from inpaint.lama.args import LamaInpaintArgs
from inpaint.lama.simple import SimpleLamaInpainter
from inpaint.masks import load_mask_manifest
from inpaint.media import load_mask, load_rgb_image, save_image, save_side_by_side

LAMA_FILENAME_PATTERN = "lama_{:06d}.png"
VIS_FILENAME_PATTERN = "vis_{:06d}.png"


@dataclass
class LamaVideoArgs:
    """Arguments for frame-by-frame LaMa inpainting of a whole video."""

    masks_json: Path | None = None
    """Path to the ``segment`` stage's ``masks.json`` (required for video mode)."""

    output_root: Path = Path(__file__).parents[3] / "outputs"
    """Root under which ``<clip_stem>/lama/`` is created."""

    vis: bool = True
    """Write an original + edited side-by-side image for every processed frame."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames in the manifest)."""

    lama: LamaInpaintArgs = field(default_factory=LamaInpaintArgs)
    """LaMa settings (checkpoint, device, ...)."""


@dataclass(frozen=True)
class LamaEntry:
    """Per-frame inpaint record written into ``lama.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    has_mask: bool
    mask_filename: str | None
    lama_filename: str
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
            "lama_filename": self.lama_filename,
            "vis_filename": self.vis_filename,
            "height": self.height,
            "width": self.width,
        }


@dataclass(frozen=True)
class LamaVideoOutputs:
    """Everything produced by one LaMa video inpainting run."""

    clip_root: Path
    stage_dir: Path
    lama_dir: Path
    lama_vis_dir: Path | None
    lama_json_path: Path
    config_json_path: Path
    entries: list[LamaEntry]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_existing_entries(lama_json: Path) -> dict[int, LamaEntry]:
    """Reuse previously written lama entries (idempotent non-overwrite runs)."""

    if not lama_json.exists():
        return {}
    try:
        data = json.loads(lama_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    existing: dict[int, LamaEntry] = {}
    for raw in data.get("entries", []):
        try:
            existing[int(raw["index"])] = LamaEntry(
                index=int(raw["index"]),
                frame_filename=str(raw.get("frame_filename", "")),
                timestamp_sec=raw.get("timestamp_sec"),
                has_mask=bool(raw.get("has_mask")),
                mask_filename=raw.get("mask_filename"),
                lama_filename=str(raw.get("lama_filename", "")),
                vis_filename=raw.get("vis_filename"),
                height=int(raw.get("height", 0)),
                width=int(raw.get("width", 0)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return existing


def _lama_manifest_dict(
    args: LamaVideoArgs,
    mask_manifest,
    lama_dir: Path,
    lama_vis_dir: Path | None,
    entries: list[LamaEntry],
) -> dict:
    return {
        "backend": "lama",
        "source_masks_json": str(args.masks_json.expanduser().resolve()),
        "source_frames_json": mask_manifest.source_frames_json,
        "source_video": mask_manifest.source_video,
        "fps": mask_manifest.fps,
        "width": mask_manifest.width,
        "height": mask_manifest.height,
        "frame_format": mask_manifest.frame_format,
        "frame_count": mask_manifest.frame_count,
        "processed_count": len(entries),
        "masked_count": sum(1 for entry in entries if entry.has_mask),
        "lama_format": "png",
        "lama_dir": str(lama_dir),
        "lama_vis_dir": str(lama_vis_dir) if lama_vis_dir is not None else None,
        "vis_enabled": args.vis,
        "entries": [entry.to_dict() for entry in entries],
    }


def _config_dict(args: LamaVideoArgs) -> dict:
    model_path = (
        str(args.lama.model_path.expanduser())
        if args.lama.model_path is not None
        else None
    )
    return {
        "package": {"name": "inpaint.lama_inpaint", "version": _lama_version},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "lama",
        "inpaint": {
            "backend": "simple",
            "model_path": model_path,
            "overwrite": args.lama.overwrite,
            "vis": args.vis,
            "max_frames": args.max_frames,
            "dilate_px": args.lama.dilate_px,
        },
        "software": {},
    }


def run_lama_video_inpaint(
    args: LamaVideoArgs,
    inpainter: SimpleLamaInpainter | None = None,
) -> LamaVideoOutputs:
    """Remove the hand/arm from every masked frame of a segment output with LaMa.

    ``args.masks_json`` is read from the ``segment`` stage; its referenced
    frame manifest is resolved to the original frames. The LaMa model is loaded
    once and reused for all frames. When ``inpainter`` is provided (e.g. a test
    double) it must expose ``inpaint(image, output_path, args, mask=...)``.
    """
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")
    if args.masks_json is None:
        raise ValueError("--lama-video.masks-json is required for video mode.")

    masks_json = args.masks_json.expanduser().resolve()
    mask_manifest = load_mask_manifest(masks_json)

    frames_json = Path(mask_manifest.source_frames_json)
    manifest = load_frame_manifest(frames_json)
    frame_by_index = {entry.index: entry for entry in manifest.entries}

    clip_stem = masks_json.parent.parent.name  # outputs/<clip>/segment/masks.json
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "inpaint"
    lama_dir = stage_dir / "lama"
    lama_vis_dir = stage_dir / "lama_vis" if args.vis else None

    if args.lama.overwrite:
        if lama_dir.exists():
            shutil.rmtree(lama_dir)
        if lama_vis_dir is not None and lama_vis_dir.exists():
            shutil.rmtree(lama_vis_dir)
    lama_dir.mkdir(parents=True, exist_ok=True)
    if lama_vis_dir is not None:
        lama_vis_dir.mkdir(parents=True, exist_ok=True)

    selected = mask_manifest.entries
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    active_inpainter = inpainter or SimpleLamaInpainter(args.lama)
    existing = _load_existing_entries(stage_dir / "lama.json")

    entries: list[LamaEntry] = []
    for mask_entry in selected:
        frame_entry = frame_by_index.get(mask_entry.index)
        if frame_entry is None:
            logger.warning(
                "[lama] skip frame {}: not in the frame manifest",
                mask_entry.index,
            )
            continue

        lama_filename = LAMA_FILENAME_PATTERN.format(mask_entry.index)
        vis_filename = (
            VIS_FILENAME_PATTERN.format(mask_entry.index)
            if lama_vis_dir is not None
            else None
        )
        lama_path = lama_dir / lama_filename
        vis_path = (
            lama_vis_dir / vis_filename
            if lama_vis_dir is not None and vis_filename is not None
            else None
        )

        prior = existing.get(mask_entry.index)
        if (
            not args.lama.overwrite
            and lama_path.exists()
            and prior is not None
            and (vis_path is None or vis_path.exists())
        ):
            entries.append(prior)
            continue

        frame_rgb = load_rgb_image(frame_entry.path)
        frame_h, frame_w = frame_rgb.shape[:2]

        if mask_entry.has_mask and mask_entry.mask_path is not None:
            mask = load_mask(mask_entry.mask_path)
            image = torch.from_numpy(frame_rgb).permute(2, 0, 1)
            active_inpainter.inpaint(image, lama_path, args.lama, mask=mask)
        else:
            save_image(frame_rgb, lama_path, overwrite=args.lama.overwrite)

        edited_rgb = load_rgb_image(lama_path)
        if edited_rgb.shape[:2] != (frame_h, frame_w):
            edited_rgb = _resize_rgb(edited_rgb, frame_w, frame_h)
            save_image(edited_rgb, lama_path, True)

        if vis_path is not None:
            save_side_by_side(
                frame_rgb,
                edited_rgb,
                vis_path,
                overwrite=args.lama.overwrite,
            )

        entries.append(
            LamaEntry(
                index=mask_entry.index,
                frame_filename=frame_entry.filename,
                timestamp_sec=frame_entry.timestamp_sec,
                has_mask=mask_entry.has_mask,
                mask_filename=mask_entry.mask_filename,
                lama_filename=lama_filename,
                vis_filename=vis_filename,
                height=frame_h,
                width=frame_w,
            )
        )

    _write_json(
        stage_dir / "lama.json",
        _lama_manifest_dict(args, mask_manifest, lama_dir, lama_vis_dir, entries),
    )
    _write_json(stage_dir / "config.json", _config_dict(args))

    masked_count = sum(1 for entry in entries if entry.has_mask)
    logger.info(
        "[lama] Done: processed={} masked={} vis={} out={}",
        len(entries),
        masked_count,
        args.vis,
        stage_dir,
    )

    return LamaVideoOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        lama_dir=lama_dir,
        lama_vis_dir=lama_vis_dir,
        lama_json_path=stage_dir / "lama.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
    )


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB array to the given resolution (LANCZOS, matching the
    reference EgoDataPipe behavior)."""

    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.LANCZOS))
