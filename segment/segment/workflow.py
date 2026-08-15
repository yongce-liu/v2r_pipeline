"""Video-mode (frame-by-frame) hand-mask segmentation.

Reads the ``process`` stage's ``frames.json``, segments every frame with SAM3
(frame-by-frame "video mode" — the model is loaded once and reused), and writes
a per-frame mask image plus a ``masks.json`` manifest. When ``vis`` is enabled
an overlay (original frame + mask) image is written for every frame too.

Output layout mirrors the ``process`` stage:

.. code-block:: text

    outputs/<clip>/segment/
    ├── config.json     # effective run config (same style as process)
    ├── masks.json      # per-frame mask manifest (index / paths / bbox / area)
    ├── masks/
    │   ├── 000000.png
    │   └── ...
    └── masks_vis/      # only when vis=True
        ├── 000000.jpg
        └── ...
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from segment import __version__
from segment.frames import FrameManifest, load_frame_manifest
from segment.media import (
    load_rgb_image,
    mask_stats,
    save_mask,
    save_overlay,
)
from segment.sam_mask import Sam3MaskGenerator, SamMaskArgs

MASK_FILENAME_PATTERN = "{:06d}.png"
VIS_FILENAME_PATTERN = "{:06d}.jpg"


@dataclass
class SegmentVideoArgs:
    """Arguments for frame-by-frame segmentation of a whole video."""

    frames_json: Path
    """Path to the ``process`` stage's ``frames.json`` (the frame manifest)."""

    output_root: Path = Path(__file__).parents[2] / "outputs"
    """Root under which ``<clip_stem>/segment/`` is created."""

    vis: bool = True
    """Write an original frame + mask overlay image for every processed frame."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames in the manifest)."""

    sam_mask: SamMaskArgs = field(default_factory=SamMaskArgs)
    """SAM3 segmentation settings (checkpoint, prompt, thresholds, ...)."""


@dataclass(frozen=True)
class MaskEntry:
    """Per-frame mask record written into ``masks.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    mask_filename: str | None
    vis_filename: str | None
    has_mask: bool
    instance_count: int
    area: int
    bbox: dict | None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "timestamp_sec": self.timestamp_sec,
            "mask_filename": self.mask_filename,
            "vis_filename": self.vis_filename,
            "has_mask": self.has_mask,
            "instance_count": self.instance_count,
            "area": self.area,
            "bbox": self.bbox,
        }


@dataclass(frozen=True)
class SegmentVideoOutputs:
    """Everything produced by one video segmentation run."""

    clip_root: Path
    stage_dir: Path
    masks_dir: Path
    masks_vis_dir: Path | None
    masks_json_path: Path
    config_json_path: Path
    entries: list[MaskEntry]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_existing_entries(masks_json: Path) -> dict[int, MaskEntry]:
    """Reuse previously written mask entries (idempotent non-overwrite runs)."""

    if not masks_json.exists():
        return {}
    try:
        data = json.loads(masks_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    existing: dict[int, MaskEntry] = {}
    for raw in data.get("entries", []):
        try:
            existing[int(raw["index"])] = MaskEntry(
                index=int(raw["index"]),
                frame_filename=raw["frame_filename"],
                timestamp_sec=raw.get("timestamp_sec"),
                mask_filename=raw.get("mask_filename"),
                vis_filename=raw.get("vis_filename"),
                has_mask=bool(raw.get("has_mask")),
                instance_count=int(raw.get("instance_count", 0)),
                area=int(raw.get("area", 0)),
                bbox=raw.get("bbox"),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return existing


def _mask_manifest_dict(
    manifest: FrameManifest,
    args: SegmentVideoArgs,
    masks_dir: Path,
    masks_vis_dir: Path | None,
    entries: list[MaskEntry],
) -> dict:
    return {
        "schema_version": "1.0",
        "stage": "segment",
        "source_frames_json": str(args.frames_json.expanduser().resolve()),
        "source_video": manifest.source_video,
        "fps": manifest.fps,
        "width": manifest.width,
        "height": manifest.height,
        "frame_format": manifest.format,
        "frame_count": manifest.frame_count,
        "frames_dir": str(masks_dir),
        "processed_count": len(entries),
        "masks_dir": str(masks_dir),
        "masks_vis_dir": str(masks_vis_dir) if masks_vis_dir is not None else None,
        "mask_format": "png",
        "vis_enabled": args.vis,
        "entries": [entry.to_dict() for entry in entries],
    }


def _config_dict(
    args: SegmentVideoArgs,
    manifest: FrameManifest,
) -> dict:
    checkpoint = (
        str(args.sam_mask.checkpoint.expanduser())
        if args.sam_mask.checkpoint is not None
        else None
    )
    return {
        "package": {"name": "segment", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "frames_json": str(args.frames_json.expanduser().resolve()),
            "source_video": manifest.source_video,
            "fps": manifest.fps,
            "width": manifest.width,
            "height": manifest.height,
            "frame_format": manifest.format,
            "frame_count": manifest.frame_count,
        },
        "segment": {
            "checkpoint": checkpoint,
            "allow_hf_download": args.sam_mask.allow_hf_download,
            "device": args.sam_mask.device,
            "text_prompt": args.sam_mask.text_prompt,
            "score_threshold": args.sam_mask.score_threshold,
            "overlay_alpha": args.sam_mask.overlay_alpha,
            "mask_color_rgb": list(args.sam_mask.mask_color_rgb),
            "vis": args.vis,
            "overwrite": args.sam_mask.overwrite,
            "max_frames": args.max_frames,
        },
        "software": {},
    }


def run_video_segment(
    args: SegmentVideoArgs,
    generator: Sam3MaskGenerator | None = None,
) -> SegmentVideoOutputs:
    """Segment every frame of a process stage output, frame by frame.

    ``args.frames_json`` is read from the ``process`` stage. The SAM3 model is
    loaded once and reused for all frames. When ``generator`` is provided (e.g.
    a test double), it must expose ``segment(frame_rgb, text_prompt) -> (mask,
    instance_count)``.
    """
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")

    frames_json = args.frames_json.expanduser().resolve()
    manifest = load_frame_manifest(frames_json)

    clip_stem = frames_json.parent.parent.name  # outputs/<clip>/process/frames.json
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "segment"
    masks_dir = stage_dir / "masks"
    masks_vis_dir = stage_dir / "masks_vis" if args.vis else None

    if args.sam_mask.overwrite:
        if masks_dir.exists():
            shutil.rmtree(masks_dir)
        if masks_vis_dir is not None and masks_vis_dir.exists():
            shutil.rmtree(masks_vis_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if masks_vis_dir is not None:
        masks_vis_dir.mkdir(parents=True, exist_ok=True)

    selected = manifest.entries
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    active_generator = generator or Sam3MaskGenerator(args.sam_mask)
    existing = _load_existing_entries(stage_dir / "masks.json")

    entries: list[MaskEntry] = []
    for frame in selected:
        mask_filename = MASK_FILENAME_PATTERN.format(frame.index)
        vis_filename = (
            VIS_FILENAME_PATTERN.format(frame.index)
            if masks_vis_dir is not None
            else None
        )
        mask_path = masks_dir / mask_filename
        vis_path = (
            masks_vis_dir / vis_filename
            if masks_vis_dir is not None and vis_filename is not None
            else None
        )

        prior = existing.get(frame.index)
        if (
            not args.sam_mask.overwrite
            and mask_path.exists()
            and prior is not None
            and (vis_path is None or vis_path.exists())
        ):
            entries.append(
                MaskEntry(
                    index=prior.index,
                    frame_filename=prior.frame_filename,
                    timestamp_sec=prior.timestamp_sec,
                    mask_filename=prior.mask_filename,
                    vis_filename=prior.vis_filename,
                    has_mask=prior.has_mask,
                    instance_count=prior.instance_count,
                    area=prior.area,
                    bbox=prior.bbox,
                )
            )
            continue

        frame_rgb = load_rgb_image(frame.path)
        mask, instance_count = active_generator.segment(
            frame_rgb, args.sam_mask.text_prompt
        )
        stats = mask_stats(mask)

        save_mask(mask, mask_path, args.sam_mask.overwrite)
        if vis_path is not None:
            save_overlay(
                frame_rgb,
                mask,
                vis_path,
                alpha=args.sam_mask.overlay_alpha,
                mask_color_rgb=args.sam_mask.mask_color_rgb,
                overwrite=args.sam_mask.overwrite,
            )

        entries.append(
            MaskEntry(
                index=frame.index,
                frame_filename=frame.frame_filename,
                timestamp_sec=frame.timestamp_sec,
                mask_filename=mask_filename,
                vis_filename=vis_filename,
                has_mask=stats.has_mask,
                instance_count=instance_count,
                area=stats.area,
                bbox=stats.to_dict()["bbox"],
            )
        )

    _write_json(
        stage_dir / "masks.json",
        _mask_manifest_dict(manifest, args, masks_dir, masks_vis_dir, entries),
    )
    _write_json(stage_dir / "config.json", _config_dict(args, manifest))

    with_mask = sum(1 for entry in entries if entry.has_mask)
    logger.info(
        "[segment] Done: processed={} has_mask={} vis={} out={}",
        len(entries),
        with_mask,
        args.vis,
        stage_dir,
    )

    return SegmentVideoOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        masks_dir=masks_dir,
        masks_vis_dir=masks_vis_dir,
        masks_json_path=stage_dir / "masks.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
    )
