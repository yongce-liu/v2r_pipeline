# segment

SAM3 hand-mask segmentation for the `v2r_pipeline`. Two modes:

- **single** — segment one image.
- **video** — frame-by-frame segmentation of a whole video, reading the
  `process` stage's `frames.json` (the model is loaded once and reused for
  every frame).

## Install

```bash
uv sync
```

Requires CUDA >= 12.8 (PyTorch cu128 wheels).

## Usage

### Single image

```bash
uv run python -m segment.cli --command single \
  --single.image-path frame.png \
  --single.output-dir out \
  --single.sam-mask.checkpoint ckpts/sam3/sam3.pt
```

Writes `hand_seg.png` (mask) and `hand_seg_vis.jpg` (overlay) into `out/`.

### Video (frame-by-frame)

```bash
uv run python -m segment.cli --command video \
  --video.frames-json outputs/0/process/frames.json \
  --video.sam-mask.checkpoint ckpts/sam3/sam3.pt \
  --video.vis
```

For `outputs/0/process/frames.json` this creates (mirroring the `process`
stage layout):

```
outputs/0/segment/
├── config.json      # effective run config (same style as process)
├── masks.json       # per-frame mask manifest (index / paths / bbox / area)
├── masks/
│   ├── 000000.png
│   └── ...
└── masks_vis/       # only when --video.vis is on
    ├── 000000.jpg
    └── ...
```

Each `masks.json` entry records the frame index, source frame filename, mask
filename, optional overlay filename, whether a mask was found, the number of
detected instances, and the mask **position** as an inclusive bounding box
(`min_row`/`min_col`/`max_row`/`max_col`) plus foreground pixel `area`.

## Options (video mode)

- `--video.frames-json <path>` — path to the `process` stage `frames.json`.
- `--video.output-root <dir>` — root under which `<clip>/segment/` is created
  (default `outputs`, same convention as `process`).
- `--video.vis` / `--video.no-vis` — write original frame + mask overlay images
  (default on).
- `--video.max-frames <N>` — limit the number of frames processed.
- `--video.sam-mask.*` — SAM3 settings: `checkpoint`, `allow-hf-download`,
  `device`, `text-prompt` (default `人手`), `score-threshold`, `overlay-alpha`,
  `mask-color-rgb`, `overwrite`.

Downstream stages read `masks.json` (per-frame mask paths + positions) rather
than guessing filenames.
