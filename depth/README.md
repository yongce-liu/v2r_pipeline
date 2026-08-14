# depth

Depth Anything 3 depth estimation for `v2r_pipeline`, as a single image or as a
whole video (frame-by-frame, reading the `process` stage's `frames.json`).

## Install

```bash
cd depth
uv sync
```

## Usage

Single image:

```bash
uv run python -m depth.cli --command single \
  --single.image-path <input.png> \
  --single.output-dir <output_dir> \
  --single.da3.model-path <path/to/depth-anything-3-checkpoint> \
  --single.da3.device auto
```

Whole video (reads the `process` stage output; writes one aggregate file plus
per-frame depth maps):

```bash
uv run python -m depth.cli --command video \
  --video.frames-json outputs/0/process/frames.json \
  --video.vis \
  --video.da3.model-path <path/to/depth-anything-3-checkpoint>
```

For `outputs/0/process/frames.json` this creates, mirroring the `process`
stage layout:

```
outputs/0/depth/
├── config.json     # effective run config (same style as process)
├── depth.json      # per-frame depth manifest (index / paths / depth stats)
├── depth.npz       # single aggregate file (depth + intrinsics + timestamps)
├── depths/         # per-frame depth arrays (000000.npy, ...)
└── depths_vis/     # colorized depth maps (000000.png, ...), only when vis=True
```

The single aggregate file holds the whole clip's depths stacked as
`(frame_count, height, width)` float32 plus per-frame `(3, 3)` intrinsics and
timestamps. The `process` stage's `frames.json` (frame paths + timestamps) is
resolved via `depth.json` / `depths.npz` rather than by guessing filenames.

## Options

- `--command single|video` — one image, or every frame of a video (default `video`).
- `--video.frames-json <path>` — the `process` stage's `frames.json`.
- `--video.vis` / `--video.no-vis` — write colorized per-frame depth maps (default on).
- `--video.max-frames <N>` — limit the number of frames processed (default all).
- `--video.aggregate-format npz|pkl|json` — aggregate file format (default `npz`).
- `--video.da3.model-path <path>` — DA3 checkpoint (required for inference).
- `--video.da3.device auto|cuda[:N]|cpu` — torch device (default `auto`).
- `--video.da3.process-res <px>` — internal DA3 longer-side resolution (default 504).
- `--video.da3.overwrite` — clear existing depth outputs and re-run (default keeps
  prior per-frame outputs on re-runs).

Requires CUDA >= 12.8 (PyTorch cu128 wheels).
