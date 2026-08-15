# composite

Depth-aware compositing of rendered robot arms into the inpainted scene — the
final stage of `v2r_pipeline` (blog: Ego2Robot's "depth composite" step).

## What it does

Takes three earlier stage outputs and writes the final robot-operation frames:

- `inpaint` — hand-removed background frames (`inpainted.json`);
- `depth` — DA3 relative scene depth of the inpainted frames (`depth.json`);
- `retarget` — robot-arm RGB + metric-depth renders from the mounted camera
  (`camera.json`).

For every frame the robot arm is pasted wherever its depth shows it is closer
to the camera than the scene surface behind it. Depth values live in two
different spaces (metric meters vs. DA3 relative units), so the step first
calibrates the affine mapping between them using the DA3 depth of the
*original* frames (the human hand occupied the same location as the robot arm,
so its depth anchors the arm in scene space), then applies the occlusion test.

## Install

```bash
cd composite
uv sync
```

## Usage

```bash
uv run python -m composite.cli \
  --inpainted-json ../outputs/0/inpaint/inpainted.json \
  --depth-json ../outputs/0/depth/depth.json \
  --camera-json ../outputs/0/retarget/camera.json \
  --calibration-depth-json ../outputs_orig/0/depth/depth.json \
  --masks-json ../outputs/0/segment/masks.json \
  --video
```

For `outputs/0/` this creates:

```
outputs/0/composite/
├── config.json       # effective run config
├── composite.json    # per-frame manifest (paths, stats, calibration)
├── composite.mp4     # muxed video of the composited frames
├── frames/           # composited robot-operation frames (000000.png, ...)
└── frames_vis/       # side-by-side (inpainted | robot | composite)
```

The calibration depth lives in its own output root (`outputs_orig/0/depth/`),
so the original depth stage under `outputs/0/depth/` is never touched.

## Depth matching

1. `arm` mask from the metric depth buffer (anything closer than the far plane).
2. Arm calibration — `DA3_orig ≈ slope·z_robot + intercept` — fitted on pixels
   where the robot arm overlaps the original-frame depth estimate. When
   `--masks-json` is given, the fit is restricted to pixels where the rendered
   arm overlaps the human-hand mask, which is much more reliable than fitting
   over the whole rendered arm.
3. Run calibration — `DA3_inp ≈ bg_slope·DA3_orig + bg_intercept` — fitted
   outside the arm region to align the two DA3 runs' normalizations.
4. Occlusion test in inpainted-depth space: show the arm iff
   `bg_slope·(slope·z_robot + intercept) + bg_intercept < DA3_inp + margin`
   (the margin biases toward showing the arm when arm and background depths are
   within depth noise of each other).

Fits are robust (median-split two-point), subsampled for speed, and median-
gap-filled over a short temporal window (per-frame fits are kept, only missing
fits inherit their neighbours). `--feather-px` softens the silhouette edge and
`--depth-margin-frac` controls the show-the-arm bias.

## Poisson blending

The robot render (MuJoCo) and the inpainted camera frames come from different
render pipelines, so the composited arm can look uniformly brighter or darker
than the scene — two different "exposures" stacked on top of each other.
Before the alpha mix, the visible arm is therefore fused into the scene with a
gradient-domain blend (`cv2.seamlessClone`, Poisson reconstruction): the arm
keeps its interior shading and texture while its overall brightness and color
are pushed to be continuous with the background at the silhouette, so the arm
reads as being lit by the scene instead of sitting on top of it. Disable with
`--no-poisson-blend` to paste the raw render with only the alpha feather.

Without `--calibration-depth-json` the step falls back to plain mask
compositing (no depth matching) and logs a warning. To enable it, produce the
calibration depth with the existing depth stage on the original frames:

```bash
cd depth && uv run python -m depth.cli --video.frames-json \
  ../outputs/0/process/frames.json --video.output-root ../outputs_orig
```

## Options

- `--inpainted-json` / `--depth-json` / `--camera-json` — stage manifests (required).
- `--calibration-depth-json` — DA3 depth of the original frames (enables depth matching).
- `--masks-json` — segment-stage hand masks (improves the arm calibration when
  given together with `--calibration-depth-json`).
- `--output-root <dir>` — root under which `<clip_stem>/composite/` is created (default `outputs`).
- `--vis` — write side-by-side debug PNGs (default on).
- `--video` — mux `composite.mp4` (default on).
- `--overwrite` — clear an existing `composite` workspace and re-run (default on).
- `--max-frames <n>` — limit the number of frames processed.
- `--feather-px <n>` / `--depth-margin-frac <f>` — silhouette feathering and depth margin.
- `--poisson-blend` / `--no-poisson-blend` — gradient-domain fusion of the arm
  into the scene before the alpha mix (default on).
- `--smooth-window <n>` / `--max-corr-samples <n>` / `--calibration-erode-px <n>` — calibration tuning.
