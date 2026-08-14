# inpaint

Qwen-Image-Edit arm-removal inpainting for `v2r_pipeline`, run per frame and
reading the `segment` stage's `masks.json` (which references the `process`
stage's frame manifest). For every frame the SAM3 hand/arm mask is painted over
the frame with a solid color, Qwen-Image-Edit removes the hand/arm and
completes the background, and the result is written to `outputs/<clip>/inpaint/`.

## Install

```bash
cd 02_inpaint
uv sync
```

## Usage

Whole video (frame-by-frame, reading the `segment` stage output):

```bash
uv run python -m inpaint.cli --command video \
  --video.masks-json outputs/0/segment/masks.json \
  --video.vis --video.qwen.model-path ckpts/Qwen-Image-Edit-2511
```

Single masked frame (image already painted with the mask color):

```bash
uv run python -m inpaint.cli --command single \
  --single.image-path frame.png --single.output-path out.png \
  --single.qwen.model-path ckpts/Qwen-Image-Edit-2511
```

For `outputs/0/segment/masks.json` this creates, mirroring the earlier stages:

```
outputs/0/inpaint/
├── config.json     # effective run config (same style as process)
├── inpainted.json  # per-frame inpaint manifest (index / paths / prompt)
├── inpainted/      # edited frames (inpainted_000000.png, ...)
└── inpainted_vis/  # original + edited side-by-side (vis_000000.png, ...), only when vis=True
```

Frames whose mask is empty (no hand/arm detected) are copied through unchanged,
so the output always has one image per source frame.

## Models

The Qwen model can be either of:

- A **diffusers directory** (e.g. `ckpts/Qwen-Image-Edit-2511`, the `bf16`
  reference release) — loaded with
  `QwenImageEditPlusPipeline.from_pretrained`.
- A **single-file ComfyUI checkpoint** (e.g.
  `ckpts/Qwen-Image-Edit-2511-FP8/Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors` or
  `ckpts/Qwen-Image-Edit-2511-NVFP4/qwen_image_edit_2511_nvfp4.safetensors`, the
  quantized releases). These are state dicts of the *same* `QwenImageTransformer2DModel`
  as the diffusers base model (verified: identical 1933 key sets). We load the
  diffusers reference base model (`--qwen.base-model-path ckpts/Qwen-Image-Edit-2511`)
  once, then `load_state_dict` the quantized transformer onto it — torch's
  in-place `copy_` converts `float8_e4m3fn` (or packed `uint8` NVFP4) weights to
  `bfloat16` in place.

## Options

- `--command single|video` — one image, or every frame of a video (default `video`).
- `--video.masks-json <path>` — the `segment` stage's `masks.json` (required for video mode).
- `--video.vis` / `--video.no-vis` — write original + edited side-by-side per frame (default on).
- `--video.max-frames <N>` — limit the number of frames processed (default all).
- `--video.qwen.model-path <path>` — Qwen model (diffusers dir or single-file safetensors).
- `--video.qwen.base-model-path <path>` — diffusers base model used with a single-file
  checkpoint (defaults to `ckpts/Qwen-Image-Edit-2511`).
- `--video.qwen.device auto|cuda[:N]|cpu` — torch device (default `auto`).
- `--video.qwen.prompt <str>` — instruction prompt (defaults to the hand/arm removal prompt).
- `--video.qwen.negative-prompt <str>` — negative prompt (defaults to hands/arms/skin).
- `--video.qwen.steps <N>` — inference steps (default `50`).
- `--video.qwen.true-cfg-scale <float>` — true-CFG scale (default `6.0`).
- `--video.qwen.seed <int>` — random seed (default `42`).
- `--video.qwen.overwrite` / `--video.qwen.no-overwrite` — recompute vs. reuse outputs (default recompute).

## Notes on the NVFP4 checkpoint

`ckpts/Qwen-Image-Edit-2511-NVFP4` ships a single
`qwen_image_edit_2511_nvfp4.safetensors` (≈19.8 GB) — a ComfyUI state dict of
the `Qwen-Image-Edit-2511` transformer with NVFP4-quantized weights. It is not a
diffusers directory (no `model_index.json` / tokenizer / VAE), so it cannot be
passed to `from_pretrained` on its own. The loader in `qwen_model.py` maps its
`transformer_blocks.*` keys onto the diffusers `QwenImageTransformer2DModel`
layout. NVFP4 values (float8 matrix/scales) are loaded as raw `uint8` data.
