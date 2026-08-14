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

RTX 5090 NVFP4, reducing a 1920x1080 input to 1280x720 and using the release
benchmark settings:

```bash
uv run python -m inpaint.cli --command single \
  --single.image-path outputs/0/process/frames/frame_000000.png \
  --single.output-path outputs/0/inpaint_nvfp4_720p_test.png \
  --single.qwen.model-path \
    ckpts/Qwen-Image-Edit-2511-NVFP4/qwen_image_edit_2511_nvfp4.safetensors \
  --single.qwen.base-model-path ckpts/Qwen-Image-Edit-2511 \
  --single.qwen.downsample 0.6666667 \
  --single.qwen.steps 4 --single.qwen.true-cfg-scale 1.0
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
  reference release) — loaded with `QwenImageEditPlusPipeline.from_pretrained`.

> **Offline-only**: the loader passes `local_files_only=True`, so it never
> downloads from the Hub — every component must already be present on disk
> (download them manually, see links below). A partial checkout fails with an
> error naming the exact missing file.
- A **single-file ComfyUI checkpoint** (e.g.
  `ckpts/Qwen-Image-Edit-2511-NVFP4/qwen_image_edit_2511_nvfp4.safetensors`, the
  quantized release). The loader reads `_quantization_metadata`, constructs the
  transformer on the meta device, replaces the selected diffusers Linear modules
  with comfy-kitchen-backed quantized modules, and streams packed weights and
  scales into them. NVFP4 weights stay packed and activations are quantized for
  native FP4 matrix multiplication; the full transformer is never expanded to
  BF16.

## Offline checkpoints

To run without the Hub, download these into `ckpts/` manually:

- **Base diffusers model** — `Qwen/Qwen-Image-Edit-2511`. A BF16 run needs:
  - `transformer/diffusion_pytorch_model.safetensors.index.json`
  - `transformer/diffusion_pytorch_model-00001-of-00005.safetensors`
  - `transformer/diffusion_pytorch_model-00002-of-00005.safetensors`
  - `transformer/diffusion_pytorch_model-00003-of-00005.safetensors`
  - `transformer/diffusion_pytorch_model-00004-of-00005.safetensors`
  - `transformer/diffusion_pytorch_model-00005-of-00005.safetensors`

  The NVFP4 single-file path does **not** need these BF16 transformer shards. It
  only uses `transformer/config.json` plus the base model's `text_encoder/`,
  `vae/`, `scheduler/`, `tokenizer/`, `processor/`, and `model_index.json`.
- **Single-file quantized transformer** —
  `ckpts/Qwen-Image-Edit-2511-NVFP4/qwen_image_edit_2511_nvfp4.safetensors`.

The loader remains offline-only. For the quantized path it supplies the already
constructed transformer to `from_pretrained`, which prevents diffusers from
attempting to open the missing BF16 transformer shards.

## Options

- `--command single|video` — one image, or every frame of a video (default `video`).
- `--video.masks-json <path>` — the `segment` stage's `masks.json` (required for video mode).
- `--video.vis` / `--video.no-vis` — write original + edited side-by-side per frame (default on).
- `--video.max-frames <N>` — limit the number of frames processed (default all).
- `--video.qwen.model-path <path>` — Qwen model (diffusers dir or single-file safetensors).
- `--video.qwen.base-model-path <path>` — diffusers base model used with a single-file
  checkpoint (defaults to `ckpts/Qwen-Image-Edit-2511`).
- `--video.qwen.device auto|cuda[:N]|cpu` — torch device (default `auto`).
- `--video.qwen.cpu-offload` — component-level CPU offload (default on); needed
  on a 32GB RTX 5090 so the text encoder and transformer do not coexist in VRAM.
- `--video.qwen.downsample <ratio>` — inference dimensions relative to the input,
  rounded to multiples of 16; `0.6666667` maps 1920x1080 to 1280x720.
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
layout. NVFP4 weights remain packed `uint8`; block scales remain FP8 and are
consumed directly by comfy-kitchen.

Native NVFP4 requires a Blackwell GPU, PyTorch built with CUDA 12.8 or newer, and
the comfy-kitchen CUDA extension. `uv sync` installs the pinned
`torch==2.10.0+cu128` and `comfy-kitchen==0.2.31` environment. Startup validates
the GPU capability and the `scaled_mm_nvfp4` CUDA backend instead of silently
falling back to full-precision execution. This loader calls comfy-kitchen
directly, so ComfyUI's separate policy that disables its CUDA backend below
CUDA 13 does not apply here.
