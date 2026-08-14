"""Tests for the LaMa (big-lama) backend via simple-lama-inpainting.

The real checkpoint is never loaded (no model download). The ``SimpleLama``
wrapper is duck-typed with a fake, and the workflow is driven with a fake
inpainter — mirroring how ``test_workflow.py`` tests the Qwen backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from inpaint.lama.args import LamaInpaintArgs
from inpaint.lama.simple import SimpleLamaInpainter
from inpaint.lama.workflow import LamaVideoArgs, run_lama_video_inpaint
from tests.test_frames import build_process_layout
from tests.test_masks import build_segment_layout


class FakeLamaInpainter:
    """Duck-typed stand-in for the inpainter (no torch model needed).

    Records every call and paints the mask region red in the output so the
    result is distinguishable from a passthrough.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def inpaint(self, image, output_path: Path, args, mask=None) -> None:
        self.calls.append(len(self.calls))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = (
            image.permute(1, 2, 0).numpy()
            if isinstance(image, torch.Tensor)
            else np.asarray(image)
        )
        out = rgb.copy()
        if mask is not None:
            mb = np.asarray(mask) > 0
        else:
            # default black-mask extraction on the painted input
            mb = (rgb <= 8).all(axis=-1)
        out[mb] = (255, 0, 0)
        Image.fromarray(out).save(output_path)


class FakeSimpleLama:
    """Duck-typed stand-in for ``simple_lama_inpainting.SimpleLama``.

    ``SimpleLamaInpainter`` reaches it through the ``simple_lama`` property, so a
    test can inject this by assigning ``inpainter._simple_lama = fake``. Returns
    a fixed-size PIL result so the adapter's conversion/save path is exercised
    without the real TorchScript weights.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.device = torch.device("cpu")

    def __call__(self, image, mask):
        self.calls.append((image, mask))
        size = image.size if isinstance(image, Image.Image) else (8, 8)
        return Image.new("RGB", size, (0, 0, 255))


def _make_simple_inpainter(**kwargs) -> SimpleLamaInpainter:
    # Default to no dilation so conversion-contract tests aren't affected by
    # the default dilate_ratio=0.05.
    kwargs.setdefault("dilate_ratio", 0.0)
    inpainter = SimpleLamaInpainter(LamaInpaintArgs(**kwargs))
    inpainter._simple_lama = FakeSimpleLama()
    return inpainter


def test_simple_lama_inpainter_pil_contract(tmp_path: Path) -> None:
    """image/mask as PIL: clean RGB frame in, white mask marks the hole, PIL out."""

    image = Image.new("RGB", (10, 8), (120, 120, 120))
    mask = Image.new("L", (10, 8), 0)
    for y in range(2, 6):
        for x in range(2, 6):
            mask.putpixel((x, y), 255)

    inpainter = _make_simple_inpainter()
    out = tmp_path / "out.png"
    inpainter.inpaint(image, out, inpainter.args, mask=mask)

    fake = inpainter.simple_lama
    assert isinstance(fake, FakeSimpleLama)
    assert len(fake.calls) == 1
    passed_image, passed_mask = fake.calls[0]
    assert isinstance(passed_image, Image.Image) and passed_image.mode == "RGB"
    assert isinstance(passed_mask, Image.Image) and passed_mask.mode == "L"
    # mask contract: nonzero (255) = inpaint region
    assert passed_mask.getpixel((3, 3)) == 255
    assert passed_mask.getpixel((0, 0)) == 0
    assert out.exists()


def test_simple_lama_inpainter_ndarray_and_tensor_inputs(tmp_path: Path) -> None:
    """image as ndarray and as CHW uint8 tensor both reach SimpleLama as PIL RGB."""

    rgb = np.full((8, 10, 3), 90, dtype=np.uint8)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    out = tmp_path / "out.png"

    inpainter = _make_simple_inpainter()
    inpainter.inpaint(rgb, out, inpainter.args, mask=mask)
    assert isinstance(inpainter.simple_lama.calls[0][0], Image.Image)

    inpainter2 = _make_simple_inpainter()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W) uint8 CHW
    inpainter2.inpaint(tensor, out, inpainter2.args, mask=torch.from_numpy(mask))
    passed_image, passed_mask = inpainter2.simple_lama.calls[0]
    assert isinstance(passed_image, Image.Image) and passed_image.mode == "RGB"
    assert isinstance(passed_mask, Image.Image) and passed_mask.mode == "L"


def test_simple_lama_inpainter_requires_mask(tmp_path: Path) -> None:
    """The adapter must never fall back to a painted-input heuristic."""

    inpainter = _make_simple_inpainter()
    with pytest.raises(ValueError, match="mask-path"):
        inpainter.inpaint(None, None, inpainter.args, mask=None)  # type: ignore[arg-type]


def test_lama_video_workflow(tmp_path: Path) -> None:
    """The workflow drives the inpainter per masked frame and writes manifests."""

    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    fake = FakeLamaInpainter()

    outputs = run_lama_video_inpaint(
        LamaVideoArgs(masks_json=masks_json, output_root=tmp_path, vis=True),
        inpainter=fake,
    )

    assert outputs.stage_dir == tmp_path / "0" / "inpaint"
    assert outputs.inpainted_dir.exists()
    assert outputs.inpainted_vis_dir is not None and outputs.inpainted_vis_dir.exists()

    inpainted = sorted(p.name for p in outputs.inpainted_dir.glob("*.png"))
    vis = sorted(p.name for p in outputs.inpainted_vis_dir.glob("*.png"))
    assert inpainted == ["000000.png", "000001.png", "000002.png"]
    assert vis == ["000000.png", "000001.png", "000002.png"]

    manifest = json.loads(outputs.inpainted_json_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "lama"
    assert manifest["masked_count"] == 2  # frame 2 has no mask
    # frames without a mask are copied through
    assert outputs.inpainted_dir.joinpath("000002.png").exists()

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["backend"] == "lama"
    assert config["inpaint"]["backend"] == "simple"
    assert config["package"]["name"] == "inpaint.lama"


def test_simple_lama_inpainter_overwrite_skip(tmp_path: Path) -> None:
    """With overwrite=False an existing output is reused (no SimpleLama call)."""

    out = tmp_path / "out.png"
    out.write_bytes(b"existing")

    inpainter = _make_simple_inpainter()
    args = LamaInpaintArgs(overwrite=False)
    inpainter.inpaint(_dummy_image(), out, args, mask=_dummy_mask())

    assert inpainter.simple_lama.calls == []  # never reached the model


def test_dilation_ratio_scales_with_frame_size() -> None:
    """The outward dilation is a fraction of the frame's shorter side."""

    # 1080p: 0.05 * 1080 = 54 px; 540p half that; 4K (2160p) twice that.
    assert (
        SimpleLamaInpainter._dilation_px_for_frame(min_side_px=1080, dilate_ratio=0.05)
        == 54
    )
    assert (
        SimpleLamaInpainter._dilation_px_for_frame(min_side_px=540, dilate_ratio=0.05)
        == 27
    )
    assert (
        SimpleLamaInpainter._dilation_px_for_frame(min_side_px=2160, dilate_ratio=0.05)
        == 108
    )
    # tiny frame still dilates at least 1 px
    assert (
        SimpleLamaInpainter._dilation_px_for_frame(min_side_px=10, dilate_ratio=0.001)
        == 1
    )
    # empty frame or disabled -> 0
    assert (
        SimpleLamaInpainter._dilation_px_for_frame(min_side_px=0, dilate_ratio=0.05)
        == 0
    )
    assert (
        SimpleLamaInpainter._dilation_px_for_frame(min_side_px=1080, dilate_ratio=0.0)
        == 0
    )


def test_dilate_mask_expands_outward() -> None:
    """Dilation grows the mask outward by the frame-relative amount."""

    # 200x200 frame, block 28..172 -> k = 0.04 * 200 = 8 px.
    mask = Image.new("L", (200, 200), 0)
    for y in range(28, 172):
        for x in range(28, 172):
            mask.putpixel((x, y), 255)
    args = LamaInpaintArgs(dilate_ratio=0.04)

    dilated = SimpleLamaInpainter._dilate_mask(mask, args)
    assert dilated.mode == "L"
    assert dilated.size == mask.size
    k = SimpleLamaInpainter._dilation_px_for_frame(200, args.dilate_ratio)
    assert k == 8  # 0.04 * 200
    # the block grew outward by ~k px: the left edge moved from 28 toward 28-k
    # (assert on the edge, not the rounded corner)
    assert dilated.getpixel((28 - k, 28)) > 0  # left edge expanded by ~k
    assert dilated.getpixel((28 - k - 3, 28)) == 0  # still outside
    assert dilated.getpixel((28, 28 - k)) > 0  # top edge expanded by ~k


def test_dilate_mask_disabled_is_noop() -> None:
    """dilate_ratio=0 leaves the mask unchanged."""

    mask = Image.new("L", (16, 16), 0)
    mask.putpixel((8, 8), 255)
    args = LamaInpaintArgs(dilate_ratio=0.0)

    dilated = SimpleLamaInpainter._dilate_mask(mask, args)
    assert np.array_equal(np.asarray(dilated), np.asarray(mask))


def test_inpaint_passes_dilated_mask(tmp_path: Path) -> None:
    """With dilate_ratio>0 the mask handed to SimpleLama is the dilated one."""

    inpainter = _make_simple_inpainter(dilate_ratio=0.04)
    out = tmp_path / "out.png"
    # same realistic mask as above (200x200 frame, 144x144 block) -> k = 8
    mask = Image.new("L", (200, 200), 0)
    for y in range(28, 172):
        for x in range(28, 172):
            mask.putpixel((x, y), 255)

    inpainter.inpaint(_dummy_image(), out, inpainter.args, mask=mask)

    passed_mask = inpainter.simple_lama.calls[0][1]
    assert isinstance(passed_mask, Image.Image)
    assert passed_mask.mode == "L"
    # the dilated mask has more nonzero pixels than the original 144x144 block
    assert int(np.asarray(passed_mask).sum() / 255) > 144 * 144


def test_inpaint_repeat_runs_multiple_passes(tmp_path: Path) -> None:
    """repeat=N re-feeds the previous output with the SAME mask N times."""

    inpainter = _make_simple_inpainter()
    out = tmp_path / "out.png"
    inpainter.inpaint(
        _dummy_image(),
        out,
        LamaInpaintArgs(repeat=3),
        mask=_dummy_mask(),
    )

    fake = inpainter.simple_lama
    assert len(fake.calls) == 3
    # every pass sees a PIL RGB frame and the identical mask (unchanged size)
    first_mask = np.asarray(fake.calls[0][1])
    for passed_image, passed_mask in fake.calls:
        assert isinstance(passed_image, Image.Image) and passed_image.mode == "RGB"
        assert isinstance(passed_mask, Image.Image) and passed_mask.mode == "L"
        assert np.array_equal(np.asarray(passed_mask), first_mask)
    assert out.exists()


def test_lama_video_workflow_records_dilation_and_repeat(tmp_path: Path) -> None:
    """config.json records the effective dilate_ratio and repeat."""

    masks_json, _ = _build_segmented_clip(tmp_path, frame_count=3)
    run_lama_video_inpaint(
        LamaVideoArgs(
            masks_json=masks_json,
            output_root=tmp_path,
            lama=LamaInpaintArgs(dilate_ratio=0.07, repeat=2),
        ),
        inpainter=FakeLamaInpainter(),
    )

    config = json.loads(
        (tmp_path / "0" / "inpaint" / "config.json").read_text(encoding="utf-8")
    )
    assert config["inpaint"]["dilate_ratio"] == 0.07
    assert config["inpaint"]["repeat"] == 2


def _dummy_image():
    return Image.new("RGB", (8, 8), (10, 10, 10))


def _dummy_mask():
    m = Image.new("L", (8, 8), 0)
    m.putpixel((1, 1), 255)
    return m


def _build_segmented_clip(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    """Create a full process + segment layout; return (masks_json, clip_root)."""

    build_process_layout(tmp_path, frame_count=frame_count)
    return build_segment_layout(tmp_path, frame_count=frame_count)
