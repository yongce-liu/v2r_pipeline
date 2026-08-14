"""Tests for Qwen image preprocessing without loading model weights."""

from __future__ import annotations

import torch
from PIL import Image

from inpaint.qwen.inpainter import QwenInpainter


def test_downsample_pil_to_720p() -> None:
    image = Image.new("RGB", (1920, 1080))

    resized, height, width = QwenInpainter._downsample_image(image, 2 / 3)

    assert resized.size == (1280, 720)
    assert (height, width) == (720, 1280)


def test_downsample_tensor_to_pil() -> None:
    """Tensor input is always converted to a PIL image (uint8 RGB) and resized."""

    image = torch.zeros((3, 1080, 1920), dtype=torch.uint8)

    resized, height, width = QwenInpainter._downsample_image(image, 0.5)

    assert isinstance(resized, Image.Image)
    assert resized.mode == "RGB"
    assert resized.size == (960, 544)
    assert (height, width) == (544, 960)


def test_downsample_one_is_noop_when_already_aligned() -> None:
    """ratio=1.0 on an already-16-aligned image is returned unchanged."""

    image = Image.new("RGB", (1920, 1072))  # both dims already multiples of 16

    resized, height, width = QwenInpainter._downsample_image(image, 1.0)

    assert resized is image
    assert height == 1072
    assert width == 1920


def test_downsample_one_aligns_to_16() -> None:
    """ratio=1.0 still aligns dimensions to 16 (1080 -> 1072)."""

    image = Image.new("RGB", (1920, 1080))

    resized, height, width = QwenInpainter._downsample_image(image, 1.0)

    assert isinstance(resized, Image.Image)
    assert resized.size == (1920, 1072)
    assert (height, width) == (1072, 1920)
