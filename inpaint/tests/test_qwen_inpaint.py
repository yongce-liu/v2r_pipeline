"""Tests for Qwen image preprocessing without loading model weights."""

from __future__ import annotations

import torch
from PIL import Image

from inpaint.qwen_inpaint import QwenInpainter


def test_downsample_pil_to_720p() -> None:
    image = Image.new("RGB", (1920, 1080))

    resized, height, width = QwenInpainter._downsample_image(image, 2 / 3)

    assert resized.size == (1280, 720)
    assert (height, width) == (720, 1280)


def test_downsample_tensor_preserves_dtype() -> None:
    image = torch.zeros((3, 1080, 1920), dtype=torch.uint8)

    resized, height, width = QwenInpainter._downsample_image(image, 0.5)

    assert resized.shape == (3, 544, 960)
    assert resized.dtype == torch.uint8
    assert (height, width) == (544, 960)


def test_downsample_one_is_noop() -> None:
    image = Image.new("RGB", (1920, 1080))

    resized, height, width = QwenInpainter._downsample_image(image, 1.0)

    assert resized is image
    assert height == 1072
    assert width == 1920
