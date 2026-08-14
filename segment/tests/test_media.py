"""Tests for mask geometry helpers."""

from __future__ import annotations

import numpy as np

from segment.media import load_mask, mask_is_empty, mask_stats


def test_mask_stats_with_foreground() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255

    stats = mask_stats(mask)
    assert stats.has_mask
    assert stats.area == 3 * 4
    assert stats.bbox == (2, 3, 4, 6)
    assert stats.to_dict() == {
        "has_mask": True,
        "area": 12,
        "bbox": {"min_row": 2, "min_col": 3, "max_row": 4, "max_col": 6},
    }


def test_mask_stats_empty() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    stats = mask_stats(mask)
    assert not stats.has_mask
    assert stats.area == 0
    assert stats.bbox is None
    assert stats.to_dict() == {"has_mask": False, "area": 0, "bbox": None}


def test_mask_is_empty() -> None:
    assert mask_is_empty(np.zeros((4, 4), dtype=np.uint8))
    assert not mask_is_empty(np.ones((4, 4), dtype=np.uint8))


def test_load_mask_roundtrip(tmp_path) -> None:
    from PIL import Image

    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)

    loaded = load_mask(path)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, mask)
