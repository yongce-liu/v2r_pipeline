"""Tests for source-type dispatch in :mod:`retarget.loaders` (no robot assets)."""

from pathlib import Path

import pytest

from retarget import SOURCE_LOADERS, iter_source_frames, known_source_types


def test_egodex_is_registered() -> None:
    assert "egodex" in SOURCE_LOADERS
    assert "egodex" in known_source_types()


def test_iter_source_frames_rejects_unknown_type(tmp_path: Path) -> None:
    cfg = {"human_root_name": "hip"}
    with pytest.raises(ValueError, match="Unknown source_type"):
        list(iter_source_frames("smpl", tmp_path / "x.hdf5", cfg))


def test_iter_source_frames_missing_file(tmp_path: Path) -> None:
    """Unknown source is rejected before touching the filesystem."""
    cfg = {"human_root_name": "hip"}
    with pytest.raises(ValueError, match="Unknown source_type"):
        list(iter_source_frames("smpl", tmp_path / "x.hdf5", cfg))


def test_loader_signature_has_no_fps(tmp_path: Path) -> None:
    """Loaders are fps-free; timestamps come from the process manifest."""
    import inspect

    from retarget import SOURCE_LOADERS

    for name, loader in SOURCE_LOADERS.items():
        assert "fps" not in inspect.signature(loader).parameters, name
