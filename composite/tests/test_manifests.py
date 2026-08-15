"""Unit tests for path resolution in composite.manifests."""

from __future__ import annotations

from pathlib import Path

from composite.manifests import _resolve


def test_resolve_absolute_path_passes_through(tmp_path: Path) -> None:
    target = tmp_path / "depths"
    target.mkdir()
    assert _resolve(str(target), tmp_path) == target


def test_resolve_relative_to_manifest(tmp_path: Path) -> None:
    depths = tmp_path / "depths"
    depths.mkdir()
    assert _resolve("depths", tmp_path) == depths


def test_resolve_relative_to_cwd(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    depths = root / "outputs/0/depth_orig/depths"
    depths.mkdir(parents=True)
    monkeypatch.chdir(root)
    manifest_dir = root / "outputs/0/depth_orig"
    assert _resolve("outputs/0/depth_orig/depths", manifest_dir) == depths


def test_resolve_manifest_relative_wins_over_cwd(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    both = root / "outputs/0/depth_orig"
    (both / "depths").mkdir(parents=True)
    (root / "depths").mkdir()
    monkeypatch.chdir(root)
    assert _resolve("depths", both) == both / "depths"
