"""Batch-run retarget + camera rendering across many EgoDex episodes.

Every ``*.hdf5`` under ``--input-root`` is retargeted with ``--ik-config`` and
rendered from the camera declared by the ik config's ``camera`` entry (RGB PNGs,
plus depth NPYs and a ``camera.json`` manifest when enabled). Outputs land in
``<output-root>/<rel-dir>/<episode>/retarget/``, mirroring the input path so
episodes that share a name across tasks never collide.

Episodes whose ``camera_rgb/000000.png`` already exists are skipped, so an
interrupted run resumes by re-invoking the same command.

Usage:
    retarget/.venv/bin/python scripts/retarget_batch.py \\
        --input-root inputs/test/play_reset_connect_four \\
        --ik-config configs/egodex_UnitreeG1InspireDfq_camera.json
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGET_PY = REPO_ROOT / "retarget" / ".venv" / "bin" / "python"


@dataclass
class BatchArgs:
    input_root: Path
    """Directory tree containing episode hdf5 files (scanned recursively)."""

    ik_config: Path
    """GMR IK config path (repo-root-relative, as in the retarget CLI)."""

    output_root: Path = REPO_ROOT / "outputs"
    """Root under which per-episode ``retarget/`` dirs are created."""

    task: str | None = None
    """Optional task subdirectory filter (e.g. ``play_reset_connect_four``).
    Episodes are then read from ``<input-root>/<task>/`` so outputs keep the
    task name and never collide with same-named episodes from other tasks."""

    workers: int = 1
    """Number of retarget processes to run concurrently."""


def episode_output_dir(args: BatchArgs, hdf5: Path) -> Path:
    """Mirror the episode's input-relative path under the output root."""
    rel = hdf5.relative_to(args.input_root)
    return args.output_root / rel.parent / rel.stem / "retarget"


def run_episode(args: BatchArgs, hdf5: Path) -> tuple[Path, str, str | None]:
    """Retarget one episode; returns (output_dir, status, error-or-None)."""
    output_dir = episode_output_dir(args, hdf5)
    if (output_dir / "camera_rgb" / "000000.png").is_file():
        return output_dir, "skipped", None

    proc = subprocess.run(
        [
            str(RETARGET_PY),
            "-m",
            "retarget",
            "--input",
            str(hdf5),
            "--ik-config",
            str(args.ik_config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return output_dir, "failed", (detail[-1] if detail else "unknown error")
    return output_dir, "done", None


def main() -> None:
    args = tyro.cli(BatchArgs)
    args.input_root = args.input_root.resolve()
    args.ik_config = args.ik_config.resolve()
    args.output_root = args.output_root.resolve()
    if not RETARGET_PY.is_file():
        raise FileNotFoundError(
            f"retarget interpreter not found: {RETARGET_PY} "
            "(run `cd retarget && uv sync` first)"
        )

    search_root = args.input_root / args.task if args.task else args.input_root
    episodes = sorted(search_root.rglob("*.hdf5"))
    if not episodes:
        raise RuntimeError(f"No hdf5 episodes found under {search_root}")
    logger.info(
        "batch retarget: {} episode(s) from {}, {} worker(s)",
        len(episodes),
        search_root,
        args.workers,
    )

    counts = {"done": 0, "skipped": 0, "failed": 0}
    failed: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_episode, args, hdf5): hdf5 for hdf5 in episodes}
        for future in as_completed(futures):
            hdf5 = futures[future]
            try:
                output_dir, status, error = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                counts["failed"] += 1
                failed.append((hdf5, str(exc)))
                logger.error("episode failed: {} ({})", hdf5, exc)
                continue
            counts[status] += 1
            if status == "done":
                logger.info("ok: {} -> {}", hdf5.name, output_dir)
            elif status == "failed":
                failed.append((hdf5, error))
                logger.error("episode failed: {} ({})", hdf5, error)

    logger.info(
        "batch complete: {} done, {} skipped, {} failed",
        counts["done"],
        counts["skipped"],
        counts["failed"],
    )
    for hdf5, error in failed[:20]:
        logger.warning("failed: {} ({})", hdf5, error)


if __name__ == "__main__":
    main()
