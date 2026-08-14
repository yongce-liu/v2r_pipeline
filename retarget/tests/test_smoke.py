"""Import/smoke tests for the retarget package."""

import importlib

import numpy as np


def test_package_imports() -> None:
    import retarget
    from retarget import (
        GMR_SRC_HUMAN,
        GMR_TGT_ROBOT,
    )

    assert GMR_SRC_HUMAN == "ego2robo_human"
    assert GMR_TGT_ROBOT == "ego2robo_robot"
    assert all(export in retarget.__all__ for export in __import__("retarget").__all__)


def test_submodules_import() -> None:
    for module in (
        "retarget.retargeter",
        "retarget.dex_retargeter",
        "retarget.gmr_retargeter",
    ):
        importlib.import_module(module)


def test_retarget_result_repr() -> None:
    """RetargetResult carries parallel qpos/action arrays."""
    from retarget import RetargetResult

    qpos = np.zeros(60, dtype=np.float64)
    action = np.zeros(53, dtype=np.float32)
    result = RetargetResult(qpos=qpos, action=action)
    assert result.qpos.shape == (60,)
    assert result.action.shape == (53,)
