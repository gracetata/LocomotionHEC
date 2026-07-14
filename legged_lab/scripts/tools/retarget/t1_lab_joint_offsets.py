"""T1 joint zero-offset helpers for GMR-to-Lab conversion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# Lab T1 URDF now follows the GMR T1 mocap XML zero-pose convention, so no
# GMR-to-Lab joint angle offset compensation is required.
T1_LAB_BAKED_ZERO_OFFSETS = {}


def compensate_t1_lab_baked_zero_offsets(dof_pos: np.ndarray, dof_names: Sequence[str]) -> np.ndarray:
    """Return T1 joint angles in the Lab/GMR shared zero convention."""

    compensated = np.asarray(dof_pos).copy()
    for joint_name, baked_offset in T1_LAB_BAKED_ZERO_OFFSETS.items():
        if joint_name in dof_names:
            joint_index = dof_names.index(joint_name)
            compensated[:, joint_index] -= baked_offset
    return compensated
