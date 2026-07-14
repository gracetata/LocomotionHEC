from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

@configclass 
class MotionDataTermCfg:
    """
    Configuration for the motion data term in the motion data manager.
    """
    
    weight: float = 1.0
    """Weight of this term in the motion data manager."""
    
    motion_data_dir: str = MISSING
    """Directory containing motion data files.
    
    Only supports reading .pkl files from this directory.
    """
    
    motion_data_weights: dict[str, float] = MISSING
    """Weights for the motion data in this term."""

    root_rot_order: str = "wxyz"
    """Quaternion order stored in ``root_rot``.

    Legged Lab reference consumers use ``wxyz`` internally. Some GMR/MuJoCo
    conversion outputs store root quaternions as ``xyzw`` and must opt in to
    conversion here instead of being silently interpreted as ``wxyz``.
    """

    target_dof_names: list[str] | None = None
    """Optional target DoF order for reference motion columns.

    When motion pickles contain ``dof_names``, this reorders ``dof_pos`` by
    name before any reference state, demo observation, or animation uses it.
    """

    strict_dof_names: bool = False
    """Whether to fail if ``target_dof_names`` cannot be matched exactly."""
    
