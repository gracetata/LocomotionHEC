"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_rl_lab
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets import unitree_actuators


def _repair_s3_g1_29dof_usd(stage, cfg):
    asset_path = getattr(cfg, "asset_path", "")
    if "s3_g1_29dof" not in asset_path:
        return

    from pathlib import Path

    import numpy as np
    from pxr import Gf, UsdPhysics

    try:
        import trimesh
    except ImportError:
        trimesh = None

    asset_dir = Path(asset_path).resolve().parent
    gripper_mesh_dir = asset_dir / "2F85_assets"

    def _mesh_density(mesh_name, mass):
        if trimesh is None:
            return None
        mesh_path = gripper_mesh_dir / f"{mesh_name}.stl"
        if not mesh_path.is_file():
            return None
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        mesh.apply_scale(0.001)
        volume = abs(float(mesh.volume))
        if volume <= 0.0:
            return None
        return mass / volume

    known_density_values = [
        value
        for value in (
            _mesh_density("base", 0.777441),
            _mesh_density("driver", 0.00899563),
            _mesh_density("coupler", 0.0140974),
            _mesh_density("follower", 0.0125222),
            _mesh_density("pad", 0.0035),
            _mesh_density("spring_link", 0.0221642),
        )
        if value is not None
    ]
    default_density = float(np.median(known_density_values)) if known_density_values else 1500.0
    pad_density = _mesh_density("pad", 0.0035) or default_density

    def _fallback_inertia(mass, extents):
        extents = np.maximum(np.asarray(extents, dtype=float), 1.0e-4)
        return np.array(
            [
                mass * (extents[1] ** 2 + extents[2] ** 2) / 12.0,
                mass * (extents[0] ** 2 + extents[2] ** 2) / 12.0,
                mass * (extents[0] ** 2 + extents[1] ** 2) / 12.0,
            ],
            dtype=float,
        )

    def _mass_properties(mesh_name, density):
        if trimesh is None:
            return 1.0e-4, np.zeros(3), np.array([1.0e-8, 1.0e-8, 1.0e-8])
        mesh_path = gripper_mesh_dir / f"{mesh_name}.stl"
        if not mesh_path.is_file():
            return 1.0e-4, np.zeros(3), np.array([1.0e-8, 1.0e-8, 1.0e-8])
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        mesh.apply_scale(0.001)
        volume = abs(float(mesh.volume))
        mass = max(volume * density, 1.0e-6)
        center = np.asarray(mesh.center_mass, dtype=float)
        inertia = np.asarray(mesh.moment_inertia, dtype=float) * density
        diagonal = np.diag(inertia)
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            diagonal = _fallback_inertia(mass, mesh.extents)
        return mass, center, np.maximum(diagonal, 1.0e-10)

    missing_specs = {
        "base_mount": _mass_properties("base_mount", default_density),
        "silicone_pad": _mass_properties("silicone_pad", pad_density),
    }

    fixed_count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        prim_name = prim.GetName()
        spec_key = None
        if prim_name.endswith("base_mount"):
            spec_key = "base_mount"
        elif prim_name.endswith("silicone_pad"):
            spec_key = "silicone_pad"
        if spec_key is None:
            continue

        mass_api = UsdPhysics.MassAPI.Apply(prim)
        current_mass = mass_api.GetMassAttr().Get()
        current_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        needs_fix = current_mass is None or current_mass <= 0.0
        if current_inertia is not None:
            needs_fix = needs_fix or any(value <= 0.0 for value in current_inertia)
        if not needs_fix:
            continue

        mass, center, diagonal = missing_specs[spec_key]
        mass_api.CreateMassAttr(float(mass))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(float(center[0]), float(center[1]), float(center[2])))
        mass_api.CreateDiagonalInertiaAttr(
            Gf.Vec3f(float(diagonal[0]), float(diagonal[1]), float(diagonal[2]))
        )
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        fixed_count += 1

    if fixed_count:
        print(f"[S3 USD repair] Wrote missing mass/inertia for {fixed_count} fixed S3 gripper bodies.")


@sim_utils.clone
def _spawn_from_mjcf_with_importer_extension(prim_path, cfg, translation=None, orientation=None):
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import PhysxSchema, Usd, UsdPhysics

    from isaaclab.sim import converters
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    enable_extension("isaacsim.asset.importer.mjcf")
    mjcf_loader = converters.MjcfConverter(cfg)

    stage = Usd.Stage.Open(mjcf_loader.usd_path)
    default_prim = stage.GetDefaultPrim() if stage is not None else None
    if default_prim is not None:
        world_body_prim = stage.GetPrimAtPath(f"{default_prim.GetPath()}/worldBody")
        if world_body_prim.IsValid():
            if world_body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                world_body_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if world_body_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                world_body_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    if stage is not None:
        _repair_s3_g1_29dof_usd(stage, cfg)
        stage.GetRootLayer().Save()

    return _spawn_from_usd_file(prim_path, mjcf_loader.usd_path, cfg, translation=translation, orientation=orientation)


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


UNITREE_GO2_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/go2/usd/go2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)


UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)


UNITREE_S3_G1_29DOF_CFG = UNITREE_G1_29DOF_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/s3_g1_29dof/usd/s3_g1_29dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
)


UNITREE_S3_G1_29DOF_MJCF_CFG = UNITREE_G1_29DOF_CFG.replace(
    spawn=sim_utils.MjcfFileCfg(
        func=_spawn_from_mjcf_with_importer_extension,
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/s3_g1_29dof/g1_29dof.xml",
        usd_dir=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/s3_g1_29dof/usd",
        usd_file_name="s3_g1_29dof.usd",
        force_usd_conversion=False,
        fix_base=False,
        import_sites=True,
        self_collision=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
)
