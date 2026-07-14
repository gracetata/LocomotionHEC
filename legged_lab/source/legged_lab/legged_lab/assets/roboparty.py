# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab import LEGGED_LAB_ROOT_DIR

ATOM01_LONG_BASE_LINK_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/atom01_long_base_link/atom01_long_base_link.urdf",
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_thigh_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.12,
            "left_elbow_pitch_joint": 0.78,
            "right_thigh_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.18,
            "right_arm_roll_joint": -0.12,
            "right_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.3,
                ".*_thigh_roll_joint": 3.3,
                ".*_thigh_pitch_joint": 3.3,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=54.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
    },
)

ATOM01_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/atom01/atom01.usd",
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
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_thigh_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.06,
            "left_elbow_pitch_joint": 0.78,
            "right_thigh_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.18,
            "right_arm_roll_joint": -0.06,
            "right_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.3,
                ".*_thigh_roll_joint": 3.3,
                ".*_thigh_pitch_joint": 3.3,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=54.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            armature=0.01,
            min_delay=3,
            max_delay=5,
        ),
    },
)


# ATOM01_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/atom01/atom01.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.75),
#         joint_pos={
#             "left_thigh_pitch_joint": -0.1,
#             "left_knee_joint": 0.3,
#             "left_arm_roll_joint": 0.06,
#             "left_elbow_pitch_joint": 0.78,
#             "right_thigh_pitch_joint": -0.1,
#             "right_knee_joint": 0.3,
#             "right_ankle_pitch_joint": -0.2,
#             "right_arm_pitch_joint": 0.18,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.90,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#                 ".*_thigh_roll_joint",
#                 ".*_thigh_pitch_joint",
#                 ".*_knee_joint",
#                 ".*torso.*",
#             ],
#             effort_limit_sim=120.0,
#                 ".*_thigh_yaw_joint": 100.0,
#                 ".*_thigh_roll_joint": 100.0,
#                 ".*_thigh_pitch_joint": 100.0,
#                 ".*_knee_joint": 150.0,
#                 ".*torso.*": 150.0,
#             },
#                 ".*_thigh_yaw_joint": 3.3,
#                 ".*_thigh_roll_joint": 3.3,
#                 ".*_thigh_pitch_joint": 3.3,
#                 ".*_knee_joint": 5.0,
#                 ".*torso.*": 5.0,
#             },
#             armature=0.01,
#             joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
#             effort_limit_sim=54.0,
#             velocity_limit_sim=8.0,
#             stiffness=40.0,
#             damping=2.0,
#             armature=0.01,
#             joint_names_expr=[
#                 ".*_arm_pitch_joint",
#                 ".*_arm_roll_joint",
#                 ".*_arm_yaw_joint",
#             ],
#             effort_limit_sim=27.0,
#             damping=2.0,
#             armature=0.01,
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_elbow_pitch_joint",
#             stiffness={
#                 ".*_elbow_pitch_joint": 30.0,
#                 ".*_elbow_yaw_joint": 20.0,
#             },
#             damping={
#                 ".*_elbow_pitch_joint": 1.5,
#             effort_limit_sim=27.0,
#             velocity_limit_sim=8.0,
#             armature=0.01,
#         ),
#     },
# )
# Joint order (from URDF declaration order used by IsaacLab):
#  0 AAHead_yaw       1 Head_pitch
#  2 Left_Shoulder_Pitch   3 Left_Shoulder_Roll   4 Left_Elbow_Pitch
#  5 Left_Elbow_Yaw        6 Left_Wrist_Pitch      7 Left_Wrist_Yaw
#  8 Left_Hand_Roll
#  9 Right_Shoulder_Pitch 10 Right_Shoulder_Roll  11 Right_Elbow_Pitch
# 12 Right_Elbow_Yaw      13 Right_Wrist_Pitch    14 Right_Wrist_Yaw
# 15 Right_Hand_Roll
# 16 Waist
# 17 Left_Hip_Pitch  18 Left_Hip_Roll   19 Left_Hip_Yaw
# 20 Left_Knee_Pitch 21 Left_Ankle_Pitch 22 Left_Ankle_Roll
# 23 Right_Hip_Pitch 24 Right_Hip_Roll  25 Right_Hip_Yaw
# 26 Right_Knee_Pitch 27 Right_Ankle_Pitch 28 Right_Ankle_Roll
T1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/T1_29dof/T1_29dof.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.82),
        joint_pos={
            # Standing leg posture (confirmed visually)
            "Left_Shoulder_Roll": -1.25,
            "Right_Shoulder_Roll": 1.25,
            "Left_Hand_Roll": -0.26,
            "Right_Hand_Roll": 0.26,
            "Right_Elbow_Yaw": 0.09,
            "Left_Hip_Pitch": -0.2,
            "Right_Hip_Pitch": -0.2,
            "Left_Knee_Pitch": 0.45,
            "Right_Knee_Pitch": 0.45,
            "Left_Ankle_Pitch": -0.25,
            "Right_Ankle_Pitch": -0.25,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit_sim=40.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*_Shoulder_Pitch", ".*_Shoulder_Roll"],
            effort_limit_sim=80.0,
            velocity_limit_sim=15.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "elbows": ImplicitActuatorCfg(
            joint_names_expr=[".*_Elbow_Pitch", ".*_Elbow_Yaw"],
            effort_limit_sim=60.0,
            velocity_limit_sim=15.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_Wrist_Pitch", ".*_Wrist_Yaw"],
            effort_limit_sim=30.0,
            velocity_limit_sim=15.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_Hand_Roll"],
            effort_limit_sim=20.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=200.0,
            velocity_limit_sim=10.0,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "hips_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_Hip_Pitch"],
            effort_limit_sim=200.0,
            velocity_limit_sim=15.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "hips_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_Hip_Roll"],
            effort_limit_sim=200.0,
            velocity_limit_sim=15.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "hips_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_Hip_Yaw"],
            effort_limit_sim=150.0,
            velocity_limit_sim=15.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_Knee_Pitch"],
            effort_limit_sim=300.0,
            velocity_limit_sim=15.0,
            stiffness=150.0,
            damping=4.0,
            armature=0.01,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim=50.0,
            velocity_limit_sim=15.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
