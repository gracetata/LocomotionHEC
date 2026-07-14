"""Run a guarded Unitree G1 high-level FSM handoff and WASD control loop.

Core flow:
    1. Verify DDS network, rt/lowstate, MotionSwitcher, robot_state, and Loco RPC.
    2. Optionally send a safe zero-velocity command to prove command delivery.
     3. Run one guarded handoff flow:
         - squat2stand_auto: Damp(FSM 1) -> Squat2StandUp(FSM 706) -> wait FSM 500.
             - standup_start: Damp(FSM 1) -> StandUp(FSM 4) -> Start candidates
                 such as FSM 500/801/802.
    4. Either wait for keyboard confirmation before WASD, or run the guarded
        ZeroTorque full-body rt/lowcmd takeover: align, preheat q_t0 at 500 Hz,
        switch FSM 0, hold all 29 joints, micro-bend both knees, then recover.
    5. Optionally stop, enter BalanceStand, and run a right-arm DDS swing through
            rt/arm_sdk while holding non-target arm_sdk joints near their current pose
            and leaving the lower body on Unitree built-in balance.

Inputs/outputs:
    Inputs are a Unitree network interface, service name, safety confirmation,
    velocity limits, and timing parameters. Outputs are status lines and RPC
    return codes; any non-zero command code aborts before continuing.

Usage:
    python g1_loco_fsm_wasd.py enp11s0 --probe-only
    python g1_loco_fsm_wasd.py enp11s0 --verify-zero-command --confirm-real-robot I_UNDERSTAND
    python g1_loco_fsm_wasd.py enp11s0 --confirm-real-robot I_UNDERSTAND --arm-only-test --run-arm-swing-after-wasd
    python g1_loco_fsm_wasd.py enp11s0 --confirm-real-robot I_UNDERSTAND --run-zero-torque-takeover
    python g1_loco_fsm_wasd.py enp11s0 --confirm-real-robot I_UNDERSTAND --direct-zero-arm-motor-test
    python g1_loco_fsm_wasd.py enp11s0 --confirm-real-robot I_UNDERSTAND
"""

import argparse
import json
import math
import select
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as idl_types

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC


CONFIRM_TOKEN = "I_UNDERSTAND"
RPC_ERR_CLIENT_SEND = 3102
ARM_SDK_WEIGHT_INDEX = 29
DEFAULT_ARM_ALLOWED_FSM_IDS = "4,500,501,801,802"
DEFAULT_LOCOMOTION_FSM_IDS = "500,801,802"
FULL_BODY_JOINT_COUNT = 29

LEG_JOINT_INDICES = list(range(12))
LEG_JOINT_NAMES = {
    0: "left_hip_pitch",
    1: "left_hip_roll",
    2: "left_hip_yaw",
    3: "left_knee",
    4: "left_ankle_pitch",
    5: "left_ankle_roll",
    6: "right_hip_pitch",
    7: "right_hip_roll",
    8: "right_hip_yaw",
    9: "right_knee",
    10: "right_ankle_pitch",
    11: "right_ankle_roll",
}


class G1JointIndex:
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


@dataclass
@annotate.final
@annotate.autoid("sequential")
class G1SportModeState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.SportModeState_"):
    fsm_id: idl_types.uint32
    fsm_mode: idl_types.uint32
    task_id: idl_types.uint32
    task_time: idl_types.float32


@dataclass
class LowStateSnapshot:
    tick: int
    mode_pr: int
    mode_machine: int
    rpy: Tuple[float, float, float]
    motor0_q: float
    motor0_dq: float
    motor0_tau: float


@dataclass
class SportModeSnapshot:
    fsm_id: int
    fsm_mode: int
    task_id: int
    task_time: float


@dataclass
class ArmJointCommand:
    name: str
    index: int
    amplitude_rad: float
    phase_rad: float
    swing: bool = True


WAIST_JOINT_INDICES = [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]
LEFT_ARM5_JOINT_INDICES = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
]
RIGHT_ARM5_JOINT_INDICES = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
]
LEFT_WRIST_EXTRA_JOINT_INDICES = [G1JointIndex.LeftWristPitch, G1JointIndex.LeftWristYaw]
RIGHT_WRIST_EXTRA_JOINT_INDICES = [G1JointIndex.RightWristPitch, G1JointIndex.RightWristYaw]
ARM_JOINT_NAMES = {
    G1JointIndex.WaistYaw: "waist_yaw",
    G1JointIndex.WaistRoll: "waist_roll",
    G1JointIndex.WaistPitch: "waist_pitch",
    G1JointIndex.LeftShoulderPitch: "left_shoulder_pitch",
    G1JointIndex.LeftShoulderRoll: "left_shoulder_roll",
    G1JointIndex.LeftShoulderYaw: "left_shoulder_yaw",
    G1JointIndex.LeftElbow: "left_elbow",
    G1JointIndex.LeftWristRoll: "left_wrist_roll",
    G1JointIndex.LeftWristPitch: "left_wrist_pitch",
    G1JointIndex.LeftWristYaw: "left_wrist_yaw",
    G1JointIndex.RightShoulderPitch: "right_shoulder_pitch",
    G1JointIndex.RightShoulderRoll: "right_shoulder_roll",
    G1JointIndex.RightShoulderYaw: "right_shoulder_yaw",
    G1JointIndex.RightElbow: "right_elbow",
    G1JointIndex.RightWristRoll: "right_wrist_roll",
    G1JointIndex.RightWristPitch: "right_wrist_pitch",
    G1JointIndex.RightWristYaw: "right_wrist_yaw",
}
ARM_JOINT_LIMITS = {
    G1JointIndex.WaistYaw: (-2.618, 2.618),
    G1JointIndex.WaistRoll: (-0.52, 0.52),
    G1JointIndex.WaistPitch: (-0.52, 0.52),
    G1JointIndex.LeftShoulderPitch: (-3.0892, 2.6704),
    G1JointIndex.LeftShoulderRoll: (-1.5882, 2.2515),
    G1JointIndex.LeftShoulderYaw: (-2.618, 2.618),
    G1JointIndex.LeftElbow: (-1.0472, 2.0944),
    G1JointIndex.LeftWristRoll: (-1.97222, 1.97222),
    G1JointIndex.LeftWristPitch: (-1.61443, 1.61443),
    G1JointIndex.LeftWristYaw: (-1.61443, 1.61443),
    G1JointIndex.RightShoulderPitch: (-3.0892, 2.6704),
    G1JointIndex.RightShoulderRoll: (-2.2515, 1.5882),
    G1JointIndex.RightShoulderYaw: (-2.618, 2.618),
    G1JointIndex.RightElbow: (-1.0472, 2.0944),
    G1JointIndex.RightWristRoll: (-1.97222, 1.97222),
    G1JointIndex.RightWristPitch: (-1.61443, 1.61443),
    G1JointIndex.RightWristYaw: (-1.61443, 1.61443),
}
FULL_BODY_JOINT_INDICES = list(range(FULL_BODY_JOINT_COUNT))
FULL_BODY_JOINT_NAMES = {**LEG_JOINT_NAMES, **ARM_JOINT_NAMES}
KNEE_JOINT_INDICES = [3, 9]


def install_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}; entering safety shutdown.")

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


class RawTerminal:
    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._old_settings: Optional[List[object]] = None

    def __enter__(self) -> "RawTerminal":
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def print_header(title: str) -> None:
    print("=====================================")
    print(f"  {title}")
    print("=====================================")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_int_set(raw_value: str) -> List[int]:
    values: List[int] = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def get_swing_arm_joint_commands(joint_set: str, amplitude_scale: float) -> List[ArmJointCommand]:
    left_arm5 = [
        ArmJointCommand("left_shoulder_pitch", G1JointIndex.LeftShoulderPitch, 0.05, 0.0),
        ArmJointCommand("left_shoulder_roll", G1JointIndex.LeftShoulderRoll, 0.06, 0.0),
        ArmJointCommand("left_shoulder_yaw", G1JointIndex.LeftShoulderYaw, 0.04, math.pi),
        ArmJointCommand("left_elbow", G1JointIndex.LeftElbow, 0.05, math.pi),
        ArmJointCommand("left_wrist_roll", G1JointIndex.LeftWristRoll, 0.04, 0.0),
    ]
    right_arm5 = [
        ArmJointCommand("right_shoulder_pitch", G1JointIndex.RightShoulderPitch, 0.90, math.pi),
        ArmJointCommand("right_shoulder_roll", G1JointIndex.RightShoulderRoll, -0.80, math.pi),
        ArmJointCommand("right_shoulder_yaw", G1JointIndex.RightShoulderYaw, 0.85, 0.0),
        ArmJointCommand("right_elbow", G1JointIndex.RightElbow, 0.75, 0.0),
        ArmJointCommand("right_wrist_roll", G1JointIndex.RightWristRoll, 0.65, math.pi),
    ]
    left_wrist_extra = [
        ArmJointCommand("left_wrist_pitch", G1JointIndex.LeftWristPitch, 0.025, math.pi),
        ArmJointCommand("left_wrist_yaw", G1JointIndex.LeftWristYaw, 0.025, 0.0),
    ]
    right_wrist_extra = [
        ArmJointCommand("right_wrist_pitch", G1JointIndex.RightWristPitch, 0.35, 0.0),
        ArmJointCommand("right_wrist_yaw", G1JointIndex.RightWristYaw, 0.35, math.pi),
    ]
    single_joint_commands = {
        "right_shoulder_pitch": [
            ArmJointCommand("right_shoulder_pitch", G1JointIndex.RightShoulderPitch, 0.90, math.pi)
        ],
        "right_shoulder_roll": [
            ArmJointCommand("right_shoulder_roll", G1JointIndex.RightShoulderRoll, -0.80, math.pi)
        ],
        "right_shoulder_yaw": [
            ArmJointCommand("right_shoulder_yaw", G1JointIndex.RightShoulderYaw, 0.85, 0.0)
        ],
    }

    if joint_set in {"arm5", "both_arm5"}:
        base_commands = left_arm5 + right_arm5
    elif joint_set in single_joint_commands:
        base_commands = single_joint_commands[joint_set]
    elif joint_set == "right_arm5":
        base_commands = right_arm5
    elif joint_set == "left_arm5":
        base_commands = left_arm5
    elif joint_set in {"arm7", "both_arm7"}:
        base_commands = left_arm5 + left_wrist_extra + right_arm5 + right_wrist_extra
    elif joint_set == "right_arm7":
        base_commands = right_arm5 + right_wrist_extra
    elif joint_set == "left_arm7":
        base_commands = left_arm5 + left_wrist_extra
    else:
        raise ValueError(f"Unsupported arm joint set: {joint_set}")
    return [
        ArmJointCommand(command.name, command.index, command.amplitude_rad * amplitude_scale, command.phase_rad)
        for command in base_commands
    ]


def get_arm_sdk_scope_indices(joint_set: str) -> List[int]:
    single_joint_full_scope = {"right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw"}
    if "7" in joint_set or joint_set in single_joint_full_scope:
        return (
            WAIST_JOINT_INDICES
            + LEFT_ARM5_JOINT_INDICES
            + LEFT_WRIST_EXTRA_JOINT_INDICES
            + RIGHT_ARM5_JOINT_INDICES
            + RIGHT_WRIST_EXTRA_JOINT_INDICES
        )
    return WAIST_JOINT_INDICES + LEFT_ARM5_JOINT_INDICES + RIGHT_ARM5_JOINT_INDICES


def get_arm_joint_commands(joint_set: str, amplitude_scale: float, hold_non_target: bool) -> List[ArmJointCommand]:
    swing_commands = get_swing_arm_joint_commands(joint_set, amplitude_scale)
    if not hold_non_target:
        return swing_commands

    swing_indices = {command.index for command in swing_commands}
    commands = list(swing_commands)
    for index in get_arm_sdk_scope_indices(joint_set):
        if index in swing_indices:
            continue
        commands.append(ArmJointCommand(f"hold_{ARM_JOINT_NAMES[index]}", index, 0.0, 0.0, False))
    return commands


def apply_joint_limit_safety(
    commands: Sequence[ArmJointCommand],
    base_positions: Dict[int, float],
    limit_margin_rad: float,
    max_amplitude_rad: float,
) -> List[ArmJointCommand]:
    safe_commands: List[ArmJointCommand] = []
    for command in commands:
        if not command.swing:
            safe_commands.append(command)
            continue
        lower, upper = ARM_JOINT_LIMITS[command.index]
        base_q = base_positions[command.index]
        safe_lower = lower + limit_margin_rad
        safe_upper = upper - limit_margin_rad
        usable_margin = max(0.0, min(base_q - safe_lower, safe_upper - base_q))
        safe_amplitude = min(abs(command.amplitude_rad), usable_margin, max_amplitude_rad)
        signed_amplitude = math.copysign(safe_amplitude, command.amplitude_rad)
        safe_commands.append(ArmJointCommand(command.name, command.index, signed_amplitude, command.phase_rad, command.swing))
    return safe_commands


def get_joint_q(state: LowState_, index: int) -> float:
    return float(state.motor_state[index].q)


def get_joint_dq(state: LowState_, index: int) -> float:
    return float(state.motor_state[index].dq)


def get_joint_tau(state: LowState_, index: int) -> float:
    return float(state.motor_state[index].tau_est)


def print_commanded_joint_table(
    state: LowState_,
    commands: Sequence[ArmJointCommand],
    base_positions: Dict[int, float],
    target_positions: Dict[int, float],
    title: str = "arm_sdk commanded joint readback:",
) -> None:
    print(title)
    for command in commands:
        role = "swing" if command.swing else "hold"
        current_q = get_joint_q(state, command.index)
        target_q = target_positions.get(command.index, base_positions[command.index])
        base_error = current_q - base_positions[command.index]
        target_error = current_q - target_q
        print(
            f"  {role:<5} {command.name:<27} idx={command.index:2d} "
            f"q={current_q: .4f} dq={get_joint_dq(state, command.index): .4f} "
            f"tau={get_joint_tau(state, command.index): .3f} "
            f"base={base_positions[command.index]: .4f} target={target_q: .4f} "
            f"err_base={base_error: .4f} err_target={target_error: .4f}"
        )


def get_max_hold_error(
    state: LowState_,
    commands: Sequence[ArmJointCommand],
    base_positions: Dict[int, float],
    allowed_indices: Optional[Iterable[int]] = None,
) -> Tuple[float, str]:
    allowed_set = set(allowed_indices) if allowed_indices is not None else None
    max_error = 0.0
    max_name = "<none>"
    for command in commands:
        if command.swing:
            continue
        if allowed_set is not None and command.index not in allowed_set:
            continue
        error = abs(get_joint_q(state, command.index) - base_positions[command.index])
        if error > max_error:
            max_error = error
            max_name = command.name
    return max_error, max_name


def update_joint_readback_stats(
    state: LowState_,
    indices: Iterable[int],
    base_positions: Dict[int, float],
    max_delta: Dict[int, float],
    max_dq: Dict[int, float],
    max_tau: Dict[int, float],
) -> None:
    for index in indices:
        delta_q = abs(get_joint_q(state, index) - base_positions[index])
        max_delta[index] = max(max_delta.get(index, 0.0), delta_q)
        max_dq[index] = max(max_dq.get(index, 0.0), abs(get_joint_dq(state, index)))
        max_tau[index] = max(max_tau.get(index, 0.0), abs(get_joint_tau(state, index)))


def get_max_stat(indices: Iterable[int], stats: Dict[int, float], names: Dict[int, str]) -> Tuple[float, str]:
    max_value = 0.0
    max_name = "<none>"
    for index in indices:
        value = stats.get(index, 0.0)
        if value > max_value:
            max_value = value
            max_name = names.get(index, f"joint_{index}")
    return max_value, max_name


def print_readback_stats_table(
    title: str,
    indices: Iterable[int],
    names: Dict[int, str],
    base_positions: Dict[int, float],
    max_delta: Dict[int, float],
    max_dq: Dict[int, float],
    max_tau: Dict[int, float],
) -> None:
    print(title)
    for index in indices:
        print(
            f"  idx={index:2d} {names.get(index, f'joint_{index}'):<27} "
            f"base_q={base_positions[index]: .4f} "
            f"max_abs_delta_q={max_delta.get(index, 0.0): .4f} "
            f"max_abs_dq={max_dq.get(index, 0.0): .4f} "
            f"max_abs_tau={max_tau.get(index, 0.0): .3f}"
        )


def summarize_leg_motion(state: LowState_, base_leg_q: Dict[int, float]) -> Tuple[float, float]:
    max_abs_dq = max(abs(get_joint_dq(state, index)) for index in LEG_JOINT_INDICES)
    max_abs_delta_q = max(abs(get_joint_q(state, index) - base_leg_q[index]) for index in LEG_JOINT_INDICES)
    return max_abs_dq, max_abs_delta_q


def decode_json_data(data: object) -> object:
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def rpc_value_as_int(data: object) -> Optional[int]:
    decoded = decode_json_data(data)
    if isinstance(decoded, bool):
        return int(decoded)
    if isinstance(decoded, int):
        return decoded
    if isinstance(decoded, float) and decoded.is_integer():
        return int(decoded)
    if isinstance(decoded, dict):
        for key in ("data", "value", "fsm_id", "fsm_mode"):
            if key in decoded:
                value = rpc_value_as_int(decoded[key])
                if value is not None:
                    return value
    if isinstance(decoded, str):
        text = decoded.strip()
        if text:
            try:
                return int(text)
            except ValueError:
                return None
    return None


def snapshot_lowstate(state: LowState_) -> LowStateSnapshot:
    return LowStateSnapshot(
        tick=int(state.tick),
        mode_pr=int(state.mode_pr),
        mode_machine=int(state.mode_machine),
        rpy=tuple(float(v) for v in state.imu_state.rpy),
        motor0_q=float(state.motor_state[0].q),
        motor0_dq=float(state.motor_state[0].dq),
        motor0_tau=float(state.motor_state[0].tau_est),
    )


def print_lowstate(prefix: str, state: LowState_) -> None:
    snapshot = snapshot_lowstate(state)
    rpy = ", ".join(f"{value:.4f}" for value in snapshot.rpy)
    print(
        f"{prefix}: tick={snapshot.tick} mode_pr={snapshot.mode_pr} "
        f"mode_machine={snapshot.mode_machine} rpy=[{rpy}] "
        f"motor0(q={snapshot.motor0_q:.4f}, dq={snapshot.motor0_dq:.4f}, tau={snapshot.motor0_tau:.4f})"
    )


def wait_lowstate(timeout_s: float) -> Tuple[ChannelSubscriber, LowState_]:
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = subscriber.Read(0.5)
        if state is not None:
            print_lowstate("lowstate received", state)
            return subscriber, state
    raise RuntimeError(f"No rt/lowstate sample received within {timeout_s:.1f}s.")


def snapshot_sport_state(state: G1SportModeState_) -> SportModeSnapshot:
    return SportModeSnapshot(
        fsm_id=int(state.fsm_id),
        fsm_mode=int(state.fsm_mode),
        task_id=int(state.task_id),
        task_time=float(state.task_time),
    )


def print_sport_state(prefix: str, state: G1SportModeState_) -> None:
    snapshot = snapshot_sport_state(state)
    print(
        f"{prefix}: fsm_id={snapshot.fsm_id} fsm_mode={snapshot.fsm_mode} "
        f"task_id={snapshot.task_id} task_time={snapshot.task_time:.3f}"
    )


def wait_sport_state(topic: str, timeout_s: float) -> Tuple[ChannelSubscriber, Optional[G1SportModeState_]]:
    subscriber = ChannelSubscriber(topic, G1SportModeState_)
    subscriber.Init()
    state = subscriber.Read(timeout_s)
    if state is not None:
        print_sport_state(f"{topic} received", state)
        return subscriber, state
    print(f"{topic}: no SportModeState sample within {timeout_s:.1f}s")
    return subscriber, None


def drain_lowstate(subscriber: ChannelSubscriber, timeout_s: float = 0.05) -> Optional[LowState_]:
    state = subscriber.Read(timeout_s)
    return state


def wait_latest_lowstate(subscriber: ChannelSubscriber, timeout_s: float) -> LowState_:
    deadline = time.monotonic() + timeout_s
    latest_state: Optional[LowState_] = None
    while time.monotonic() < deadline:
        state = drain_lowstate(subscriber, 0.1)
        if state is not None:
            latest_state = state
    if latest_state is None:
        raise RuntimeError(f"No fresh rt/lowstate sample received within {timeout_s:.1f}s.")
    return latest_state


def print_motion_switcher_state() -> Tuple[int, object]:
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(3.0)
    motion_switcher.Init()
    code, data = motion_switcher.CheckMode()
    print_header("MotionSwitcher RPC")
    print(f"CheckMode code : {code}")
    print(f"CheckMode data : {data}")
    return code, data


def service_status_text(status: int) -> str:
    return f"reported_status_{status}"


def print_robot_state_services(args: argparse.Namespace) -> None:
    client = RobotStateClient()
    client.SetTimeout(args.rpc_timeout_s)
    client.Init()
    print_header("RobotState RPC")
    code, version = client.GetServerApiVersion()
    print(f"GetServerApiVersion code={code} version={version}")
    code, services = client.ServiceList()
    print(f"ServiceList code={code}")
    if code != 0 or services is None:
        return
    for service in services:
        print(
            f"service name={service.name} status={service.status} "
            f"({service_status_text(int(service.status))}) protect={service.protect}"
        )
    for service_name in args.start_service:
        if not service_name:
            continue
        wait_for_space(
            args,
            f"robot_state.ServiceSwitch({service_name}, True)",
            [
                "即将启动机器人内置高层运动服务。",
                f"service={service_name}",
                "按空格执行；按 q 或 Ctrl+C 取消。",
            ],
        )
        switch_code = client.ServiceSwitch(service_name, True)
        print(f"ServiceSwitch({service_name}, True) code={switch_code}")
        if switch_code != 0:
            raise RuntimeError(f"Failed to start robot_state service {service_name}: code={switch_code}")
        if args.service_start_wait_s > 0.0:
            time.sleep(args.service_start_wait_s)


def print_loco_rpc_state(client: LocoClient, strict: bool) -> bool:
    print_header(f"G1 Loco RPC service={client.service_name}")
    code, version = client.GetServerApiVersion()
    print(f"GetServerApiVersion code={code} version={version}")
    if code == RPC_ERR_CLIENT_SEND:
        print(
            "Meaning: client could not match a server subscription on "
            f"rt/api/{client.service_name}/request. Network/DDS may still be OK, "
            "but this Loco RPC service is unavailable in the current robot mode."
        )
    if code != 0:
        if strict:
            raise RuntimeError(f"Loco RPC service {client.service_name} is unavailable: code={code}")
        return False
    getters: Iterable[Tuple[str, Callable[[], Tuple[int, object]]]] = [
        ("fsm_id", client.GetFsmId),
        ("fsm_mode", client.GetFsmMode),
        ("balance_mode", client.GetBalanceMode),
        ("swing_height", client.GetSwingHeight),
        ("stand_height", client.GetStandHeight),
        ("phase", client.GetPhase),
    ]
    for name, getter in getters:
        get_code, value = getter()
        print(f"{name:<14}: code={get_code} value={decode_json_data(value)}")
    return True


def require_code(label: str, code: int) -> None:
    print(f"{label}: code={code}")
    if code != 0:
        raise RuntimeError(f"{label} failed with code {code}")


def best_effort_loco_damping(args: argparse.Namespace, client: LocoClient, label: str) -> None:
    print_header(label)
    for attempt in range(1, args.safe_damp_rpc_retries + 1):
        try:
            stop_code = client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s)
            damp_code = client.SetFsmId(1)
            print(f"safety damping attempt {attempt}: SetVelocity code={stop_code}; SetFsmId(1) Damp code={damp_code}")
            if damp_code == 0:
                return
        except BaseException as exc:
            print(f"safety damping attempt {attempt}: exception={exc}")
        time.sleep(args.safe_damp_retry_interval_s)


def verify_zero_velocity(args: argparse.Namespace, client: LocoClient) -> None:
    print_header("Safe Zero-Velocity Command")
    wait_for_space(
        args,
        "SetVelocity(0,0,0) zero-command check",
        [
            "即将发送一条零速度命令，用来验证 Loco RPC 命令通道。",
            f"duration={args.zero_command_duration_s:.2f}s",
        ],
        client=client,
    )
    code = client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s)
    require_code(f"SetVelocity(0,0,0,{args.zero_command_duration_s:.2f})", code)


def sleep_with_lowstate(subscriber: ChannelSubscriber, duration_s: float, label: str) -> None:
    deadline = time.monotonic() + duration_s
    next_print = time.monotonic()
    while time.monotonic() < deadline:
        state = drain_lowstate(subscriber, 0.1)
        now = time.monotonic()
        if state is not None and now >= next_print:
            print_lowstate(label, state)
            next_print = now + 1.0


def poll_target_fsm(
    client: LocoClient,
    timeout_s: float,
    sport_state_subscriber: Optional[ChannelSubscriber],
    target_fsm_ids: Sequence[int],
    label: str,
) -> bool:
    target_values = {int(value) for value in target_fsm_ids}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if sport_state_subscriber is not None:
            sport_state = sport_state_subscriber.Read(0.2)
            if sport_state is not None:
                print_sport_state("SportModeState poll", sport_state)
                if int(sport_state.fsm_id) in target_values:
                    return True
        code, fsm_id = client.GetFsmId()
        print(f"GetFsmId poll ({label}): code={code} value={fsm_id}")
        rpc_fsm_id = rpc_value_as_int(fsm_id)
        if code == 0 and rpc_fsm_id in target_values:
            return True
        time.sleep(1.0)
    return False


def poll_locomotion_fsm(
    client: LocoClient,
    timeout_s: float,
    sport_state_subscriber: Optional[ChannelSubscriber],
    target_fsm_ids: Sequence[int],
) -> bool:
    label = "target=" + "/".join(str(value) for value in target_fsm_ids)
    return poll_target_fsm(client, timeout_s, sport_state_subscriber, target_fsm_ids, label)


def require_confirmed_fsm(
    client: LocoClient,
    timeout_s: float,
    sport_state_subscriber: Optional[ChannelSubscriber],
    target_fsm_ids: Sequence[int],
    label: str,
) -> None:
    if poll_target_fsm(client, timeout_s, sport_state_subscriber, target_fsm_ids, label):
        print(f"Confirmed FSM for {label}: target={list(target_fsm_ids)}")
        return
    raise RuntimeError(f"FSM confirmation failed for {label}; expected one of {list(target_fsm_ids)}.")


def wait_arm_allowed_fsm(
    args: argparse.Namespace,
    client: LocoClient,
    sport_state_subscriber: Optional[ChannelSubscriber],
) -> None:
    allowed_ids = set(parse_int_set(args.arm_allowed_fsm_ids))
    deadline = time.monotonic() + args.arm_fsm_timeout_s
    while time.monotonic() < deadline:
        if sport_state_subscriber is not None:
            sport_state = sport_state_subscriber.Read(0.2)
            if sport_state is not None:
                snapshot = snapshot_sport_state(sport_state)
                print_sport_state("SportModeState before arm_sdk", sport_state)
                if snapshot.fsm_id in allowed_ids and (not args.arm_require_static_fsm_mode or snapshot.fsm_mode == 0):
                    return
        code, fsm_id = client.GetFsmId()
        mode_code, fsm_mode = client.GetFsmMode()
        print(f"GetFsmId before arm_sdk: code={code} value={fsm_id}; GetFsmMode code={mode_code} value={fsm_mode}")
        rpc_fsm_id = rpc_value_as_int(fsm_id)
        rpc_fsm_mode = rpc_value_as_int(fsm_mode)
        if code == 0 and rpc_fsm_id in allowed_ids:
            if not args.arm_require_static_fsm_mode or (mode_code == 0 and rpc_fsm_mode == 0):
                return
        time.sleep(0.5)
    raise RuntimeError(
        "Current FSM is not confirmed as an arm_sdk-capable built-in mode; "
        f"allowed={sorted(allowed_ids)}, require_static={args.arm_require_static_fsm_mode}."
    )


def run_fsm_handoff(
    args: argparse.Namespace,
    client: LocoClient,
    subscriber: ChannelSubscriber,
    sport_state_subscriber: Optional[ChannelSubscriber],
) -> None:
    locomotion_fsm_ids = parse_int_set(args.locomotion_fsm_ids)
    start_fsm_ids = parse_int_set(args.start_fsm_ids)
    print_header("FSM Handoff")
    wait_for_space(
        args,
        "SetFsmId(1) Damp",
        [
            "即将请求进入阻尼模式，作为后续模式切换前的保底状态。",
            "SDK example/g1 中也先调用 Damp() 再进行起立类动作。",
        ],
        client=client,
        sport_state_subscriber=sport_state_subscriber,
        lowstate_subscriber=subscriber,
    )
    require_code("SetFsmId(1) Damp", client.SetFsmId(1))
    sleep_with_lowstate(subscriber, args.damp_hold_s, "damp hold")

    if args.handoff_flow == "squat2stand_auto":
        wait_for_space(
            args,
            "SetFsmId(706) Squat2StandUp",
            [
                "即将请求 706 平衡下蹲/蹲起任务。",
                "本机历史实测中 706 返回 0 但不一定离开 FSM 1，所以执行后仍会严格轮询 SportModeState/GetFsmId。",
            ],
            client=client,
            sport_state_subscriber=sport_state_subscriber,
            lowstate_subscriber=subscriber,
        )
        print("开始执行 Squat2StandUp，请保持吊架松弛但能起到保护作用...")
        require_code("SetFsmId(706) Squat2StandUp", client.Squat2StandUp())
        print("等待底层轨迹规划器完成起立，并观察是否自动进入 Locomotion 候选 FSM...")
        sleep_with_lowstate(subscriber, args.standup_wait_s, "squat2stand wait")
        reached_locomotion = poll_locomotion_fsm(client, args.fsm_poll_timeout_s, sport_state_subscriber, locomotion_fsm_ids)
    elif args.handoff_flow == "standup_start":
        wait_for_space(
            args,
            "SetFsmId(4) StandUp",
            [
                "即将请求 StandUp/FSM 4 锁定站立。",
                "这是当前真机已验证能从阻尼进入的站立入口。",
            ],
            client=client,
            sport_state_subscriber=sport_state_subscriber,
            lowstate_subscriber=subscriber,
        )
        print("开始执行 StandUp(FSM 4 锁定站立)，请保持吊架松弛但能起到保护作用...")
        require_code("SetFsmId(4) StandUp", client.StandUp())
        sleep_with_lowstate(subscriber, args.standup_wait_s, "standup wait")
        stand_or_locomotion_ids = [4] + locomotion_fsm_ids
        reached_stand = poll_target_fsm(
            client,
            args.fsm_poll_timeout_s,
            sport_state_subscriber,
            stand_or_locomotion_ids,
            "target=" + "/".join(str(value) for value in stand_or_locomotion_ids),
        )
        if not reached_stand:
            raise RuntimeError("StandUp FSM 4 was not confirmed; refusing to request locomotion candidates.")
        reached_locomotion = poll_locomotion_fsm(client, 0.2, sport_state_subscriber, locomotion_fsm_ids)
        if not reached_locomotion:
            print(
                "StandUp confirmed; requesting built-in locomotion candidates: "
                + ",".join(str(value) for value in start_fsm_ids)
            )
            for start_fsm_id in start_fsm_ids:
                wait_for_space(
                    args,
                    f"SetFsmId({start_fsm_id}) Start candidate",
                    [
                        "即将请求主运控候选 FSM。",
                        f"target_fsm={start_fsm_id}",
                        f"接受的主运控观测集合={locomotion_fsm_ids}",
                        "当前 29DoF ai_sport 实测通常是请求 801 后观测到 802。",
                    ],
                    client=client,
                    sport_state_subscriber=sport_state_subscriber,
                    lowstate_subscriber=subscriber,
                )
                code = client.SetFsmId(start_fsm_id)
                print(f"SetFsmId({start_fsm_id}) Start candidate: code={code}")
                if code != 0:
                    continue
                reached_locomotion = poll_locomotion_fsm(
                    client,
                    args.fsm_poll_timeout_s,
                    sport_state_subscriber,
                    locomotion_fsm_ids,
                )
                if reached_locomotion:
                    break
    else:
        raise ValueError(f"Unsupported handoff flow: {args.handoff_flow}")

    if reached_locomotion:
        print("Locomotion FSM observed after handoff.")
        return

    print("Locomotion FSM was not observed during the poll window.")
    if args.set_locomotion_after_standup:
        for start_fsm_id in start_fsm_ids:
            wait_for_space(
                args,
                f"SetFsmId({start_fsm_id}) Locomotion fallback",
                [
                    "即将重新请求主运控候选 FSM。",
                    f"target_fsm={start_fsm_id}",
                ],
                client=client,
                sport_state_subscriber=sport_state_subscriber,
                lowstate_subscriber=subscriber,
            )
            code = client.SetFsmId(start_fsm_id)
            print(f"SetFsmId({start_fsm_id}) Locomotion fallback: code={code}")
            if code != 0:
                continue
            time.sleep(1.0)
            reached_locomotion = poll_locomotion_fsm(client, 2.0, sport_state_subscriber, locomotion_fsm_ids)
            if reached_locomotion:
                break

    if not reached_locomotion:
        message = (
            "Locomotion FSM was not confirmed; refusing to enter WASD control. "
            f"Expected one of {locomotion_fsm_ids}."
        )
        if args.allow_missing_fsm_500:
            print(f"WARNING: {message}")
            return
        raise RuntimeError(message)


def read_key(timeout_s: float) -> Optional[str]:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    return sys.stdin.read(1)


def print_current_state(
    client: Optional[LocoClient] = None,
    sport_state_subscriber: Optional[ChannelSubscriber] = None,
    lowstate_subscriber: Optional[ChannelSubscriber] = None,
) -> None:
    if lowstate_subscriber is not None:
        state = drain_lowstate(lowstate_subscriber, 0.05)
        if state is not None:
            print_lowstate("current lowstate", state)
    if sport_state_subscriber is not None:
        sport_state = sport_state_subscriber.Read(0.05)
        if sport_state is not None:
            print_sport_state("current SportModeState", sport_state)
    if client is not None:
        code, fsm_id = client.GetFsmId()
        mode_code, fsm_mode = client.GetFsmMode()
        print(f"current GetFsmId: code={code} value={fsm_id}; GetFsmMode: code={mode_code} value={fsm_mode}")


def wait_for_space(
    args: argparse.Namespace,
    title: str,
    detail_lines: Sequence[str],
    client: Optional[LocoClient] = None,
    sport_state_subscriber: Optional[ChannelSubscriber] = None,
    lowstate_subscriber: Optional[ChannelSubscriber] = None,
) -> None:
    print_header(f"Manual Step: {title}")
    for line in detail_lines:
        print(line)
    print_current_state(client, sport_state_subscriber, lowstate_subscriber)
    if not args.manual_step_confirm:
        print("MANUAL_STEP_CONFIRM=False，自动继续。")
        return
    if not sys.stdin.isatty():
        raise RuntimeError(f"Manual step '{title}' requires an interactive TTY.")
    print("按空格执行本步；按 q、Esc 或 Ctrl+C 取消。")
    with RawTerminal():
        while True:
            key = read_key(0.1)
            if key is None:
                continue
            if key == " ":
                print("Space confirmed; executing step.")
                return
            if key in {"q", "Q", "\x1b", "\x03"}:
                raise KeyboardInterrupt(f"Manual step cancelled before {title}.")
            print(f"等待空格确认，忽略按键: {repr(key)}")


def run_wasd_loop(args: argparse.Namespace, client: LocoClient) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("WASD loop requires an interactive TTY.")

    key_to_velocity = {
        "w": (args.speed_mps, 0.0, 0.0),
        "s": (-args.speed_mps, 0.0, 0.0),
        "a": (0.0, args.speed_mps, 0.0),
        "d": (0.0, -args.speed_mps, 0.0),
    }
    print_header("WASD Control")
    print("Press w/a/s/d to latch a velocity, space for stop, q to quit.")
    print(
        f"Active command is resent every {args.wasd_repeat_s:.2f}s: "
        f"speed={args.speed_mps:.3f} m/s, command duration={args.command_duration_s:.2f}s."
    )
    start_time = time.monotonic()
    active_key: Optional[str] = None
    active_velocity = (0.0, 0.0, 0.0)
    next_send_time = 0.0
    with RawTerminal():
        while True:
            now = time.monotonic()
            if args.keyboard_seconds > 0 and time.monotonic() - start_time >= args.keyboard_seconds:
                print("Keyboard duration reached; exiting WASD loop.")
                break
            key = read_key(args.keyboard_poll_s)
            if key is not None:
                key = key.lower()
                if key == "q" or key == "\x03" or key == "\x1b":
                    print("Exit key received.")
                    break
                if key == " ":
                    active_key = None
                    active_velocity = (0.0, 0.0, 0.0)
                    code = client.SetVelocity(0.0, 0.0, 0.0, args.command_duration_s)
                    require_code("SetVelocity stop", code)
                    print("Stop command sent; no active WASD velocity.")
                    continue
                if key not in key_to_velocity:
                    print(f"Ignored key: {repr(key)}")
                    continue
                active_key = key
                active_velocity = key_to_velocity[key]
                next_send_time = 0.0

            if active_key is None:
                continue
            now = time.monotonic()
            if now < next_send_time:
                continue
            vx, vy, omega = active_velocity
            code = client.SetVelocity(vx, vy, omega, args.command_duration_s)
            require_code(
                f"SetVelocity active_key={active_key} vx={vx:.3f} vy={vy:.3f} omega={omega:.3f} duration={args.command_duration_s:.2f}",
                code,
            )
            next_send_time = now + args.wasd_repeat_s
    code = client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s)
    require_code("SetVelocity final WASD stop", code)


def run_auto_forward_test(
    args: argparse.Namespace,
    client: LocoClient,
    lowstate_subscriber: ChannelSubscriber,
    sport_state_subscriber: Optional[ChannelSubscriber],
) -> bool:
    if not args.auto_forward_test:
        return True

    print_header("Auto Forward Motion Readback Test")
    if args.auto_forward_continuous_gait:
        require_code("ContinuousGait(True) before auto forward test", client.ContinuousGait(True))
        time.sleep(0.5)

    first_state = wait_latest_lowstate(lowstate_subscriber, args.lowstate_timeout_s)
    base_leg_q = {index: get_joint_q(first_state, index) for index in LEG_JOINT_INDICES}
    print("base leg joints before forward command:")
    for index in LEG_JOINT_INDICES:
        print(f"  {LEG_JOINT_NAMES[index]:<20} idx={index:2d} q={base_leg_q[index]: .4f}")

    max_leg_abs_dq = 0.0
    max_leg_abs_delta_q = 0.0
    observed_fsm_modes = set()
    next_command_time = 0.0
    next_print_time = time.monotonic()
    start_time = time.monotonic()
    try:
        while time.monotonic() - start_time < args.auto_forward_duration_s:
            now = time.monotonic()
            if now >= next_command_time:
                code = client.SetVelocity(args.auto_forward_speed_mps, 0.0, 0.0, args.command_duration_s)
                require_code(
                    f"SetVelocity auto_forward vx={args.auto_forward_speed_mps:.3f} duration={args.command_duration_s:.2f}",
                    code,
                )
                next_command_time = now + args.auto_command_period_s
            state = drain_lowstate(lowstate_subscriber, 0.01)
            if state is not None:
                leg_abs_dq, leg_abs_delta_q = summarize_leg_motion(state, base_leg_q)
                max_leg_abs_dq = max(max_leg_abs_dq, leg_abs_dq)
                max_leg_abs_delta_q = max(max_leg_abs_delta_q, leg_abs_delta_q)
                if now >= next_print_time:
                    print_lowstate("auto_forward lowstate", state)
                    print(
                        f"auto_forward leg readback: max_abs_dq={max_leg_abs_dq:.4f} rad/s, "
                        f"max_abs_delta_q={max_leg_abs_delta_q:.4f} rad"
                    )
                    next_print_time = now + 0.5
            if sport_state_subscriber is not None:
                sport_state = sport_state_subscriber.Read(0.001)
                if sport_state is not None:
                    snapshot = snapshot_sport_state(sport_state)
                    observed_fsm_modes.add(snapshot.fsm_mode)
        require_code("SetVelocity auto_forward final stop", client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s))
        if args.auto_forward_continuous_gait:
            require_code("ContinuousGait(False) after auto forward test", client.ContinuousGait(False))
    except BaseException:
        client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s)
        if args.auto_forward_continuous_gait:
            client.ContinuousGait(False)
        raise

    moved = (
        max_leg_abs_dq >= args.auto_forward_min_leg_dq
        or max_leg_abs_delta_q >= args.auto_forward_min_leg_delta_q
        or 1 in observed_fsm_modes
    )
    print(
        "auto_forward summary: "
        f"speed={args.auto_forward_speed_mps:.3f}, duration={args.auto_forward_duration_s:.2f}, "
        f"max_leg_abs_dq={max_leg_abs_dq:.4f}, max_leg_abs_delta_q={max_leg_abs_delta_q:.4f}, "
        f"observed_fsm_modes={sorted(observed_fsm_modes)}, moved={moved}"
    )
    if args.auto_forward_require_motion and not moved:
        raise RuntimeError(
            "Auto forward test did not observe leg motion or fsm_mode=1. "
            "This suggests host SetVelocity is accepted but not taking control of locomotion input."
        )
    return moved


def set_motor_zero_velocity(motor_cmd: object) -> None:
    if hasattr(motor_cmd, "dq"):
        motor_cmd.dq = 0.0
    elif hasattr(motor_cmd, "qd"):
        motor_cmd.qd = 0.0


class FullBodyLowCmdTakeover:
    def __init__(self, args: argparse.Namespace, base_state: LowState_) -> None:
        self.args = args
        self.publisher = ChannelPublisher(args.takeover_lowcmd_topic, LowCmd_)
        self.publisher.Init()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.mode_pr = 0
        self.mode_machine = int(base_state.mode_machine)
        self.base_q = [float(base_state.motor_state[index].q) for index in FULL_BODY_JOINT_INDICES]
        self.base_tau = [float(base_state.motor_state[index].tau_est) for index in FULL_BODY_JOINT_INDICES]
        self.tau_ff = self._make_tau_ff()
        self._target_q = list(self.base_q)
        self._damping_active = False
        self._damping_kd = float(args.takeover_emergency_damping_kd)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.publish_count = 0
        self.last_publish_t = 0.0
        self.max_publish_lag_s = 0.0
        self.kp = [args.takeover_hold_kp for _ in FULL_BODY_JOINT_INDICES]
        self.kd = [args.takeover_hold_kd for _ in FULL_BODY_JOINT_INDICES]
        for index in LEG_JOINT_INDICES:
            self.kp[index] = args.takeover_leg_kp
            self.kd[index] = args.takeover_leg_kd
        for index in WAIST_JOINT_INDICES:
            self.kp[index] = args.takeover_waist_kp
            self.kd[index] = args.takeover_waist_kd

    def _make_tau_ff(self) -> List[float]:
        if self.args.takeover_tau_ff_mode == "zero":
            return [0.0 for _ in FULL_BODY_JOINT_INDICES]
        max_abs_tau = max(0.0, float(self.args.takeover_max_abs_tau_ff))
        return [
            clamp(tau * self.args.takeover_tau_ff_scale, -max_abs_tau, max_abs_tau)
            for tau in self.base_tau
        ]

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="g1_full_body_lowcmd_takeover", daemon=False)
        self._thread.start()
        if not self._started_event.wait(timeout=self.args.takeover_warmup_timeout_s):
            raise RuntimeError("Full-body lowcmd takeover publisher did not start in time.")

    def stop(self, join_timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(join_timeout_s)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_target_q(self) -> List[float]:
        with self._lock:
            return list(self._target_q)

    def set_target_q(self, target_q: Sequence[float]) -> None:
        if len(target_q) != FULL_BODY_JOINT_COUNT:
            raise ValueError(f"Expected {FULL_BODY_JOINT_COUNT} joint targets, got {len(target_q)}.")
        with self._lock:
            self._target_q = [float(value) for value in target_q]

    def enter_damping_mode(self, damping_kd: float) -> None:
        with self._lock:
            self._damping_active = True
            self._damping_kd = float(damping_kd)

    def wait_for_warmup(self) -> None:
        deadline = time.monotonic() + self.args.takeover_warmup_timeout_s
        while time.monotonic() < deadline:
            if self.publish_count >= self.args.takeover_warmup_min_cycles:
                return
            time.sleep(0.001)
        raise RuntimeError(
            "Full-body lowcmd takeover did not publish enough warm-up cycles: "
            f"count={self.publish_count}, required={self.args.takeover_warmup_min_cycles}."
        )

    def _run(self) -> None:
        next_publish_t = time.monotonic()
        self._started_event.set()
        while not self._stop_event.is_set():
            now = time.monotonic()
            lag_s = max(0.0, now - next_publish_t)
            self.max_publish_lag_s = max(self.max_publish_lag_s, lag_s)
            with self._lock:
                target_q = list(self._target_q)
                damping_active = self._damping_active
                damping_kd = self._damping_kd
            self.low_cmd.mode_pr = self.mode_pr
            self.low_cmd.mode_machine = self.mode_machine
            for index in FULL_BODY_JOINT_INDICES:
                motor_cmd = self.low_cmd.motor_cmd[index]
                motor_cmd.mode = 1
                set_motor_zero_velocity(motor_cmd)
                if damping_active:
                    motor_cmd.tau = 0.0
                    motor_cmd.q = 0.0
                    motor_cmd.kp = 0.0
                    motor_cmd.kd = damping_kd
                else:
                    motor_cmd.tau = float(self.tau_ff[index])
                    motor_cmd.q = target_q[index]
                    motor_cmd.kp = float(self.kp[index])
                    motor_cmd.kd = float(self.kd[index])
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)
            self.publish_count += 1
            self.last_publish_t = time.monotonic()
            next_publish_t += self.args.takeover_control_dt_s
            sleep_s = next_publish_t - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_publish_t = time.monotonic()


def check_takeover_stability(args: argparse.Namespace, state: LowState_) -> None:
    snapshot = snapshot_lowstate(state)
    roll = abs(snapshot.rpy[0])
    pitch = abs(snapshot.rpy[1])
    if roll > args.takeover_max_abs_roll_rad or pitch > args.takeover_max_abs_pitch_rad:
        raise RuntimeError(
            "Full-body takeover aborted because body attitude exceeded limits: "
            f"roll={roll:.3f}, pitch={pitch:.3f}."
        )


def print_full_body_latch_table(state: LowState_) -> None:
    print("full-body takeover q_t0 latch:")
    for index in FULL_BODY_JOINT_INDICES:
        name = FULL_BODY_JOINT_NAMES.get(index, f"joint_{index}")
        print(
            f"  idx={index:2d} {name:<22} "
            f"q={get_joint_q(state, index): .4f} dq={get_joint_dq(state, index): .4f} "
            f"tau={get_joint_tau(state, index): .3f}"
        )


def print_takeover_command_table(controller: FullBodyLowCmdTakeover) -> None:
    print(
        "full-body takeover command gains/feedforward: "
        f"tau_ff_mode={controller.args.takeover_tau_ff_mode}, "
        f"tau_ff_scale={controller.args.takeover_tau_ff_scale}, "
        f"max_abs_tau_ff={controller.args.takeover_max_abs_tau_ff}"
    )
    for index in FULL_BODY_JOINT_INDICES:
        name = FULL_BODY_JOINT_NAMES.get(index, f"joint_{index}")
        print(
            f"  idx={index:2d} {name:<22} "
            f"kp={controller.kp[index]: .3f} kd={controller.kd[index]: .3f} "
            f"latched_tau={controller.base_tau[index]: .3f} tau_ff={controller.tau_ff[index]: .3f}"
        )


def get_takeover_readback_summary(
    state: LowState_,
    target_q: Sequence[float],
    base_q: Sequence[float],
) -> Tuple[float, str, float, str, float, str]:
    max_target_error = 0.0
    max_target_error_name = "<none>"
    max_abs_dq = 0.0
    max_abs_dq_name = "<none>"
    max_base_delta = 0.0
    max_base_delta_name = "<none>"
    for index in FULL_BODY_JOINT_INDICES:
        name = FULL_BODY_JOINT_NAMES.get(index, f"joint_{index}")
        target_error = abs(get_joint_q(state, index) - target_q[index])
        if target_error > max_target_error:
            max_target_error = target_error
            max_target_error_name = name
        abs_dq = abs(get_joint_dq(state, index))
        if abs_dq > max_abs_dq:
            max_abs_dq = abs_dq
            max_abs_dq_name = name
        base_delta = abs(get_joint_q(state, index) - base_q[index])
        if base_delta > max_base_delta:
            max_base_delta = base_delta
            max_base_delta_name = name
    return (
        max_target_error,
        max_target_error_name,
        max_abs_dq,
        max_abs_dq_name,
        max_base_delta,
        max_base_delta_name,
    )


def print_takeover_monitor(label: str, state: LowState_, controller: FullBodyLowCmdTakeover) -> None:
    target_q = controller.get_target_q()
    (
        max_target_error,
        max_target_error_name,
        max_abs_dq,
        max_abs_dq_name,
        max_base_delta,
        max_base_delta_name,
    ) = get_takeover_readback_summary(state, target_q, controller.base_q)
    snapshot = snapshot_lowstate(state)
    rpy = ", ".join(f"{value:.4f}" for value in snapshot.rpy)
    knee_text = ", ".join(
        f"{FULL_BODY_JOINT_NAMES[index]} q={get_joint_q(state, index):.4f} target={target_q[index]:.4f}"
        for index in KNEE_JOINT_INDICES
    )
    print(
        f"{label}: tick={snapshot.tick} fsm_mode_machine={snapshot.mode_machine} "
        f"rpy=[{rpy}] publish_count={controller.publish_count} "
        f"max_target_error={max_target_error:.4f} rad joint={max_target_error_name}; "
        f"max_abs_dq={max_abs_dq:.4f} rad/s joint={max_abs_dq_name}; "
        f"max_base_delta={max_base_delta:.4f} rad joint={max_base_delta_name}; "
        f"{knee_text}"
    )


def wait_for_space_with_takeover_monitor(
    args: argparse.Namespace,
    title: str,
    detail_lines: Sequence[str],
    controller: FullBodyLowCmdTakeover,
    lowstate_subscriber: ChannelSubscriber,
    client: Optional[LocoClient] = None,
    sport_state_subscriber: Optional[ChannelSubscriber] = None,
    enforce_stability: bool = True,
) -> None:
    print_header(f"Manual Step: {title}")
    for line in detail_lines:
        print(line)
    print_current_state(client, sport_state_subscriber, lowstate_subscriber)
    print(
        f"lowcmd warm-up publisher: topic={args.takeover_lowcmd_topic}, "
        f"dt={args.takeover_control_dt_s:.4f}s, publish_count={controller.publish_count}, "
        f"max_publish_lag={controller.max_publish_lag_s * 1000.0:.2f}ms"
    )
    if not args.manual_step_confirm:
        print("MANUAL_STEP_CONFIRM=False，自动继续。")
        return
    if not sys.stdin.isatty():
        raise RuntimeError(f"Manual step '{title}' requires an interactive TTY.")
    print("按空格执行本步；按 q、Esc 或 Ctrl+C 取消。等待期间低层线程持续发布全身保持指令。")
    next_print = time.monotonic()
    with RawTerminal():
        while True:
            key = read_key(args.keyboard_poll_s)
            if key == " ":
                print("Space confirmed; executing step.")
                return
            if key in {"q", "Q", "\x1b", "\x03"}:
                raise KeyboardInterrupt(f"Manual step cancelled before {title}.")
            now = time.monotonic()
            state = drain_lowstate(lowstate_subscriber, 0.001)
            if state is not None and enforce_stability:
                check_takeover_stability(args, state)
            if state is not None and now >= next_print:
                print_takeover_monitor("takeover warm/hold monitor", state, controller)
                next_print = now + args.takeover_monitor_period_s


def run_takeover_knee_bend_until_space(
    args: argparse.Namespace,
    controller: FullBodyLowCmdTakeover,
    lowstate_subscriber: ChannelSubscriber,
) -> float:
    if args.manual_step_confirm and not sys.stdin.isatty():
        raise RuntimeError("Knee bend stop requires an interactive TTY when MANUAL_STEP_CONFIRM=True.")

    print_header("Full-Body Takeover Knee Bend")
    print(
        "双膝目标从接管姿态同步线性增加；空格停止弯曲并冻结当前目标。"
    )
    print(
        f"knee_bend_rad={args.takeover_knee_bend_rad:.4f}, "
        f"knee_bend_s={args.takeover_knee_bend_s:.2f}, "
        f"min_readback_delta={args.takeover_min_knee_delta_rad:.4f}"
    )
    start_q = controller.get_target_q()
    final_q = list(start_q)
    for index in KNEE_JOINT_INDICES:
        final_q[index] = start_q[index] + args.takeover_knee_bend_rad

    start_t = time.monotonic()
    next_print = start_t
    max_knee_delta = 0.0
    max_knee_name = "<none>"
    stopped = False

    def publish_interpolated_target(ratio: float) -> None:
        target_q = list(start_q)
        for index in KNEE_JOINT_INDICES:
            target_q[index] = start_q[index] + (final_q[index] - start_q[index]) * ratio
        controller.set_target_q(target_q)

    with RawTerminal():
        while True:
            now = time.monotonic()
            elapsed = now - start_t
            ratio = clamp(elapsed / max(args.takeover_knee_bend_s, 1.0e-6), 0.0, 1.0)
            publish_interpolated_target(ratio)
            key = read_key(0.0) if args.manual_step_confirm else None
            if key in {" ", "q", "Q", "\x1b"}:
                print("Knee bend stop key received; freezing current lowcmd target.")
                stopped = True
                break
            if key == "\x03":
                raise KeyboardInterrupt("Knee bend cancelled by Ctrl+C.")

            state = drain_lowstate(lowstate_subscriber, 0.001)
            if state is not None:
                check_takeover_stability(args, state)
                for index in KNEE_JOINT_INDICES:
                    delta = abs(get_joint_q(state, index) - start_q[index])
                    if delta > max_knee_delta:
                        max_knee_delta = delta
                        max_knee_name = FULL_BODY_JOINT_NAMES[index]
                if now >= next_print:
                    print_takeover_monitor("knee bend monitor", state, controller)
                    print(
                        f"knee bend readback: max_knee_delta={max_knee_delta:.4f} rad "
                        f"joint={max_knee_name}, ratio={ratio:.3f}"
                    )
                    next_print = now + args.takeover_monitor_period_s
            if not args.manual_step_confirm and elapsed >= args.takeover_knee_bend_s + args.takeover_auto_stop_hold_s:
                print("Non-interactive knee bend hold duration reached.")
                stopped = True
                break
            time.sleep(args.takeover_control_dt_s)

    if not stopped:
        controller.set_target_q(final_q)
    return max_knee_delta


def recover_builtin_after_takeover(
    args: argparse.Namespace,
    client: LocoClient,
    controller: FullBodyLowCmdTakeover,
    lowstate_subscriber: ChannelSubscriber,
    sport_state_subscriber: Optional[ChannelSubscriber],
) -> bool:
    if args.takeover_recover_fsm_id < 0:
        print(
            "TAKEOVER_RECOVER_FSM_ID<0: keeping full-body lowcmd publisher active. "
            "Do not stop this process until the robot is secured or another controller has taken over."
        )
        return False

    print_header("Takeover Recovery")
    print(
        "Final stop requested. Holding the current lowcmd posture while asking the built-in "
        f"controller to enter FSM {args.takeover_recover_fsm_id}."
    )
    code = client.SetFsmId(args.takeover_recover_fsm_id)
    print(f"SetFsmId({args.takeover_recover_fsm_id}) recovery: code={code}")
    if code != 0:
        raise RuntimeError(f"Recovery SetFsmId({args.takeover_recover_fsm_id}) failed with code {code}.")
    target_ids = parse_int_set(args.takeover_recover_confirm_fsm_ids)
    confirmed = poll_target_fsm(
        client,
        args.takeover_recover_timeout_s,
        sport_state_subscriber,
        target_ids,
        "recovery target=" + "/".join(str(value) for value in target_ids),
    )
    sleep_with_lowstate(lowstate_subscriber, args.takeover_recover_hold_s, "takeover recovery hold")
    if not confirmed:
        raise RuntimeError(
            "Built-in recovery FSM was not confirmed; keeping lowcmd takeover alive would be safer "
            "than silently dropping motor commands."
        )
    controller.stop()
    print("Full-body lowcmd takeover publisher stopped after built-in recovery was confirmed.")
    return True


def emergency_damp_takeover(
    args: argparse.Namespace,
    client: LocoClient,
    controller: FullBodyLowCmdTakeover,
    reason: str,
) -> None:
    print_header("Emergency Takeover Damping")
    print(f"reason={reason}")
    if controller.is_running():
        controller.enter_damping_mode(args.takeover_emergency_damping_kd)
        print(
            f"lowcmd emergency damping active: kd={args.takeover_emergency_damping_kd}, "
            f"duration={args.takeover_emergency_damping_s:.2f}s"
        )
    deadline = time.monotonic() + args.takeover_emergency_damping_s
    next_rpc_t = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_rpc_t:
            try:
                stop_code = client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s)
                damp_code = client.SetFsmId(1)
                print(f"emergency RPC: SetVelocity code={stop_code}; SetFsmId(1) Damp code={damp_code}")
            except BaseException as exc:
                print(f"emergency RPC exception: {exc}")
            next_rpc_t = now + args.safe_damp_retry_interval_s
        time.sleep(0.02)
    controller.stop()
    print("Full-body lowcmd publisher stopped after emergency damping.")


class DirectZeroArmMotorController:
    def __init__(
        self,
        args: argparse.Namespace,
        base_state: LowState_,
        commands: Sequence[ArmJointCommand],
        base_positions: Dict[int, float],
    ) -> None:
        self.args = args
        self.publisher = ChannelPublisher(args.takeover_lowcmd_topic, LowCmd_)
        self.publisher.Init()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.mode_pr = 0
        self.mode_machine = int(base_state.mode_machine)
        self.commands = list(commands)
        self.base_positions = dict(base_positions)
        self._target_positions = dict(base_positions)
        self._damping_active = False
        self._damping_kd = float(args.takeover_emergency_damping_kd)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.publish_count = 0
        self.max_publish_lag_s = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="g1_direct_zero_arm_lowcmd", daemon=False)
        self._thread.start()
        if not self._started_event.wait(timeout=self.args.takeover_warmup_timeout_s):
            raise RuntimeError("Direct zero-arm lowcmd publisher did not start in time.")

    def stop(self, join_timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(join_timeout_s)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_target_positions(self, target_positions: Dict[int, float]) -> None:
        with self._lock:
            self._target_positions = dict(target_positions)

    def enter_damping_mode(self, damping_kd: float) -> None:
        with self._lock:
            self._damping_active = True
            self._damping_kd = float(damping_kd)

    def wait_for_warmup(self) -> None:
        deadline = time.monotonic() + self.args.takeover_warmup_timeout_s
        while time.monotonic() < deadline:
            if self.publish_count >= self.args.takeover_warmup_min_cycles:
                return
            time.sleep(0.001)
        raise RuntimeError(
            "Direct zero-arm lowcmd publisher did not publish enough warm-up cycles: "
            f"count={self.publish_count}, required={self.args.takeover_warmup_min_cycles}."
        )

    def _command_gains(self, command: ArmJointCommand) -> Tuple[float, float]:
        if command.swing:
            return float(self.args.arm_kp), float(self.args.arm_kd)
        if command.index in WAIST_JOINT_INDICES:
            return float(self.args.arm_waist_hold_kp), float(self.args.arm_waist_hold_kd)
        return float(self.args.arm_hold_kp), float(self.args.arm_hold_kd)

    def _run(self) -> None:
        next_publish_t = time.monotonic()
        self._started_event.set()
        while not self._stop_event.is_set():
            now = time.monotonic()
            lag_s = max(0.0, now - next_publish_t)
            self.max_publish_lag_s = max(self.max_publish_lag_s, lag_s)
            with self._lock:
                target_positions = dict(self._target_positions)
                damping_active = self._damping_active
                damping_kd = self._damping_kd

            self.low_cmd.mode_pr = self.mode_pr
            self.low_cmd.mode_machine = self.mode_machine
            for index in FULL_BODY_JOINT_INDICES:
                motor_cmd = self.low_cmd.motor_cmd[index]
                motor_cmd.mode = 1
                motor_cmd.tau = 0.0
                motor_cmd.q = 0.0
                set_motor_zero_velocity(motor_cmd)
                motor_cmd.kp = 0.0
                motor_cmd.kd = damping_kd if damping_active else 0.0

            if not damping_active:
                for command in self.commands:
                    motor_cmd = self.low_cmd.motor_cmd[command.index]
                    motor_cmd.mode = 1
                    motor_cmd.tau = 0.0
                    motor_cmd.q = float(target_positions.get(command.index, self.base_positions[command.index]))
                    set_motor_zero_velocity(motor_cmd)
                    motor_cmd.kp, motor_cmd.kd = self._command_gains(command)

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)
            self.publish_count += 1
            next_publish_t += self.args.takeover_control_dt_s
            sleep_s = next_publish_t - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_publish_t = time.monotonic()


def print_direct_arm_motor_command_table(
    commands: Sequence[ArmJointCommand],
    base_positions: Dict[int, float],
    args: argparse.Namespace,
) -> None:
    print("direct rt/lowcmd arm motor commands:")
    print(
        "  non-commanded joints, including both legs, are published as zero torque "
        "(tau=0, kp=0, kd=0)."
    )
    for command in commands:
        role = "swing" if command.swing else "hold"
        lower, upper = ARM_JOINT_LIMITS[command.index]
        min_target = base_positions[command.index] - abs(command.amplitude_rad)
        max_target = base_positions[command.index] + abs(command.amplitude_rad)
        if command.swing:
            kp, kd = args.arm_kp, args.arm_kd
        elif command.index in WAIST_JOINT_INDICES:
            kp, kd = args.arm_waist_hold_kp, args.arm_waist_hold_kd
        else:
            kp, kd = args.arm_hold_kp, args.arm_hold_kd
        print(
            f"  {role:<5} {command.name:<27} idx={command.index:2d} "
            f"base_q={base_positions[command.index]: .4f} amp={command.amplitude_rad: .4f} "
            f"kp/kd={kp}/{kd} target_range=[{min_target: .4f},{max_target: .4f}] "
            f"limit=[{lower: .4f},{upper: .4f}]"
        )


def recover_direct_zero_arm_to_damp(
    args: argparse.Namespace,
    client: LocoClient,
    controller: DirectZeroArmMotorController,
    sport_state_subscriber: Optional[ChannelSubscriber],
    reason: str,
    strict_confirm: bool,
) -> bool:
    print_header("Direct LowCmd Safety Recovery")
    print(f"reason={reason}")
    if controller.is_running():
        controller.enter_damping_mode(args.takeover_emergency_damping_kd)
        print(
            f"direct lowcmd damping active: kd={args.takeover_emergency_damping_kd}, "
            f"duration={args.takeover_emergency_damping_s:.2f}s"
        )

    deadline = time.monotonic() + args.takeover_emergency_damping_s
    next_rpc_t = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_rpc_t:
            try:
                stop_code = client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s)
                damp_code = client.SetFsmId(1)
                print(f"direct recovery RPC: SetVelocity code={stop_code}; SetFsmId(1) Damp code={damp_code}")
            except BaseException as exc:
                print(f"direct recovery RPC exception: {exc}")
            next_rpc_t = now + args.safe_damp_retry_interval_s
        time.sleep(0.02)

    confirmed = poll_target_fsm(
        client,
        min(args.takeover_recover_timeout_s, 2.0),
        sport_state_subscriber,
        [1],
        "direct lowcmd recovery Damp FSM 1",
    )
    controller.stop()
    print("Direct lowcmd publisher stopped after safety recovery.")
    if strict_confirm and not confirmed:
        raise RuntimeError("Direct lowcmd safety recovery did not confirm FSM 1 Damp.")
    if not confirmed:
        print("WARNING: FSM 1 Damp was not confirmed after direct lowcmd safety recovery.")
    return confirmed


def run_direct_zero_arm_motor_motion(
    args: argparse.Namespace,
    controller: DirectZeroArmMotorController,
    commands: Sequence[ArmJointCommand],
    base_positions: Dict[int, float],
    first_state: LowState_,
    lowstate_subscriber: ChannelSubscriber,
) -> float:
    if args.arm_swing_s <= 0.0 and not sys.stdin.isatty():
        raise RuntimeError("Continuous direct motor arm swing requires an interactive TTY; set ARM_SWING_S>0.")

    print_header("Direct ZeroTorque Arm Motor Swing")
    print(
        f"topic={args.takeover_lowcmd_topic}, dt={args.takeover_control_dt_s:.4f}s, "
        f"frequency={args.arm_swing_frequency_hz:.3f}Hz, swing_s={args.arm_swing_s:.2f}"
    )
    print("腿部保持零力矩；目标手臂关节通过 rt/lowcmd 电机位置命令执行同一组正弦动作。")

    next_print = time.monotonic()
    arm_enable_start = time.monotonic()
    max_swing_abs_delta = 0.0
    max_swing_abs_delta_name = "<none>"
    max_swing_target_delta = 0.0
    max_swing_target_name = "<none>"
    max_hold_abs_error = 0.0
    max_hold_abs_error_name = "<none>"
    max_waist_hold_abs_error = 0.0
    max_waist_hold_abs_error_name = "<none>"
    command_indices = [command.index for command in commands]
    swing_indices = [command.index for command in commands if command.swing]
    hold_indices = [command.index for command in commands if not command.swing]
    waist_hold_indices = [index for index in hold_indices if index in WAIST_JOINT_INDICES]
    arm_readback_max_delta = {index: 0.0 for index in command_indices}
    arm_readback_max_dq = {index: 0.0 for index in command_indices}
    arm_readback_max_tau = {index: 0.0 for index in command_indices}
    leg_base_positions = {index: get_joint_q(first_state, index) for index in LEG_JOINT_INDICES}
    leg_readback_max_delta = {index: 0.0 for index in LEG_JOINT_INDICES}
    leg_readback_max_dq = {index: 0.0 for index in LEG_JOINT_INDICES}
    leg_readback_max_tau = {index: 0.0 for index in LEG_JOINT_INDICES}

    def update_cycle(motion_scale: float, motion_t: float) -> None:
        nonlocal next_print
        nonlocal max_swing_abs_delta, max_swing_abs_delta_name
        nonlocal max_swing_target_delta, max_swing_target_name
        nonlocal max_hold_abs_error, max_hold_abs_error_name
        nonlocal max_waist_hold_abs_error, max_waist_hold_abs_error_name

        target_positions: Dict[int, float] = {}
        for command in commands:
            phase = 2.0 * math.pi * args.arm_swing_frequency_hz * motion_t + command.phase_rad
            offset = command.amplitude_rad * motion_scale * math.sin(phase)
            raw_target = base_positions[command.index] + offset
            lower, upper = ARM_JOINT_LIMITS[command.index]
            target_positions[command.index] = clamp(raw_target, lower + args.arm_limit_margin_rad, upper - args.arm_limit_margin_rad)
            if command.swing:
                target_delta = abs(target_positions[command.index] - base_positions[command.index])
                if target_delta > max_swing_target_delta:
                    max_swing_target_delta = target_delta
                    max_swing_target_name = command.name
        controller.set_target_positions(target_positions)

        state = drain_lowstate(lowstate_subscriber, 0.001)
        if state is None:
            return
        check_lowstate_stability(args, state)
        update_joint_readback_stats(
            state,
            command_indices,
            base_positions,
            arm_readback_max_delta,
            arm_readback_max_dq,
            arm_readback_max_tau,
        )
        update_joint_readback_stats(
            state,
            LEG_JOINT_INDICES,
            leg_base_positions,
            leg_readback_max_delta,
            leg_readback_max_dq,
            leg_readback_max_tau,
        )
        hold_error, hold_name = get_max_hold_error(state, commands, base_positions)
        waist_hold_error, waist_hold_name = get_max_hold_error(state, commands, base_positions, WAIST_JOINT_INDICES)
        if hold_error > max_hold_abs_error:
            max_hold_abs_error = hold_error
            max_hold_abs_error_name = hold_name
        if waist_hold_error > max_waist_hold_abs_error:
            max_waist_hold_abs_error = waist_hold_error
            max_waist_hold_abs_error_name = waist_hold_name
        for command in commands:
            if not command.swing:
                continue
            swing_delta = abs(get_joint_q(state, command.index) - base_positions[command.index])
            if swing_delta > max_swing_abs_delta:
                max_swing_abs_delta = swing_delta
                max_swing_abs_delta_name = command.name

        grace_elapsed = time.monotonic() - arm_enable_start >= args.arm_hold_grace_s
        if grace_elapsed and waist_hold_error > args.arm_hold_error_threshold_rad:
            raise RuntimeError(
                f"Waist direct lowcmd hold error too large: {waist_hold_name} err={waist_hold_error:.4f} rad "
                f"> threshold={args.arm_hold_error_threshold_rad:.4f}."
            )
        if grace_elapsed and args.arm_abort_on_any_hold_error and hold_error > args.arm_hold_error_threshold_rad:
            raise RuntimeError(
                f"Non-target direct lowcmd hold error too large: {hold_name} err={hold_error:.4f} rad "
                f"> threshold={args.arm_hold_error_threshold_rad:.4f}."
            )

        now = time.monotonic()
        if now >= next_print:
            max_hold_abs_dq, max_hold_abs_dq_name = get_max_stat(hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
            max_waist_abs_dq, max_waist_abs_dq_name = get_max_stat(waist_hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
            max_leg_abs_delta, max_leg_abs_delta_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_delta, LEG_JOINT_NAMES)
            max_leg_abs_dq, max_leg_abs_dq_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_dq, LEG_JOINT_NAMES)
            print_lowstate("direct lowcmd arm monitor", state)
            print_commanded_joint_table(
                state,
                commands,
                base_positions,
                target_positions,
                "direct lowcmd commanded joint readback:",
            )
            print(
                "direct lowcmd readback summary: "
                f"max_swing_abs_delta={max_swing_abs_delta:.4f} rad joint={max_swing_abs_delta_name}; "
                f"max_swing_target_delta={max_swing_target_delta:.4f} rad joint={max_swing_target_name}; "
                f"max_hold_abs_error={max_hold_abs_error:.4f} rad joint={max_hold_abs_error_name}; "
                f"max_waist_hold_abs_error={max_waist_hold_abs_error:.4f} rad joint={max_waist_hold_abs_error_name}; "
                f"max_hold_abs_dq={max_hold_abs_dq:.4f} rad/s joint={max_hold_abs_dq_name}; "
                f"max_waist_abs_dq={max_waist_abs_dq:.4f} rad/s joint={max_waist_abs_dq_name}; "
                f"leg_max_abs_delta_q={max_leg_abs_delta:.4f} rad joint={max_leg_abs_delta_name}; "
                f"leg_max_abs_dq={max_leg_abs_dq:.4f} rad/s joint={max_leg_abs_dq_name}; "
                f"publish_count={controller.publish_count} max_publish_lag={controller.max_publish_lag_s * 1000.0:.2f}ms"
            )
            next_print = now + 1.0

    def run_timed_phase(
        duration_s: float,
        motion_start: float,
        motion_end: float,
        motion_t_start: float,
    ) -> float:
        phase_start = time.monotonic()
        last_t = motion_t_start
        while True:
            elapsed = min(time.monotonic() - phase_start, duration_s)
            ratio = clamp(elapsed / max(duration_s, 1.0e-6), 0.0, 1.0)
            motion_scale = motion_start + (motion_end - motion_start) * ratio
            last_t = motion_t_start + elapsed
            update_cycle(motion_scale, last_t)
            if elapsed >= duration_s:
                return last_t
            time.sleep(args.takeover_control_dt_s)

    last_motion_t = run_timed_phase(args.arm_ramp_s, 0.0, 0.0, 0.0)
    if args.arm_swing_s <= 0.0:
        print("Continuous direct motor arm swing: press space or q to stop.")
        swing_start = time.monotonic()
        with RawTerminal():
            while True:
                key = read_key(0.0)
                if key in {" ", "q", "Q", "\x1b"}:
                    print("Direct motor arm swing stop key received.")
                    break
                if key == "\x03":
                    raise KeyboardInterrupt("Direct motor arm swing cancelled by Ctrl+C.")
                last_motion_t = time.monotonic() - swing_start
                update_cycle(1.0, last_motion_t)
                time.sleep(args.takeover_control_dt_s)
    else:
        last_motion_t = run_timed_phase(args.arm_swing_s, 1.0, 1.0, 0.0)
    run_timed_phase(args.arm_return_s, 1.0, 0.0, last_motion_t)
    controller.set_target_positions(base_positions)

    max_hold_abs_dq, max_hold_abs_dq_name = get_max_stat(hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
    max_waist_abs_dq, max_waist_abs_dq_name = get_max_stat(waist_hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
    max_leg_abs_delta, max_leg_abs_delta_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_delta, LEG_JOINT_NAMES)
    max_leg_abs_dq, max_leg_abs_dq_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_dq, LEG_JOINT_NAMES)
    print(
        "direct lowcmd final readback summary: "
        f"max_swing_abs_delta={max_swing_abs_delta:.4f} rad joint={max_swing_abs_delta_name}; "
        f"max_swing_target_delta={max_swing_target_delta:.4f} rad joint={max_swing_target_name}; "
        f"max_hold_abs_error={max_hold_abs_error:.4f} rad joint={max_hold_abs_error_name}; "
        f"max_waist_hold_abs_error={max_waist_hold_abs_error:.4f} rad joint={max_waist_hold_abs_error_name}; "
        f"max_hold_abs_dq={max_hold_abs_dq:.4f} rad/s joint={max_hold_abs_dq_name}; "
        f"max_waist_abs_dq={max_waist_abs_dq:.4f} rad/s joint={max_waist_abs_dq_name}; "
        f"leg_max_abs_delta_q={max_leg_abs_delta:.4f} rad joint={max_leg_abs_delta_name}; "
        f"leg_max_abs_dq={max_leg_abs_dq:.4f} rad/s joint={max_leg_abs_dq_name}"
    )
    print_readback_stats_table(
        "direct lowcmd final per-joint arm/waist readback maxima:",
        command_indices,
        ARM_JOINT_NAMES,
        base_positions,
        arm_readback_max_delta,
        arm_readback_max_dq,
        arm_readback_max_tau,
    )
    print_readback_stats_table(
        "leg final read-only maxima during direct zero-arm stage:",
        LEG_JOINT_INDICES,
        LEG_JOINT_NAMES,
        leg_base_positions,
        leg_readback_max_delta,
        leg_readback_max_dq,
        leg_readback_max_tau,
    )
    return max_swing_abs_delta


def run_direct_zero_arm_motor_test(
    args: argparse.Namespace,
    client: LocoClient,
    lowstate_subscriber: ChannelSubscriber,
    sport_state_subscriber: Optional[ChannelSubscriber],
) -> None:
    print_header("Direct ZeroTorque Arm Motor Test")
    first_state = wait_latest_lowstate(lowstate_subscriber, args.lowstate_timeout_s)
    check_lowstate_stability(args, first_state)
    commands = get_arm_joint_commands(args.arm_joint_set, args.arm_amplitude_scale, args.arm_hold_non_target)
    base_positions = {command.index: get_joint_q(first_state, command.index) for command in commands}
    commands = apply_joint_limit_safety(commands, base_positions, args.arm_limit_margin_rad, args.arm_max_amplitude_rad)
    print_direct_arm_motor_command_table(commands, base_positions, args)

    controller = DirectZeroArmMotorController(args, first_state, commands, base_positions)
    recovered = False
    try:
        controller.start()
        controller.wait_for_warmup()
        print(
            "Direct lowcmd arm stream armed before ZeroTorque: "
            f"cycles={controller.publish_count}, topic={args.takeover_lowcmd_topic}, "
            f"dt={args.takeover_control_dt_s:.4f}s."
        )
        wait_for_space(
            args,
            "ZeroTorque() direct arm motor control",
            [
                "即将直接请求 FSM 0/ZeroTorque；腿部和未命令关节会进入零力矩。",
                "rt/lowcmd 线程已经在预热：目标手臂/腰臂关节保持当前 q，其它关节 tau=kp=kd=0。",
                "确认吊装/支撑已经到位；按空格后会硬确认 GetFsmId/SportModeState == 0，确认失败则立即退出并请求 Damp。",
            ],
            client=client,
            sport_state_subscriber=sport_state_subscriber,
            lowstate_subscriber=lowstate_subscriber,
        )
        require_code("ZeroTorque() direct arm motor", client.ZeroTorque())
        time.sleep(args.takeover_post_zero_settle_s)
        require_confirmed_fsm(
            client,
            args.takeover_zero_fsm_timeout_s,
            sport_state_subscriber,
            [0],
            "Direct ZeroTorque arm motor FSM 0",
        )
        max_swing_abs_delta = run_direct_zero_arm_motor_motion(
            args,
            controller,
            commands,
            base_positions,
            first_state,
            lowstate_subscriber,
        )
        recovered = recover_direct_zero_arm_to_damp(
            args,
            client,
            controller,
            sport_state_subscriber,
            "direct arm motor test finished",
            strict_confirm=True,
        )
        if args.arm_require_swing_motion and max_swing_abs_delta < args.arm_min_swing_delta_rad:
            raise RuntimeError(
                "Direct motor arm swing readback did not reach the required visible motion threshold: "
                f"max_swing_abs_delta={max_swing_abs_delta:.4f} rad < {args.arm_min_swing_delta_rad:.4f} rad."
            )
        print("Direct ZeroTorque arm motor test finished and recovered to Damp.")
    except BaseException as exc:
        if controller.is_running() and not recovered:
            recover_direct_zero_arm_to_damp(
                args,
                client,
                controller,
                sport_state_subscriber,
                repr(exc),
                strict_confirm=False,
            )
        raise
    finally:
        if controller.is_running():
            controller.stop()


def run_zero_torque_takeover_flow(
    args: argparse.Namespace,
    client: LocoClient,
    lowstate_subscriber: ChannelSubscriber,
    sport_state_subscriber: Optional[ChannelSubscriber],
) -> None:
    print_header("ZeroTorque Full-Body LowCmd Takeover")
    wait_for_space(
        args,
        "Alignment: SetVelocity(0,0,0)",
        [
            "即将让主运控停止移动输入，并依靠内置 WBC 原地站稳。",
            "随后锁存 29 个关节当前位置 q_t0，启动 500Hz rt/lowcmd 预热。",
        ],
        client=client,
        sport_state_subscriber=sport_state_subscriber,
        lowstate_subscriber=lowstate_subscriber,
    )
    require_code(
        f"SetVelocity(0,0,0,{args.zero_command_duration_s:.2f}) takeover alignment",
        client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s),
    )
    sleep_with_lowstate(lowstate_subscriber, args.takeover_alignment_s, "takeover alignment")
    base_state = wait_latest_lowstate(lowstate_subscriber, args.lowstate_timeout_s)
    check_takeover_stability(args, base_state)
    print_full_body_latch_table(base_state)

    controller = FullBodyLowCmdTakeover(args, base_state)
    recovered = False
    emergency_damped = False
    try:
        print_takeover_command_table(controller)
        controller.start()
        controller.wait_for_warmup()
        print(
            "Full-body lowcmd warm-up armed: "
            f"cycles={controller.publish_count}, target_q=q_t0, "
            f"leg_kp/kd={args.takeover_leg_kp}/{args.takeover_leg_kd}, "
            f"tau_ff_mode={args.takeover_tau_ff_mode}."
        )
        wait_for_space_with_takeover_monitor(
            args,
            "ZeroTorque() transient takeover",
            [
                "低层 500Hz 全身保持指令已经在 rt/lowcmd 预热。",
                "按空格后通过 LocoClient.ZeroTorque()/FSM 0 放开 lowcmd 总线，预热指令会在切换瞬间接管全身电机。",
                "当前自研策略动作：q_des=q_t0, dq_des=0；tau_ff 默认锁存切换前 tau_est，PD 只做小幅微调。",
            ],
            controller,
            lowstate_subscriber,
            client=client,
            sport_state_subscriber=sport_state_subscriber,
        )
        require_code("ZeroTorque() takeover", client.ZeroTorque())
        time.sleep(args.takeover_post_zero_settle_s)
        require_confirmed_fsm(
            client,
            args.takeover_zero_fsm_timeout_s,
            sport_state_subscriber,
            [0],
            "ZeroTorque FSM 0",
        )
        wait_for_space_with_takeover_monitor(
            args,
            "Confirm ZeroTorque hold",
            [
                "请确认机器人在 FSM 0 后仍被自研低层策略稳稳锁在 q_t0。",
                "等待期间 lowcmd 线程持续保持全身 29 个关节，不释放腿部。",
                "确认无误后按空格进入双膝微弯阶段。",
            ],
            controller,
            lowstate_subscriber,
            client=client,
            sport_state_subscriber=sport_state_subscriber,
        )
        wait_for_space_with_takeover_monitor(
            args,
            "Start both-knee micro bend",
            [
                "下一步会只改变 left_knee/right_knee 目标，其他 27 个关节继续保持 q_t0。",
                f"双膝目标增量={args.takeover_knee_bend_rad:.4f} rad，插值时间={args.takeover_knee_bend_s:.2f}s。",
                "开始后再次按空格停止弯曲并冻结当前目标。",
            ],
            controller,
            lowstate_subscriber,
            client=client,
            sport_state_subscriber=sport_state_subscriber,
        )
        max_knee_delta = run_takeover_knee_bend_until_space(args, controller, lowstate_subscriber)
        print(f"Knee bend phase stopped. Observed max knee delta={max_knee_delta:.4f} rad.")
        recovered = recover_builtin_after_takeover(
            args,
            client,
            controller,
            lowstate_subscriber,
            sport_state_subscriber,
        )
        if args.takeover_require_knee_motion and max_knee_delta < args.takeover_min_knee_delta_rad:
            raise RuntimeError(
                "Knee bend readback did not reach the required motion threshold: "
                f"max_knee_delta={max_knee_delta:.4f} rad < {args.takeover_min_knee_delta_rad:.4f} rad."
            )
    except BaseException as exc:
        if controller.is_running():
            emergency_damp_takeover(args, client, controller, repr(exc))
            emergency_damped = True
        raise
    finally:
        if controller.is_running() and recovered:
            controller.stop()
        elif controller.is_running() and not emergency_damped:
            emergency_damp_takeover(args, client, controller, "takeover flow exited before normal recovery")


def publish_arm_targets(
    publisher: ChannelPublisher,
    low_cmd: LowCmd_,
    crc: CRC,
    commands: Sequence[ArmJointCommand],
    target_positions: dict,
    weight: float,
    swing_kp: float,
    swing_kd: float,
    hold_kp: float,
    hold_kd: float,
    waist_hold_kp: float,
    waist_hold_kd: float,
) -> None:
    low_cmd.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = clamp(weight, 0.0, 1.0)
    for command in commands:
        motor_cmd = low_cmd.motor_cmd[command.index]
        motor_cmd.tau = 0.0
        motor_cmd.q = float(target_positions[command.index])
        set_motor_zero_velocity(motor_cmd)
        if command.swing:
            motor_cmd.kp = swing_kp
            motor_cmd.kd = swing_kd
        elif command.index in WAIST_JOINT_INDICES:
            motor_cmd.kp = waist_hold_kp
            motor_cmd.kd = waist_hold_kd
        else:
            motor_cmd.kp = hold_kp
            motor_cmd.kd = hold_kd
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


def check_lowstate_stability(args: argparse.Namespace, state: LowState_) -> None:
    snapshot = snapshot_lowstate(state)
    roll = abs(snapshot.rpy[0])
    pitch = abs(snapshot.rpy[1])
    if roll > args.arm_max_abs_roll_rad or pitch > args.arm_max_abs_pitch_rad:
        raise RuntimeError(
            "Arm DDS test aborted because body attitude exceeded limits: "
            f"roll={roll:.3f}, pitch={pitch:.3f}."
        )


def release_arm_sdk(
    publisher: ChannelPublisher,
    low_cmd: LowCmd_,
    crc: CRC,
    commands: Sequence[ArmJointCommand],
    base_positions: dict,
    swing_kp: float,
    swing_kd: float,
    hold_kp: float,
    hold_kd: float,
    waist_hold_kp: float,
    waist_hold_kd: float,
    release_s: float,
    control_dt_s: float,
) -> None:
    steps = max(1, int(release_s / control_dt_s))
    for step in range(steps + 1):
        ratio = step / steps
        publish_arm_targets(
            publisher,
            low_cmd,
            crc,
            commands,
            base_positions,
            1.0 - ratio,
            swing_kp,
            swing_kd,
            hold_kp,
            hold_kd,
            waist_hold_kp,
            waist_hold_kd,
        )
        time.sleep(control_dt_s)
    publish_arm_targets(
        publisher,
        low_cmd,
        crc,
        commands,
        base_positions,
        0.0,
        swing_kp,
        swing_kd,
        hold_kp,
        hold_kd,
        waist_hold_kp,
        waist_hold_kd,
    )


def run_arm_swing_test(
    args: argparse.Namespace,
    client: LocoClient,
    lowstate_subscriber: ChannelSubscriber,
    sport_state_subscriber: Optional[ChannelSubscriber],
) -> None:
    if not args.run_arm_swing_after_wasd:
        return

    print_header("Post-WASD Balance + Arm DDS")
    wait_for_space(
        args,
        "SetVelocity(0,0,0) before arm_sdk",
        [
            "即将退出 WASD 移动输入，先发送零速度。",
            "之后才会请求 BalanceStand 和 arm_sdk。",
        ],
        client=client,
        sport_state_subscriber=sport_state_subscriber,
        lowstate_subscriber=lowstate_subscriber,
    )
    require_code(
        f"SetVelocity(0,0,0,{args.zero_command_duration_s:.2f}) before arm_sdk",
        client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s),
    )
    if args.balance_stand_after_wasd:
        wait_for_space(
            args,
            "BalanceStand() before arm_sdk",
            [
                "即将请求宇树内置主运控进入平衡站立。",
                "这一步让下肢继续使用本体站立/平衡策略，然后上肢通过 DDS 接管。",
            ],
            client=client,
            sport_state_subscriber=sport_state_subscriber,
            lowstate_subscriber=lowstate_subscriber,
        )
        require_code("BalanceStand()", client.BalanceStand())
    sleep_with_lowstate(lowstate_subscriber, args.post_wasd_stabilize_s, "post-wasd stabilize")
    wait_arm_allowed_fsm(args, client, sport_state_subscriber)
    if args.arm_swing_s <= 0.0 and not sys.stdin.isatty():
        raise RuntimeError("Continuous arm swing requires an interactive TTY; set ARM_SWING_S>0 for a timed run.")

    commands = get_arm_joint_commands(args.arm_joint_set, args.arm_amplitude_scale, args.arm_hold_non_target)
    publisher = ChannelPublisher(args.arm_topic, LowCmd_)
    publisher.Init()
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()

    first_state = wait_latest_lowstate(lowstate_subscriber, args.lowstate_timeout_s)
    check_lowstate_stability(args, first_state)
    base_positions = {command.index: float(first_state.motor_state[command.index].q) for command in commands}
    commands = apply_joint_limit_safety(commands, base_positions, args.arm_limit_margin_rad, args.arm_max_amplitude_rad)
    print("arm_sdk joints:")
    for command in commands:
        role = "swing" if command.swing else "hold"
        lower, upper = ARM_JOINT_LIMITS[command.index]
        min_target = base_positions[command.index] - abs(command.amplitude_rad)
        max_target = base_positions[command.index] + abs(command.amplitude_rad)
        print(
            f"  {role:<5} {command.name:<27} idx={command.index:2d} "
            f"base_q={base_positions[command.index]: .4f} amp={command.amplitude_rad: .4f} "
            f"target_range=[{min_target: .4f},{max_target: .4f}] limit=[{lower: .4f},{upper: .4f}]"
        )
    wait_for_space(
        args,
        "Enable rt/arm_sdk right-arm swing",
        [
            "即将向 rt/arm_sdk 发布 LowCmd。arm_sdk 权重是上肢/腰部 SDK 接管权重。",
            "目标右臂按当前姿态附近做正弦摆动，非目标 arm_sdk 关节按启用瞬间姿态保持，避免上半身卸力。",
            f"topic={args.arm_topic}",
            f"joint_set={args.arm_joint_set}",
            f"amplitude_scale={args.arm_amplitude_scale}",
            f"limit_margin_rad={args.arm_limit_margin_rad}",
            f"max_amplitude_rad={args.arm_max_amplitude_rad}",
            f"hold_non_target={args.arm_hold_non_target}",
            f"hold_kp/kd={args.arm_hold_kp}/{args.arm_hold_kd}",
            f"waist_hold_kp/kd={args.arm_waist_hold_kp}/{args.arm_waist_hold_kd}",
            "ARM_SWING_S<=0 时持续摆动，按空格或 q 停止并平滑释放。",
            "结束时脚本会把 motor_cmd[29].q 权重插值回 0 并释放 arm_sdk。",
        ],
        client=client,
        sport_state_subscriber=sport_state_subscriber,
        lowstate_subscriber=lowstate_subscriber,
    )

    next_print = time.monotonic()
    last_motion_t = 0.0
    released = False
    arm_enable_start = time.monotonic()
    max_swing_abs_delta = 0.0
    max_swing_abs_delta_name = "<none>"
    max_swing_target_delta = 0.0
    max_swing_target_name = "<none>"
    max_hold_abs_error = 0.0
    max_hold_abs_error_name = "<none>"
    max_waist_hold_abs_error = 0.0
    max_waist_hold_abs_error_name = "<none>"
    command_indices = [command.index for command in commands]
    swing_indices = [command.index for command in commands if command.swing]
    hold_indices = [command.index for command in commands if not command.swing]
    waist_hold_indices = [index for index in hold_indices if index in WAIST_JOINT_INDICES]
    arm_readback_max_delta = {index: 0.0 for index in command_indices}
    arm_readback_max_dq = {index: 0.0 for index in command_indices}
    arm_readback_max_tau = {index: 0.0 for index in command_indices}
    leg_base_positions = {index: get_joint_q(first_state, index) for index in LEG_JOINT_INDICES}
    leg_readback_max_delta = {index: 0.0 for index in LEG_JOINT_INDICES}
    leg_readback_max_dq = {index: 0.0 for index in LEG_JOINT_INDICES}
    leg_readback_max_tau = {index: 0.0 for index in LEG_JOINT_INDICES}

    def publish_cycle(weight: float, motion_scale: float, motion_t: float) -> None:
        nonlocal next_print
        nonlocal max_swing_abs_delta, max_swing_abs_delta_name
        nonlocal max_swing_target_delta, max_swing_target_name
        nonlocal max_hold_abs_error, max_hold_abs_error_name
        nonlocal max_waist_hold_abs_error, max_waist_hold_abs_error_name
        target_positions = {}
        for command in commands:
            phase = 2.0 * math.pi * args.arm_swing_frequency_hz * motion_t + command.phase_rad
            offset = command.amplitude_rad * motion_scale * math.sin(phase)
            raw_target = base_positions[command.index] + offset
            lower, upper = ARM_JOINT_LIMITS[command.index]
            target_positions[command.index] = clamp(raw_target, lower + args.arm_limit_margin_rad, upper - args.arm_limit_margin_rad)
            if command.swing:
                target_delta = abs(target_positions[command.index] - base_positions[command.index])
                if target_delta > max_swing_target_delta:
                    max_swing_target_delta = target_delta
                    max_swing_target_name = command.name
        publish_arm_targets(
            publisher,
            low_cmd,
            crc,
            commands,
            target_positions,
            weight,
            args.arm_kp,
            args.arm_kd,
            args.arm_hold_kp,
            args.arm_hold_kd,
            args.arm_waist_hold_kp,
            args.arm_waist_hold_kd,
        )

        state = drain_lowstate(lowstate_subscriber, 0.001)
        if state is not None:
            check_lowstate_stability(args, state)
            update_joint_readback_stats(
                state,
                command_indices,
                base_positions,
                arm_readback_max_delta,
                arm_readback_max_dq,
                arm_readback_max_tau,
            )
            update_joint_readback_stats(
                state,
                LEG_JOINT_INDICES,
                leg_base_positions,
                leg_readback_max_delta,
                leg_readback_max_dq,
                leg_readback_max_tau,
            )
            hold_error, hold_name = get_max_hold_error(state, commands, base_positions)
            waist_hold_error, waist_hold_name = get_max_hold_error(state, commands, base_positions, WAIST_JOINT_INDICES)
            if hold_error > max_hold_abs_error:
                max_hold_abs_error = hold_error
                max_hold_abs_error_name = hold_name
            if waist_hold_error > max_waist_hold_abs_error:
                max_waist_hold_abs_error = waist_hold_error
                max_waist_hold_abs_error_name = waist_hold_name
            for command in commands:
                if not command.swing:
                    continue
                swing_delta = abs(get_joint_q(state, command.index) - base_positions[command.index])
                if swing_delta > max_swing_abs_delta:
                    max_swing_abs_delta = swing_delta
                    max_swing_abs_delta_name = command.name
            grace_elapsed = time.monotonic() - arm_enable_start >= args.arm_hold_grace_s
            if grace_elapsed and waist_hold_error > args.arm_hold_error_threshold_rad:
                raise RuntimeError(
                    f"Waist arm_sdk hold error too large: {waist_hold_name} err={waist_hold_error:.4f} rad "
                    f"> threshold={args.arm_hold_error_threshold_rad:.4f}."
                )
            if grace_elapsed and args.arm_abort_on_any_hold_error and hold_error > args.arm_hold_error_threshold_rad:
                raise RuntimeError(
                    f"Non-target arm_sdk hold error too large: {hold_name} err={hold_error:.4f} rad "
                    f"> threshold={args.arm_hold_error_threshold_rad:.4f}."
                )
            now = time.monotonic()
            if now >= next_print:
                max_hold_abs_dq, max_hold_abs_dq_name = get_max_stat(hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
                max_waist_abs_dq, max_waist_abs_dq_name = get_max_stat(waist_hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
                max_leg_abs_delta, max_leg_abs_delta_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_delta, LEG_JOINT_NAMES)
                max_leg_abs_dq, max_leg_abs_dq_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_dq, LEG_JOINT_NAMES)
                print_lowstate("arm_sdk monitor", state)
                print_commanded_joint_table(state, commands, base_positions, target_positions)
                print(
                    "arm_sdk readback summary: "
                    f"max_swing_abs_delta={max_swing_abs_delta:.4f} rad joint={max_swing_abs_delta_name}; "
                    f"max_swing_target_delta={max_swing_target_delta:.4f} rad joint={max_swing_target_name}; "
                    f"max_hold_abs_error={max_hold_abs_error:.4f} rad joint={max_hold_abs_error_name}; "
                    f"max_waist_hold_abs_error={max_waist_hold_abs_error:.4f} rad joint={max_waist_hold_abs_error_name}; "
                    f"max_hold_abs_dq={max_hold_abs_dq:.4f} rad/s joint={max_hold_abs_dq_name}; "
                    f"max_waist_abs_dq={max_waist_abs_dq:.4f} rad/s joint={max_waist_abs_dq_name}; "
                    f"leg_max_abs_delta_q={max_leg_abs_delta:.4f} rad joint={max_leg_abs_delta_name}; "
                    f"leg_max_abs_dq={max_leg_abs_dq:.4f} rad/s joint={max_leg_abs_dq_name}"
                )
                next_print = now + 1.0

    def run_timed_phase(duration_s: float, weight_start: float, weight_end: float, motion_start: float, motion_end: float, motion_t_start: float) -> float:
        phase_start = time.monotonic()
        last_t = motion_t_start
        while True:
            elapsed = min(time.monotonic() - phase_start, duration_s)
            ratio = clamp(elapsed / max(duration_s, 1.0e-6), 0.0, 1.0)
            weight = weight_start + (weight_end - weight_start) * ratio
            motion_scale = motion_start + (motion_end - motion_start) * ratio
            last_t = motion_t_start + elapsed
            publish_cycle(weight, motion_scale, last_t)
            if elapsed >= duration_s:
                return last_t
            time.sleep(args.arm_control_dt_s)

    try:
        last_motion_t = run_timed_phase(args.arm_ramp_s, 0.0, 1.0, 0.0, 0.0, 0.0)
        if args.arm_swing_s <= 0.0:
            print("Continuous right-arm swing: press space or q to stop and release arm_sdk.")
            swing_start = time.monotonic()
            with RawTerminal():
                while True:
                    key = read_key(0.0)
                    if key in {" ", "q", "Q", "\x1b"}:
                        print("Arm swing stop key received.")
                        break
                    if key == "\x03":
                        raise KeyboardInterrupt("Arm swing cancelled by Ctrl+C.")
                    last_motion_t = time.monotonic() - swing_start
                    publish_cycle(1.0, 1.0, last_motion_t)
                    time.sleep(args.arm_control_dt_s)
        else:
            last_motion_t = run_timed_phase(args.arm_swing_s, 1.0, 1.0, 1.0, 1.0, 0.0)
        run_timed_phase(args.arm_return_s, 1.0, 1.0, 1.0, 0.0, last_motion_t)
        release_arm_sdk(
            publisher,
            low_cmd,
            crc,
            commands,
            base_positions,
            args.arm_kp,
            args.arm_kd,
            args.arm_hold_kp,
            args.arm_hold_kd,
            args.arm_waist_hold_kp,
            args.arm_waist_hold_kd,
            args.arm_release_s,
            args.arm_control_dt_s,
        )
        released = True
        max_hold_abs_dq, max_hold_abs_dq_name = get_max_stat(hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
        max_waist_abs_dq, max_waist_abs_dq_name = get_max_stat(waist_hold_indices, arm_readback_max_dq, ARM_JOINT_NAMES)
        max_leg_abs_delta, max_leg_abs_delta_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_delta, LEG_JOINT_NAMES)
        max_leg_abs_dq, max_leg_abs_dq_name = get_max_stat(LEG_JOINT_INDICES, leg_readback_max_dq, LEG_JOINT_NAMES)
        print(
            "arm_sdk final readback summary: "
            f"max_swing_abs_delta={max_swing_abs_delta:.4f} rad joint={max_swing_abs_delta_name}; "
            f"max_swing_target_delta={max_swing_target_delta:.4f} rad joint={max_swing_target_name}; "
            f"max_hold_abs_error={max_hold_abs_error:.4f} rad joint={max_hold_abs_error_name}; "
            f"max_waist_hold_abs_error={max_waist_hold_abs_error:.4f} rad joint={max_waist_hold_abs_error_name}; "
            f"max_hold_abs_dq={max_hold_abs_dq:.4f} rad/s joint={max_hold_abs_dq_name}; "
            f"max_waist_abs_dq={max_waist_abs_dq:.4f} rad/s joint={max_waist_abs_dq_name}; "
            f"leg_max_abs_delta_q={max_leg_abs_delta:.4f} rad joint={max_leg_abs_delta_name}; "
            f"leg_max_abs_dq={max_leg_abs_dq:.4f} rad/s joint={max_leg_abs_dq_name}"
        )
        print_readback_stats_table(
            "arm_sdk final per-joint readback maxima:",
            command_indices,
            ARM_JOINT_NAMES,
            base_positions,
            arm_readback_max_delta,
            arm_readback_max_dq,
            arm_readback_max_tau,
        )
        print_readback_stats_table(
            "leg final read-only maxima during arm_sdk stage:",
            LEG_JOINT_INDICES,
            LEG_JOINT_NAMES,
            leg_base_positions,
            leg_readback_max_delta,
            leg_readback_max_dq,
            leg_readback_max_tau,
        )
        if args.arm_require_swing_motion and max_swing_abs_delta < args.arm_min_swing_delta_rad:
            raise RuntimeError(
                "Arm swing readback did not reach the required visible motion threshold: "
                f"max_swing_abs_delta={max_swing_abs_delta:.4f} rad < {args.arm_min_swing_delta_rad:.4f} rad."
            )
        print("arm_sdk swing finished and released.")
    finally:
        if not released:
            print("Releasing arm_sdk after interruption or error.")
            release_arm_sdk(
                publisher,
                low_cmd,
                crc,
                commands,
                base_positions,
                args.arm_kp,
                args.arm_kd,
                args.arm_hold_kp,
                args.arm_hold_kd,
                args.arm_waist_hold_kp,
                args.arm_waist_hold_kd,
                min(args.arm_release_s, 0.5),
                args.arm_control_dt_s,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded Unitree G1 Loco FSM handoff and WASD control.")
    parser.add_argument("net", help="Unitree DDS network interface, e.g. enp11s0.")
    parser.add_argument("--loco-service", default="sport", help="RPC service name. Default: sport.")
    parser.add_argument("--start-service", action="append", default=[], help="robot_state service to start before Loco RPC, e.g. ai_sport.")
    parser.add_argument("--rpc-timeout-s", type=float, default=5.0)
    parser.add_argument("--service-start-wait-s", type=float, default=1.0)
    parser.add_argument("--confirm-real-robot", default="", help="Must be I_UNDERSTAND for any command-sending mode.")
    parser.add_argument("--manual-step-confirm", action="store_true", help="Require pressing Space before each robot state-transition step.")
    parser.add_argument("--probe-only", action="store_true", help="Only verify connection and Loco RPC; do not send FSM or velocity commands.")
    parser.add_argument("--verify-zero-command", action="store_true", help="Send SetVelocity(0,0,0) after RPC preflight to prove command delivery.")
    parser.add_argument("--lowstate-timeout-s", type=float, default=3.0)
    parser.add_argument("--sport-state-topic", default="rt/lf/sportmodestate")
    parser.add_argument("--sport-state-timeout-s", type=float, default=1.5)
    parser.add_argument("--allow-missing-sport-state", action="store_true")
    parser.add_argument("--zero-command-duration-s", type=float, default=0.2)
    parser.add_argument("--safe-damp-on-exit", action="store_true", help="Best-effort SetFsmId(1) damping in the top-level finally block.")
    parser.add_argument("--safe-damp-rpc-retries", type=int, default=3)
    parser.add_argument("--safe-damp-retry-interval-s", type=float, default=0.2)
    parser.add_argument("--damp-hold-s", type=float, default=0.8)
    parser.add_argument("--handoff-flow", choices=["squat2stand_auto", "standup_start"], default="squat2stand_auto")
    parser.add_argument("--locomotion-fsm-ids", default=DEFAULT_LOCOMOTION_FSM_IDS)
    parser.add_argument("--start-fsm-ids", default=DEFAULT_LOCOMOTION_FSM_IDS)
    parser.add_argument("--standup-wait-s", type=float, default=6.0)
    parser.add_argument("--fsm-poll-timeout-s", type=float, default=5.0)
    parser.add_argument("--set-locomotion-after-standup", action="store_true", help="Request locomotion candidates again if auto switch is not observed.")
    parser.add_argument("--allow-missing-fsm-500", action="store_true", help="Continue to WASD even if GetFsmId does not confirm a locomotion candidate.")
    parser.add_argument("--run-zero-torque-takeover", action="store_true", help="After locomotion handoff, warm up full-body rt/lowcmd, switch to FSM 0, hold q_t0, and micro-bend both knees.")
    parser.add_argument("--direct-zero-arm-motor-test", action="store_true", help="Skip locomotion handoff, enter FSM 0, and drive selected/held arm-scope joints directly through rt/lowcmd while legs and uncommanded joints are zero torque.")
    parser.add_argument("--takeover-lowcmd-topic", default="rt/lowcmd")
    parser.add_argument("--takeover-control-dt-s", type=float, default=0.002)
    parser.add_argument("--takeover-warmup-min-cycles", type=int, default=50)
    parser.add_argument("--takeover-warmup-timeout-s", type=float, default=1.0)
    parser.add_argument("--takeover-alignment-s", type=float, default=1.0)
    parser.add_argument("--takeover-post-zero-settle-s", type=float, default=0.2)
    parser.add_argument("--takeover-zero-fsm-timeout-s", type=float, default=2.0)
    parser.add_argument("--takeover-monitor-period-s", type=float, default=1.0)
    parser.add_argument("--takeover-tau-ff-mode", choices=["latched", "zero"], default="latched")
    parser.add_argument("--takeover-tau-ff-scale", type=float, default=1.0)
    parser.add_argument("--takeover-max-abs-tau-ff", type=float, default=120.0)
    parser.add_argument("--takeover-leg-kp", type=float, default=25.0)
    parser.add_argument("--takeover-leg-kd", type=float, default=1.0)
    parser.add_argument("--takeover-waist-kp", type=float, default=20.0)
    parser.add_argument("--takeover-waist-kd", type=float, default=1.0)
    parser.add_argument("--takeover-hold-kp", type=float, default=15.0)
    parser.add_argument("--takeover-hold-kd", type=float, default=0.8)
    parser.add_argument("--takeover-knee-bend-rad", type=float, default=0.06)
    parser.add_argument("--takeover-knee-bend-s", type=float, default=1.0)
    parser.add_argument("--takeover-auto-stop-hold-s", type=float, default=1.0)
    parser.add_argument("--takeover-require-knee-motion", action="store_true")
    parser.add_argument("--takeover-min-knee-delta-rad", type=float, default=0.02)
    parser.add_argument("--takeover-recover-fsm-id", type=int, default=1)
    parser.add_argument("--takeover-recover-confirm-fsm-ids", default="1")
    parser.add_argument("--takeover-recover-timeout-s", type=float, default=5.0)
    parser.add_argument("--takeover-recover-hold-s", type=float, default=1.0)
    parser.add_argument("--takeover-emergency-damping-s", type=float, default=1.0)
    parser.add_argument("--takeover-emergency-damping-kd", type=float, default=8.0)
    parser.add_argument("--takeover-max-abs-roll-rad", type=float, default=0.35)
    parser.add_argument("--takeover-max-abs-pitch-rad", type=float, default=0.35)
    parser.add_argument("--speed-mps", type=float, default=0.5, help="WASD linear speed magnitude.")
    parser.add_argument("--command-duration-s", type=float, default=1.0, help="Duration for each WASD velocity command.")
    parser.add_argument("--keyboard-poll-s", type=float, default=0.05)
    parser.add_argument("--keyboard-seconds", type=float, default=0.0, help="0 means run until q.")
    parser.add_argument("--wasd-repeat-s", type=float, default=0.25, help="Interval for resending the latched WASD velocity command.")
    parser.add_argument("--auto-forward-test", action="store_true", help="Run a non-interactive forward command and verify leg readback before the arm stage.")
    parser.add_argument("--auto-forward-speed-mps", type=float, default=0.5)
    parser.add_argument("--auto-forward-duration-s", type=float, default=3.0)
    parser.add_argument("--auto-command-period-s", type=float, default=0.2)
    parser.add_argument("--auto-forward-min-leg-dq", type=float, default=0.10)
    parser.add_argument("--auto-forward-min-leg-delta-q", type=float, default=0.02)
    parser.add_argument("--auto-forward-require-motion", action="store_true")
    parser.add_argument("--auto-forward-continuous-gait", action="store_true")
    parser.add_argument("--arm-only-test", action="store_true", help="Skip FSM/WASD/forward stages and run only the guarded arm_sdk test from the current built-in standing mode.")
    parser.add_argument("--run-arm-swing-after-wasd", action="store_true")
    parser.add_argument("--balance-stand-after-wasd", action="store_true", help="Send BalanceStand() after WASD and before arm_sdk.")
    parser.add_argument("--post-wasd-stabilize-s", type=float, default=1.0)
    parser.add_argument("--arm-topic", default="rt/arm_sdk")
    parser.add_argument(
        "--arm-joint-set",
        choices=[
            "arm5",
            "arm7",
            "both_arm5",
            "both_arm7",
            "right_arm5",
            "right_arm7",
            "left_arm5",
            "left_arm7",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
        ],
        default="right_arm5",
    )
    parser.add_argument("--arm-hold-non-target", action="store_true", help="Hold non-target arm_sdk joints at their current q while the target arm swings.")
    parser.add_argument("--arm-allowed-fsm-ids", default=DEFAULT_ARM_ALLOWED_FSM_IDS)
    parser.add_argument("--arm-require-static-fsm-mode", action="store_true")
    parser.add_argument("--arm-fsm-timeout-s", type=float, default=3.0)
    parser.add_argument("--arm-control-dt-s", type=float, default=0.02)
    parser.add_argument("--arm-ramp-s", type=float, default=1.0)
    parser.add_argument("--arm-swing-s", type=float, default=0.0, help="Swing duration; <=0 means keep swinging until space/q.")
    parser.add_argument("--arm-return-s", type=float, default=1.0)
    parser.add_argument("--arm-release-s", type=float, default=0.8)
    parser.add_argument("--arm-swing-frequency-hz", type=float, default=0.35)
    parser.add_argument("--arm-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--arm-kp", type=float, default=40.0)
    parser.add_argument("--arm-kd", type=float, default=1.0)
    parser.add_argument("--arm-hold-kp", type=float, default=120.0)
    parser.add_argument("--arm-hold-kd", type=float, default=5.0)
    parser.add_argument("--arm-waist-hold-kp", type=float, default=160.0)
    parser.add_argument("--arm-waist-hold-kd", type=float, default=6.0)
    parser.add_argument("--arm-hold-error-threshold-rad", type=float, default=0.15)
    parser.add_argument("--arm-hold-grace-s", type=float, default=1.5)
    parser.add_argument("--arm-abort-on-any-hold-error", action="store_true")
    parser.add_argument("--arm-require-swing-motion", action="store_true")
    parser.add_argument("--arm-min-swing-delta-rad", type=float, default=0.20)
    parser.add_argument("--arm-limit-margin-rad", type=float, default=0.15)
    parser.add_argument("--arm-max-amplitude-rad", type=float, default=0.9)
    parser.add_argument("--arm-max-abs-roll-rad", type=float, default=0.35)
    parser.add_argument("--arm-max-abs-pitch-rad", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    install_signal_handlers()
    args = parse_args()
    will_send_command = args.verify_zero_command or not args.probe_only or len(args.start_service) > 0
    if will_send_command and args.confirm_real_robot != CONFIRM_TOKEN:
        raise SystemExit(f"Refusing to send robot commands without --confirm-real-robot {CONFIRM_TOKEN}.")

    print_header("Unitree G1 Loco FSM WASD")
    print(f"net={args.net}")
    print(f"loco_service={args.loco_service}")
    print(f"probe_only={args.probe_only}")
    print(f"verify_zero_command={args.verify_zero_command}")
    print(f"start_service={args.start_service}")
    print(f"handoff_flow={args.handoff_flow}")
    print(f"locomotion_fsm_ids={args.locomotion_fsm_ids}")
    print(f"start_fsm_ids={args.start_fsm_ids}")
    print(f"manual_step_confirm={args.manual_step_confirm}")
    print(f"arm_only_test={args.arm_only_test}")
    print(f"run_arm_swing_after_wasd={args.run_arm_swing_after_wasd}")
    print(f"run_zero_torque_takeover={args.run_zero_torque_takeover}")
    print(f"direct_zero_arm_motor_test={args.direct_zero_arm_motor_test}")

    ChannelFactoryInitialize(0, args.net)
    lowstate_subscriber, _ = wait_lowstate(args.lowstate_timeout_s)
    sport_state_subscriber, sport_state = wait_sport_state(args.sport_state_topic, args.sport_state_timeout_s)
    motion_code, _ = print_motion_switcher_state()
    if motion_code != 0:
        raise RuntimeError(f"MotionSwitcher CheckMode failed with code {motion_code}")
    print_robot_state_services(args)

    loco_client = LocoClient(service_name=args.loco_service)
    loco_client.SetTimeout(args.rpc_timeout_s)
    loco_client.Init()
    loco_available = print_loco_rpc_state(loco_client, strict=(not args.probe_only or args.verify_zero_command))

    if args.verify_zero_command:
        verify_zero_velocity(args, loco_client)

    if args.probe_only:
        if not loco_available:
            print("Probe finished with Loco RPC unavailable; this is expected in L2+R2 debug mode.")
        else:
            print("Probe finished with Loco RPC available.")
        print("Probe finished successfully.")
        return

    if sport_state is None and not args.allow_missing_sport_state and not args.direct_zero_arm_motor_test:
        raise RuntimeError(
            "No SportModeState DDS sample was received. The high-level built-in loco controller is likely not active "
            "or the robot is still in L2+R2 debug mode; refusing to send high-level handoff commands. "
            "Use low-level deploy in debug mode, or exit debug mode and use the built-in loco service."
        )
    if sport_state is None and args.direct_zero_arm_motor_test:
        print("WARNING: SportModeState was not received; direct test will rely on LocoClient.GetFsmId() for FSM 0/1 confirmation.")

    try:
        if args.direct_zero_arm_motor_test:
            run_direct_zero_arm_motor_test(
                args,
                loco_client,
                lowstate_subscriber,
                sport_state_subscriber,
            )
        elif args.arm_only_test:
            print_header("Arm-Only DDS Test")
            run_arm_swing_test(args, loco_client, lowstate_subscriber, sport_state_subscriber)
        else:
            run_fsm_handoff(args, loco_client, lowstate_subscriber, sport_state_subscriber)
            if args.run_zero_torque_takeover:
                run_zero_torque_takeover_flow(
                    args,
                    loco_client,
                    lowstate_subscriber,
                    sport_state_subscriber,
                )
            elif args.auto_forward_test:
                run_auto_forward_test(args, loco_client, lowstate_subscriber, sport_state_subscriber)
            else:
                wait_for_space(
                    args,
                    "Enter WASD keyboard control",
                    [
                        "即将进入实时键盘速度控制。",
                        "w/s/a/d 锁定并持续重发速度，空格发送 StopMove，q 退出 WASD 并进入下一阶段。",
                    ],
                    client=loco_client,
                    sport_state_subscriber=sport_state_subscriber,
                    lowstate_subscriber=lowstate_subscriber,
                )
                run_wasd_loop(args, loco_client)
            if not args.run_zero_torque_takeover:
                run_arm_swing_test(args, loco_client, lowstate_subscriber, sport_state_subscriber)
    finally:
        print_header("Final Stop")
        try:
            code = loco_client.SetVelocity(0.0, 0.0, 0.0, args.zero_command_duration_s)
            print(f"SetVelocity(0,0,0,{args.zero_command_duration_s:.2f}) final code={code}")
        except BaseException as exc:
            print(f"Final SetVelocity exception: {exc}")
        if args.safe_damp_on_exit and not args.probe_only:
            best_effort_loco_damping(args, loco_client, "Final Best-Effort Damp")


if __name__ == "__main__":
    main()
