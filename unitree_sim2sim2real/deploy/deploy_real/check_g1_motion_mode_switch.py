"""Check whether a Unitree G1 can switch from debug/low-level mode to native locomotion mode.

The test is intentionally conservative: it does not request StandUp, walking, or
arm motion. It only selects the high-level MotionSwitcher mode, starts/queries
the native ai_sport service, and verifies real-time state streams/RPC getters.
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as idl_types

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.rpc.internal import RPC_ERR_CLIENT_SEND


CONFIRM_TOKEN = "I_UNDERSTAND"
DEFAULT_SPORT_TOPICS = ("rt/lf/sportmodestate", "rt/sportmodestate")


@dataclass
@annotate.final
@annotate.autoid("sequential")
class G1SportModeState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.SportModeState_"):
    fsm_id: idl_types.uint32
    fsm_mode: idl_types.uint32
    task_id: idl_types.uint32
    task_time: idl_types.float32


def decode_rpc_value(data: object) -> object:
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def rpc_value_as_int(data: object) -> Optional[int]:
    decoded = decode_rpc_value(data)
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


def mode_name(result: object) -> str:
    if isinstance(result, dict):
        return str(result.get("name", "") or "")
    return ""


def print_lowstate(prefix: str, state: LowState_) -> None:
    rpy = ", ".join(f"{float(value):.4f}" for value in state.imu_state.rpy)
    print(
        f"{prefix}: tick={int(state.tick)} mode_pr={int(state.mode_pr)} "
        f"mode_machine={int(state.mode_machine)} rpy=[{rpy}] "
        f"motor0(q={float(state.motor_state[0].q):.4f}, "
        f"dq={float(state.motor_state[0].dq):.4f}, "
        f"tau={float(state.motor_state[0].tau_est):.4f})"
    )


def wait_lowstate(timeout_s: float) -> Tuple[ChannelSubscriber, LowState_]:
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = subscriber.Read(0.2)
        if state is not None:
            print_lowstate("lowstate", state)
            return subscriber, state
    raise RuntimeError(f"No rt/lowstate sample received within {timeout_s:.1f}s.")


def read_sport_state_once(topic: str, timeout_s: float) -> Tuple[ChannelSubscriber, Optional[G1SportModeState_]]:
    subscriber = ChannelSubscriber(topic, G1SportModeState_)
    subscriber.Init()
    state = subscriber.Read(timeout_s)
    if state is None:
        print(f"{topic}: no SportModeState sample within {timeout_s:.1f}s")
        return subscriber, None
    print(
        f"{topic}: fsm_id={int(state.fsm_id)} fsm_mode={int(state.fsm_mode)} "
        f"task_id={int(state.task_id)} task_time={float(state.task_time):.3f}"
    )
    return subscriber, state


def print_service_list(client: RobotStateClient) -> Tuple[int, dict]:
    code, services = client.ServiceList()
    print("RobotState.ServiceList:")
    print(f"  code={code}")
    service_status = {}
    if code != 0 or services is None:
        return code, service_status
    for service in services:
        service_status[service.name] = int(service.status)
        if service.name in {"ai_sport", "sport_mode", "motion_switcher", "robot_state"}:
            print(f"  {service.name}: status={int(service.status)} protect={service.protect}")
    return code, service_status


def print_loco_getters(client: LocoClient) -> dict:
    getters: Sequence[Tuple[str, Callable[[], Tuple[int, object]]]] = [
        ("version", client.GetServerApiVersion),
        ("fsm_id", client.GetFsmId),
        ("fsm_mode", client.GetFsmMode),
        ("balance_mode", client.GetBalanceMode),
        ("stand_height", client.GetStandHeight),
    ]
    observed = {}
    print("LocoClient RPC:")
    for name, getter in getters:
        try:
            code, value = getter()
        except BaseException as exc:
            print(f"  {name:<12}: exception={exc}")
            observed[name] = (None, None)
            continue
        decoded = decode_rpc_value(value)
        print(f"  {name:<12}: code={code} value={decoded}")
        observed[name] = (code, decoded)
    return observed


def poll_motion_mode(
    motion_switcher: MotionSwitcherClient,
    target_mode: str,
    timeout_s: float,
) -> Tuple[bool, Optional[dict]]:
    deadline = time.monotonic() + timeout_s
    latest = None
    while time.monotonic() < deadline:
        code, result = motion_switcher.CheckMode()
        latest = result
        print(f"MotionSwitcher.CheckMode poll: code={code} data={result}")
        if code == 0 and mode_name(result) == target_mode:
            return True, result
        time.sleep(0.5)
    return False, latest


def poll_loco_available(client: LocoClient, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code, version = client.GetServerApiVersion()
        print(f"Loco GetServerApiVersion poll: code={code} value={decode_rpc_value(version)}")
        if code == 0:
            return True
        if code == RPC_ERR_CLIENT_SEND:
            print("  code 3102 means no matched sport RPC server subscription yet.")
        time.sleep(0.5)
    return False


def poll_loco_fsm_valid(client: LocoClient, timeout_s: float) -> Tuple[bool, Optional[int], Optional[int]]:
    deadline = time.monotonic() + timeout_s
    latest_fsm_id = None
    latest_fsm_mode = None
    while time.monotonic() < deadline:
        fsm_code, fsm_value = client.GetFsmId()
        fsm_mode_code, fsm_mode_value = client.GetFsmMode()
        latest_fsm_id = rpc_value_as_int(fsm_value)
        latest_fsm_mode = rpc_value_as_int(fsm_mode_value)
        print(
            "Loco FSM poll: "
            f"GetFsmId code={fsm_code} value={decode_rpc_value(fsm_value)}; "
            f"GetFsmMode code={fsm_mode_code} value={decode_rpc_value(fsm_mode_value)}"
        )
        if fsm_code == 0 and latest_fsm_id is not None and fsm_mode_code == 0 and latest_fsm_mode is not None:
            return True, latest_fsm_id, latest_fsm_mode
        time.sleep(0.5)
    return False, latest_fsm_id, latest_fsm_mode


def poll_sport_state(
    subscribers: Sequence[Tuple[str, ChannelSubscriber]],
    timeout_s: float,
) -> Optional[Tuple[str, G1SportModeState_]]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for topic, subscriber in subscribers:
            state = subscriber.Read(0.05)
            if state is not None:
                print(
                    f"SportModeState poll {topic}: fsm_id={int(state.fsm_id)} "
                    f"fsm_mode={int(state.fsm_mode)} task_id={int(state.task_id)} "
                    f"task_time={float(state.task_time):.3f}"
                )
                return topic, state
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch G1 from debug/low-level mode to native high-level locomotion mode and verify live readings.")
    parser.add_argument("net", help="Unitree DDS network interface, e.g. enp11s0.")
    parser.add_argument("--target-mode", default="ai", help="MotionSwitcher target mode name. Default: ai.")
    parser.add_argument("--start-service", default="ai_sport", help="robot_state service to start after SelectMode. Empty disables.")
    parser.add_argument("--loco-service", default="sport", help="G1 Loco RPC service name. Default: sport.")
    parser.add_argument("--confirm-switch", default="", help=f"Must be {CONFIRM_TOKEN} when --switch-to-target is used.")
    parser.add_argument("--switch-to-target", action="store_true", help="Actually call MotionSwitcher.SelectMode(target).")
    parser.add_argument("--lowstate-timeout-s", type=float, default=3.0)
    parser.add_argument("--sport-state-timeout-s", type=float, default=1.0)
    parser.add_argument("--post-select-timeout-s", type=float, default=8.0)
    parser.add_argument("--service-wait-s", type=float, default=1.0)
    parser.add_argument("--sport-topic", action="append", default=[], help="SportModeState topic to probe. Can be repeated.")
    parser.add_argument("--verify-zero-command", action="store_true", help="After success, send a zero-velocity Loco command.")
    args = parser.parse_args()

    if args.switch_to_target and args.confirm_switch != CONFIRM_TOKEN:
        raise SystemExit(f"Refusing to switch modes without --confirm-switch {CONFIRM_TOKEN}.")

    sport_topics = args.sport_topic or list(DEFAULT_SPORT_TOPICS)

    ChannelFactoryInitialize(0, args.net)
    print("=====================================")
    print("  G1 Motion Mode Switch Check")
    print("=====================================")
    print(f"net={args.net}")
    print(f"target_mode={args.target_mode}")
    print(f"start_service={args.start_service or '<disabled>'}")
    print(f"switch_to_target={args.switch_to_target}")
    print(f"sport_topics={sport_topics}")
    print("=====================================")

    wait_lowstate(args.lowstate_timeout_s)

    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()
    code, initial_mode = motion_switcher.CheckMode()
    print(f"Initial MotionSwitcher.CheckMode: code={code} data={initial_mode}")
    initial_name = mode_name(initial_mode)
    if code == 0 and not initial_name:
        print("Initial diagnosis: MotionSwitcher name is empty; this matches debug/low-level released mode.")
    elif code == 0:
        print(f"Initial diagnosis: MotionSwitcher is already owning mode '{initial_name}'.")

    robot_state = RobotStateClient()
    robot_state.SetTimeout(5.0)
    robot_state.Init()
    print_service_list(robot_state)

    initial_sport_subscribers = []
    for topic in sport_topics:
        subscriber, _ = read_sport_state_once(topic, args.sport_state_timeout_s)
        initial_sport_subscribers.append((topic, subscriber))

    loco_client = LocoClient(service_name=args.loco_service)
    loco_client.SetTimeout(5.0)
    loco_client.Init()
    initial_loco = print_loco_getters(loco_client)

    if args.switch_to_target:
        print("=====================================")
        print("  Select Native Motion Mode")
        print("=====================================")
        select_code, _ = motion_switcher.SelectMode(args.target_mode)
        print(f"MotionSwitcher.SelectMode({args.target_mode!r}): code={select_code}")
        if select_code != 0:
            raise RuntimeError(f"SelectMode({args.target_mode}) failed with code {select_code}.")
        mode_ok, final_mode = poll_motion_mode(motion_switcher, args.target_mode, args.post_select_timeout_s)
        if not mode_ok:
            raise RuntimeError(f"MotionSwitcher did not report target mode {args.target_mode!r}; latest={final_mode}.")

        if args.start_service:
            service_code = robot_state.ServiceSwitch(args.start_service, True)
            print(f"RobotState.ServiceSwitch({args.start_service!r}, True): code={service_code}")
            if service_code != 0:
                raise RuntimeError(f"ServiceSwitch({args.start_service}, True) failed with code {service_code}.")
            if args.service_wait_s > 0.0:
                time.sleep(args.service_wait_s)
        print_service_list(robot_state)

        loco_ok = poll_loco_available(loco_client, args.post_select_timeout_s)
        if not loco_ok:
            raise RuntimeError("Loco RPC did not become available after switching to native motion mode.")
        fsm_ok, fsm_id, fsm_mode = poll_loco_fsm_valid(loco_client, args.post_select_timeout_s)
        final_loco = print_loco_getters(loco_client)

        sport_subscribers = []
        for topic in sport_topics:
            subscriber = ChannelSubscriber(topic, G1SportModeState_)
            subscriber.Init()
            sport_subscribers.append((topic, subscriber))
        sport_observed = poll_sport_state(sport_subscribers, args.post_select_timeout_s)
        if sport_observed is None:
            raise RuntimeError("No SportModeState sample was observed after switching to native motion mode.")

        if not fsm_ok or fsm_id is None or fsm_mode is None:
            raise RuntimeError(
                "Loco GetFsmId/GetFsmMode did not become valid after native mode switch; "
                f"latest_fsm_id={fsm_id}, latest_fsm_mode={fsm_mode}."
            )

        if args.verify_zero_command:
            zero_code = loco_client.SetVelocity(0.0, 0.0, 0.0, 0.2)
            print(f"Loco SetVelocity(0,0,0,0.2) verification: code={zero_code}")
            if zero_code != 0:
                raise RuntimeError(f"Zero velocity verification failed with code {zero_code}.")

        print("=====================================")
        print("  RESULT: SUCCESS")
        print("=====================================")
        print(
            "Native high-level motion mode is reachable: "
            f"MotionSwitcher.name={args.target_mode!r}, Loco fsm_id={fsm_id}, fsm_mode={fsm_mode}, "
            f"SportModeState topic={sport_observed[0]} fsm_id={int(sport_observed[1].fsm_id)}."
        )
    else:
        print("=====================================")
        print("  RESULT: READ-ONLY")
        print("=====================================")
        if initial_loco.get("version", (None, None))[0] == 0:
            print("Loco RPC is already available.")
        else:
            print("Loco RPC is not available in the initial reading.")


if __name__ == "__main__":
    main()
