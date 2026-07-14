"""Interactive Unitree G1 mode monitor and switch console.

The console continuously reads MotionSwitcher, robot_state service status,
Loco FSM, SportModeState, and rt/lowstate. When explicitly confirmed it also
offers keyboard actions for the mode switches that were validated on the local
G1 firmware.
"""

import argparse
import json
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as idl_types

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


CONFIRM_TOKEN = "I_UNDERSTAND"
DEFAULT_SPORT_TOPICS = ("rt/lf/sportmodestate", "rt/sportmodestate")
LOCOMOTION_FSM_IDS = (802, 801, 500)
FSM_NAMES = {
    0: "ZeroTorque",
    1: "Damp",
    2: "PositionSquat",
    3: "PositionSit",
    4: "StandUp",
    500: "Locomotion",
    501: "Locomotion-3DoF-waist",
    702: "Lie2Stand",
    706: "Squat2Stand",
    801: "WalkRun",
    802: "WalkRun-29DoF",
}


@dataclass
@annotate.final
@annotate.autoid("sequential")
class G1SportModeState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.SportModeState_"):
    fsm_id: idl_types.uint32
    fsm_mode: idl_types.uint32
    task_id: idl_types.uint32
    task_time: idl_types.float32


@dataclass
class Snapshot:
    t: float
    motion_code: Optional[int] = None
    motion_data: object = None
    motion_name: str = ""
    service_code: Optional[int] = None
    service_status: Optional[Dict[str, int]] = None
    loco_version_code: Optional[int] = None
    loco_version: object = None
    fsm_code: Optional[int] = None
    fsm_id: Optional[int] = None
    fsm_mode_code: Optional[int] = None
    fsm_mode: Optional[int] = None
    low_tick: Optional[int] = None
    mode_pr: Optional[int] = None
    mode_machine: Optional[int] = None
    rpy: Optional[Tuple[float, float, float]] = None
    sport_topic: str = ""
    sport_fsm_id: Optional[int] = None
    sport_fsm_mode: Optional[int] = None
    sport_task_id: Optional[int] = None
    sport_task_time: Optional[float] = None


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
        try:
            return int(decoded.strip())
        except ValueError:
            return None
    return None


def mode_name(result: object) -> str:
    if isinstance(result, dict):
        return str(result.get("name", "") or "")
    return ""


def fsm_label(fsm_id: Optional[int]) -> str:
    if fsm_id is None:
        return "unknown"
    return FSM_NAMES.get(int(fsm_id), f"unknown-{fsm_id}")


class RawTerminal:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self.old_attrs = None

    def __enter__(self) -> "RawTerminal":
        if self.enabled:
            self.old_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.old_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_attrs)

    def read_key(self) -> str:
        if not self.enabled:
            return ""
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return ""
        return sys.stdin.read(1)


class ModeConsole:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.t0 = time.monotonic()
        self.logs: List[str] = []
        self.motion_switcher = MotionSwitcherClient()
        self.motion_switcher.SetTimeout(args.rpc_timeout_s)
        self.motion_switcher.Init()
        self.robot_state = RobotStateClient()
        self.robot_state.SetTimeout(args.rpc_timeout_s)
        self.robot_state.Init()
        self.loco = LocoClient(service_name=args.loco_service)
        self.loco.SetTimeout(args.rpc_timeout_s)
        self.loco.Init()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init()
        self.sport_subscribers = []
        for topic in args.sport_topic or list(DEFAULT_SPORT_TOPICS):
            subscriber = ChannelSubscriber(topic, G1SportModeState_)
            subscriber.Init()
            self.sport_subscribers.append((topic, subscriber))
        self.last_lowstate: Optional[LowState_] = None
        self.last_sport: Optional[Tuple[str, G1SportModeState_]] = None
        self.last_snapshot: Optional[Snapshot] = None

    def now(self) -> float:
        return time.monotonic() - self.t0

    def log(self, message: str) -> None:
        line = f"[{self.now():8.3f}s] {message}"
        self.logs.append(line)
        self.logs = self.logs[-self.args.log_lines :]
        if self.args.action != "none" or not sys.stdin.isatty():
            print(line, flush=True)

    def call(self, label: str, func: Callable[[], object]) -> object:
        start = time.monotonic()
        self.log(f"CALL {label}")
        try:
            result = func()
        except BaseException as exc:
            self.log(f"ERR  {label} after {time.monotonic() - start:.3f}s -> {type(exc).__name__}: {exc}")
            raise
        self.log(f"RET  {label} after {time.monotonic() - start:.3f}s -> {result}")
        return result

    def require_commands(self, label: str) -> bool:
        if self.args.allow_commands:
            return True
        self.log(f"REFUSE {label}: command interface disabled; set ALLOW_COMMANDS=True and CONFIRM_REAL_ROBOT={CONFIRM_TOKEN}.")
        return False

    def read_lowstate(self) -> Optional[LowState_]:
        state = self.lowstate_sub.Read(0.001)
        if state is not None:
            self.last_lowstate = state
        return self.last_lowstate

    def read_sport_state(self) -> Optional[Tuple[str, G1SportModeState_]]:
        for topic, subscriber in self.sport_subscribers:
            state = subscriber.Read(0.001)
            if state is not None:
                self.last_sport = (topic, state)
        return self.last_sport

    def service_status(self) -> Tuple[Optional[int], Dict[str, int]]:
        try:
            code, services = self.robot_state.ServiceList()
        except BaseException as exc:
            self.log(f"ServiceList exception: {exc}")
            return None, {}
        status = {}
        if code == 0 and services is not None:
            for service in services:
                if service.name in {"ai_sport", "sport_mode", "motion_switcher", "robot_state"}:
                    status[service.name] = int(service.status)
        return code, status

    def snapshot(self) -> Snapshot:
        snap = Snapshot(t=self.now(), service_status={})
        lowstate = self.read_lowstate()
        if lowstate is not None:
            snap.low_tick = int(lowstate.tick)
            snap.mode_pr = int(lowstate.mode_pr)
            snap.mode_machine = int(lowstate.mode_machine)
            snap.rpy = tuple(float(value) for value in lowstate.imu_state.rpy)
        try:
            snap.motion_code, snap.motion_data = self.motion_switcher.CheckMode()
            snap.motion_name = mode_name(snap.motion_data)
        except BaseException as exc:
            self.log(f"MotionSwitcher.CheckMode exception: {exc}")
        snap.service_code, snap.service_status = self.service_status()
        try:
            snap.loco_version_code, version = self.loco.GetServerApiVersion()
            snap.loco_version = decode_rpc_value(version)
        except BaseException as exc:
            self.log(f"Loco.GetServerApiVersion exception: {exc}")
        try:
            snap.fsm_code, fsm_value = self.loco.GetFsmId()
            snap.fsm_id = rpc_value_as_int(fsm_value)
            snap.fsm_mode_code, fsm_mode_value = self.loco.GetFsmMode()
            snap.fsm_mode = rpc_value_as_int(fsm_mode_value)
        except BaseException as exc:
            self.log(f"Loco.GetFsmId/GetFsmMode exception: {exc}")
        sport = self.read_sport_state()
        if sport is not None:
            topic, state = sport
            snap.sport_topic = topic
            snap.sport_fsm_id = int(state.fsm_id)
            snap.sport_fsm_mode = int(state.fsm_mode)
            snap.sport_task_id = int(state.task_id)
            snap.sport_task_time = float(state.task_time)
        self.last_snapshot = snap
        return snap

    def poll_motion_name(self, target: str, timeout_s: float, label: str) -> bool:
        deadline = time.monotonic() + timeout_s
        latest = None
        while time.monotonic() < deadline:
            try:
                code, result = self.motion_switcher.CheckMode()
            except BaseException as exc:
                latest = exc
                self.log(f"POLL {label}: exception={exc}")
                time.sleep(self.args.poll_s)
                continue
            latest = (code, result)
            name = mode_name(result)
            self.log(f"POLL {label}: MotionSwitcher code={code} name={name!r} data={result}")
            if code == 0 and name == target:
                return True
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT {label}: latest={latest}")
        return False

    def wait_loco_available(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        latest = None
        while time.monotonic() < deadline:
            try:
                code, version = self.loco.GetServerApiVersion()
            except BaseException as exc:
                latest = exc
                self.log(f"POLL loco available: exception={exc}")
                time.sleep(self.args.poll_s)
                continue
            latest = (code, decode_rpc_value(version))
            self.log(f"POLL loco available: code={code} version={decode_rpc_value(version)}")
            if code == 0:
                return True
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT loco available: latest={latest}")
        return False

    def poll_loco_fsm(self, targets: Iterable[int], timeout_s: float, label: str) -> Tuple[bool, Optional[int], Optional[int]]:
        target_values = {int(value) for value in targets}
        deadline = time.monotonic() + timeout_s
        latest = (None, None)
        while time.monotonic() < deadline:
            snap = self.snapshot()
            latest = (snap.fsm_id, snap.fsm_mode)
            self.log(
                f"POLL {label}: fsm_code={snap.fsm_code} fsm={snap.fsm_id}/{fsm_label(snap.fsm_id)} "
                f"mode_code={snap.fsm_mode_code} fsm_mode={snap.fsm_mode} "
                f"sport={snap.sport_fsm_id}/{snap.sport_fsm_mode}"
            )
            if snap.fsm_code == 0 and snap.fsm_id in target_values:
                return True, snap.fsm_id, snap.fsm_mode
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT {label}: latest_fsm={latest[0]} latest_mode={latest[1]}, targets={sorted(target_values)}")
        return False, latest[0], latest[1]

    def poll_stand_stable(self, timeout_s: float) -> Tuple[bool, Optional[int], Optional[int]]:
        deadline = time.monotonic() + timeout_s
        latest = (None, None)
        while time.monotonic() < deadline:
            snap = self.snapshot()
            latest = (snap.fsm_id, snap.fsm_mode)
            self.log(
                f"POLL StandUp stable: fsm={snap.fsm_id}/{fsm_label(snap.fsm_id)} "
                f"fsm_mode={snap.fsm_mode} sport={snap.sport_fsm_id}/{snap.sport_fsm_mode}"
            )
            if snap.fsm_code == 0 and snap.fsm_mode_code == 0:
                if snap.fsm_id in LOCOMOTION_FSM_IDS:
                    return True, snap.fsm_id, snap.fsm_mode
                if snap.fsm_id == 4 and snap.fsm_mode == 0:
                    return True, snap.fsm_id, snap.fsm_mode
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT StandUp stable: latest_fsm={latest[0]} latest_mode={latest[1]}")
        return False, latest[0], latest[1]

    def action_select_ai_service(self) -> None:
        if not self.require_commands("select ai service"):
            return
        deadline = time.monotonic() + self.args.switch_timeout_s
        while time.monotonic() < deadline:
            result = self.call("MotionSwitcher.SelectMode('ai')", lambda: self.motion_switcher.SelectMode("ai"))
            if isinstance(result, tuple) and result[0] == 0:
                break
            time.sleep(self.args.select_retry_s)
        if not self.poll_motion_name("ai", self.args.switch_timeout_s, "MotionSwitcher ai"):
            return
        code = self.call("RobotState.ServiceSwitch('ai_sport', True)", lambda: self.robot_state.ServiceSwitch("ai_sport", True))
        if code != 0:
            self.log(f"ServiceSwitch ai_sport failed with code={code}")
            return
        self.wait_loco_available(self.args.switch_timeout_s)
        self.poll_loco_fsm(range(0, 1000), self.args.switch_timeout_s, "first valid Loco FSM")

    def action_release(self) -> None:
        if not self.require_commands("ReleaseMode"):
            return
        result = self.call("MotionSwitcher.ReleaseMode()", self.motion_switcher.ReleaseMode)
        if isinstance(result, tuple) and result[0] != 0:
            self.log(f"ReleaseMode failed with code={result[0]}")
            return
        self.poll_motion_name("", self.args.switch_timeout_s, "released debug/lowcmd")

    def action_stop(self) -> None:
        if not self.require_commands("SetVelocity zero"):
            return
        self.call("Loco.SetVelocity(0,0,0,0.2)", lambda: self.loco.SetVelocity(0.0, 0.0, 0.0, 0.2))

    def action_damp(self) -> None:
        if not self.require_commands("Damp"):
            return
        self.action_stop()
        code = self.call("Loco.Damp()", self.loco.Damp)
        if code == 0:
            self.poll_loco_fsm([1], self.args.switch_timeout_s, "FSM 1 Damp")

    def action_zero(self) -> None:
        if not self.require_commands("ZeroTorque"):
            return
        self.action_stop()
        code = self.call("Loco.ZeroTorque()", self.loco.ZeroTorque)
        if code == 0:
            self.poll_loco_fsm([0], self.args.switch_timeout_s, "FSM 0 ZeroTorque")

    def action_stand(self) -> None:
        if not self.require_commands("StandUp"):
            return
        snap = self.snapshot()
        if snap.fsm_id not in {1, 4} and not self.args.force_unsafe:
            self.log("REFUSE StandUp: measured firmware only accepts this reliably after Damp/FSM1. Press n for full native recovery or set FORCE_UNSAFE=True.")
            return
        standup_start = time.monotonic()
        code = self.call("Loco.StandUp()", self.loco.StandUp)
        if code == 0:
            ok, _, _ = self.poll_loco_fsm([4] + list(LOCOMOTION_FSM_IDS), self.args.stand_timeout_s, "FSM 4 StandUp")
            if ok:
                self.poll_stand_stable(self.args.stand_timeout_s)
                elapsed = time.monotonic() - standup_start
                if elapsed < self.args.min_standup_s:
                    settle_s = self.args.min_standup_s - elapsed
                    self.log(f"Waiting {settle_s:.2f}s more for minimum StandUp settle time.")
                    time.sleep(settle_s)

    def action_locomotion(self) -> None:
        if not self.require_commands("Locomotion 802"):
            return
        snap = self.snapshot()
        if snap.fsm_id == 4 and snap.fsm_mode != 0:
            self.log("StandUp is still moving; waiting for FSM4/fsm_mode=0 before locomotion.")
            self.poll_stand_stable(self.args.stand_timeout_s)
        snap = self.snapshot()
        if snap.fsm_id not in {4, 802, 801, 500} and not self.args.force_unsafe:
            self.log("REFUSE locomotion: expected stable StandUp/FSM4 first. Press n for full native recovery or set FORCE_UNSAFE=True.")
            return
        for fsm_id in self.args.start_fsm_ids:
            code = self.call(f"Loco.SetFsmId({fsm_id})", lambda fsm_id=fsm_id: self.loco.SetFsmId(fsm_id))
            if code != 0:
                continue
            ok, observed, _ = self.poll_loco_fsm(
                LOCOMOTION_FSM_IDS,
                min(self.args.switch_timeout_s, self.args.locomotion_candidate_timeout_s),
                "locomotion FSM",
            )
            if ok:
                self.action_stop()
                self.log(f"Locomotion confirmed as FSM {observed}.")
                return
        self.log(f"Locomotion was not confirmed via candidates {self.args.start_fsm_ids}.")

    def action_native(self) -> None:
        if not self.require_commands("native recovery"):
            return
        self.action_select_ai_service()
        self.action_damp()
        self.action_stand()
        self.action_locomotion()

    def action_balance(self) -> None:
        if not self.require_commands("BalanceStand"):
            return
        code = self.call("Loco.BalanceStand()", self.loco.BalanceStand)
        if code != 0:
            self.log(f"BalanceStand failed with code={code}")

    def action_snapshot(self) -> None:
        snap = self.snapshot()
        self.log(self.format_snapshot(snap).replace("\n", " | "))

    def handle_action(self, action: str) -> bool:
        handlers = {
            "snapshot": self.action_snapshot,
            "select": self.action_select_ai_service,
            "native": self.action_native,
            "release": self.action_release,
            "stop": self.action_stop,
            "damp": self.action_damp,
            "zero": self.action_zero,
            "stand": self.action_stand,
            "locomotion": self.action_locomotion,
            "balance": self.action_balance,
        }
        if action == "quit":
            return False
        handler = handlers.get(action)
        if handler is None:
            self.log(f"Unknown action: {action}")
            return True
        start = time.monotonic()
        self.log(f"BEGIN action={action}")
        try:
            handler()
        except BaseException as exc:
            self.log(f"FAILED action={action}: {type(exc).__name__}: {exc}")
        self.log(f"END action={action} elapsed={time.monotonic() - start:.3f}s")
        return True

    def handle_key(self, key: str) -> bool:
        mapping = {
            "p": "snapshot",
            "a": "select",
            "n": "native",
            "x": "release",
            "v": "stop",
            "d": "damp",
            "z": "zero",
            "u": "stand",
            "l": "locomotion",
            "b": "balance",
            "q": "quit",
        }
        action = mapping.get(key.lower())
        if action is None:
            self.log(f"Unmapped key: {key!r}. Press h for help.")
            return True
        return self.handle_action(action)

    def possible_actions(self, snap: Snapshot) -> List[str]:
        actions = ["p snapshot", "v zero-velocity"]
        if snap.motion_name != "ai" or snap.loco_version_code != 0:
            actions.extend(["a select-ai", "n full-native-recover"])
            return actions
        if snap.fsm_id in LOCOMOTION_FSM_IDS:
            actions.extend(["b balance", "d damp", "z zero", "x release"])
        elif snap.fsm_id == 0:
            actions.extend(["d damp", "n full-native-recover"])
        elif snap.fsm_id == 1:
            actions.extend(["u stand", "z zero", "n full-native-recover"])
        elif snap.fsm_id == 4:
            if snap.fsm_mode == 0:
                actions.extend(["l locomotion", "d damp"])
            else:
                actions.extend(["wait stable-stand", "d damp"])
        else:
            actions.extend(["d damp", "n full-native-recover"])
        return actions

    def format_snapshot(self, snap: Snapshot) -> str:
        service_text = ", ".join(f"{name}={status}" for name, status in sorted((snap.service_status or {}).items())) or "none"
        rpy_text = "none"
        if snap.rpy is not None:
            rpy_text = "[" + ", ".join(f"{value:.4f}" for value in snap.rpy) + "]"
        return "\n".join(
            [
                f"t={snap.t:.3f}s",
                f"MotionSwitcher: code={snap.motion_code} name={snap.motion_name!r} data={snap.motion_data}",
                f"Services: code={snap.service_code} {service_text}  (observed: status 0=running, 1=stopped on this robot)",
                f"Loco: version_code={snap.loco_version_code} version={snap.loco_version} "
                f"fsm={snap.fsm_id}/{fsm_label(snap.fsm_id)} code={snap.fsm_code} "
                f"fsm_mode={snap.fsm_mode} mode_code={snap.fsm_mode_code}",
                f"SportModeState: topic={snap.sport_topic or 'none'} fsm={snap.sport_fsm_id}/{fsm_label(snap.sport_fsm_id)} "
                f"fsm_mode={snap.sport_fsm_mode} task={snap.sport_task_id} task_time={snap.sport_task_time}",
                f"LowState: tick={snap.low_tick} mode_pr={snap.mode_pr} mode_machine={snap.mode_machine} rpy={rpy_text}",
                "Possible now: " + ", ".join(self.possible_actions(snap)),
            ]
        )

    def render(self, snap: Snapshot) -> None:
        print("\033[H\033[J", end="")
        print("=====================================")
        print("  G1 Mode Console")
        print("=====================================")
        print(f"net={self.args.net} allow_commands={self.args.allow_commands} force_unsafe={self.args.force_unsafe}")
        print(self.format_snapshot(snap))
        print("-------------------------------------")
        print("Keys: p snapshot | a select-ai | n full-native | v stop | d damp | u stand | l locomotion | b balance | z zero | x release | q quit")
        print("Estimated times: Damp ~0.1s; StandUp stable ~5-6s; clean FSM4 -> 802 can be ~0.1-0.3s, but AMP-exit FSM4 may refuse locomotion; Release -> SelectMode usable can be ~8-13s.")
        print("-------------------------------------")
        for line in self.logs[-self.args.log_lines :]:
            print(line)
        sys.stdout.flush()

    def run(self) -> None:
        if self.args.action != "none":
            self.action_snapshot()
            self.handle_action(self.args.action)
            self.action_snapshot()
            return
        end_time = None if self.args.duration_s <= 0.0 else time.monotonic() + self.args.duration_s
        with RawTerminal(enabled=not self.args.no_tui):
            while True:
                snap = self.snapshot()
                self.render(snap)
                if end_time is not None and time.monotonic() >= end_time:
                    break
                deadline = time.monotonic() + self.args.refresh_s
                while time.monotonic() < deadline:
                    key = RawTerminal(enabled=False).read_key()
                    if not key and sys.stdin.isatty() and not self.args.no_tui:
                        readable, _, _ = select.select([sys.stdin], [], [], 0.02)
                        if readable:
                            key = sys.stdin.read(1)
                    if key:
                        if key.lower() == "h":
                            self.log("Keys: p/a/n/v/d/u/l/b/z/x/q. Dangerous modes require launch confirmation.")
                        elif not self.handle_key(key):
                            return
                        break
                    time.sleep(0.02)


def parse_start_fsm_ids(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or [801, 802, 500]


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously monitor and optionally switch Unitree G1 motion modes.")
    parser.add_argument("net", help="Unitree DDS network interface, e.g. enp11s0.")
    parser.add_argument("--allow-commands", action="store_true")
    parser.add_argument("--confirm-real-robot", default="")
    parser.add_argument("--force-unsafe", action="store_true", help="Allow direct transitions that are not validated as safe.")
    parser.add_argument("--loco-service", default="sport")
    parser.add_argument("--sport-topic", action="append", default=[])
    parser.add_argument("--start-fsm-ids", type=parse_start_fsm_ids, default=parse_start_fsm_ids("801,802,500"))
    parser.add_argument("--rpc-timeout-s", type=float, default=0.3)
    parser.add_argument("--poll-s", type=float, default=0.1)
    parser.add_argument("--refresh-s", type=float, default=0.5)
    parser.add_argument("--switch-timeout-s", type=float, default=20.0)
    parser.add_argument("--stand-timeout-s", type=float, default=12.0)
    parser.add_argument("--min-standup-s", type=float, default=6.0)
    parser.add_argument("--locomotion-candidate-timeout-s", type=float, default=4.0)
    parser.add_argument("--select-retry-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--log-lines", type=int, default=12)
    parser.add_argument("--action", choices=[
        "none",
        "snapshot",
        "select",
        "native",
        "release",
        "stop",
        "damp",
        "zero",
        "stand",
        "locomotion",
        "balance",
        "quit",
    ], default="none")
    parser.add_argument("--no-tui", action="store_true")
    args = parser.parse_args()

    if args.allow_commands and args.confirm_real_robot != CONFIRM_TOKEN:
        raise SystemExit(f"Refusing command interface without --confirm-real-robot {CONFIRM_TOKEN}.")
    if args.action not in {"none", "snapshot"} and not args.allow_commands:
        raise SystemExit("Refusing action command when --allow-commands is not set.")

    ChannelFactoryInitialize(0, args.net)
    console = ModeConsole(args)
    try:
        console.run()
    finally:
        print("\033[0m", end="", flush=True)


if __name__ == "__main__":
    main()
