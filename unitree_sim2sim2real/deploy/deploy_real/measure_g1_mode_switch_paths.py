"""Measure Unitree G1 high-level/debug mode switch paths with timestamps.

This script is for hoist-protected real-robot experiments. It only sends zero
velocity and mode-switch RPCs, then records when MotionSwitcher, Loco RPC, and
SportModeState readbacks actually observe the requested state.
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

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
LOCOMOTION_FSM_IDS = (500, 801, 802)


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
        try:
            return int(decoded.strip())
        except ValueError:
            return None
    return None


def mode_name(result: object) -> str:
    if isinstance(result, dict):
        return str(result.get("name", "") or "")
    return ""


class Experiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.t0 = time.monotonic()
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

    def now(self) -> float:
        return time.monotonic() - self.t0

    def log(self, message: str) -> None:
        print(f"[{self.now():8.3f}s] {message}", flush=True)

    def call(self, label: str, func: Callable[[], object]) -> object:
        start = self.now()
        self.log(f"CALL {label}")
        result = func()
        self.log(f"RET  {label} after {self.now() - start:.3f}s -> {result}")
        return result

    def read_lowstate(self) -> Optional[LowState_]:
        return self.lowstate_sub.Read(0.02)

    def read_sport_state(self) -> Optional[Tuple[str, G1SportModeState_]]:
        for topic, subscriber in self.sport_subscribers:
            state = subscriber.Read(0.01)
            if state is not None:
                return topic, state
        return None

    def check_motion(self) -> Tuple[int, object, str]:
        code, result = self.motion_switcher.CheckMode()
        return code, result, mode_name(result)

    def check_loco_fsm(self) -> Tuple[int, Optional[int], int, Optional[int]]:
        fsm_code, fsm_value = self.loco.GetFsmId()
        mode_code, mode_value = self.loco.GetFsmMode()
        return fsm_code, rpc_value_as_int(fsm_value), mode_code, rpc_value_as_int(mode_value)

    def service_status(self) -> dict:
        code, services = self.robot_state.ServiceList()
        status = {"__code__": code}
        if code == 0 and services is not None:
            for service in services:
                if service.name in {"ai_sport", "sport_mode", "motion_switcher", "robot_state"}:
                    status[service.name] = int(service.status)
        return status

    def snapshot(self, label: str) -> None:
        self.log(f"SNAPSHOT {label}")
        lowstate = self.read_lowstate()
        if lowstate is not None:
            rpy = ", ".join(f"{float(value):.4f}" for value in lowstate.imu_state.rpy)
            self.log(
                "  lowstate "
                f"tick={int(lowstate.tick)} mode_pr={int(lowstate.mode_pr)} "
                f"mode_machine={int(lowstate.mode_machine)} rpy=[{rpy}]"
            )
        else:
            self.log("  lowstate no fresh sample")
        code, result, name = self.check_motion()
        self.log(f"  MotionSwitcher code={code} name={name!r} data={result}")
        self.log(f"  ServiceList {self.service_status()}")
        fsm_code, fsm_id, mode_code, fsm_mode = self.check_loco_fsm()
        self.log(
            f"  Loco GetFsmId code={fsm_code} value={fsm_id}; "
            f"GetFsmMode code={mode_code} value={fsm_mode}"
        )
        sport = self.read_sport_state()
        if sport is None:
            self.log("  SportModeState no fresh sample")
        else:
            topic, state = sport
            self.log(
                f"  SportModeState {topic} fsm_id={int(state.fsm_id)} "
                f"fsm_mode={int(state.fsm_mode)} task_id={int(state.task_id)} "
                f"task_time={float(state.task_time):.3f}"
            )

    def poll_motion_name(self, target_name: str, timeout_s: float, label: str) -> bool:
        deadline = time.monotonic() + timeout_s
        latest = None
        while time.monotonic() < deadline:
            code, result, name = self.check_motion()
            latest = (code, result, name)
            self.log(f"POLL {label}: MotionSwitcher code={code} name={name!r} data={result}")
            if code == 0 and name == target_name:
                return True
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT {label}: latest={latest}")
        return False

    def poll_loco_fsm(self, targets: Iterable[int], timeout_s: float, label: str) -> Tuple[bool, Optional[int]]:
        target_values = {int(value) for value in targets}
        deadline = time.monotonic() + timeout_s
        latest_fsm = None
        while time.monotonic() < deadline:
            fsm_code, fsm_id, mode_code, fsm_mode = self.check_loco_fsm()
            latest_fsm = fsm_id
            sport = self.read_sport_state()
            sport_text = ""
            if sport is not None:
                topic, state = sport
                sport_text = f"; sport={topic}:{int(state.fsm_id)}/{int(state.fsm_mode)}"
            self.log(
                f"POLL {label}: GetFsmId code={fsm_code} value={fsm_id}; "
                f"GetFsmMode code={mode_code} value={fsm_mode}{sport_text}"
            )
            if fsm_code == 0 and fsm_id in target_values:
                return True, fsm_id
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT {label}: latest_fsm={latest_fsm}, targets={sorted(target_values)}")
        return False, latest_fsm

    def poll_standup_stable(self, timeout_s: float) -> Tuple[bool, Optional[int], Optional[int]]:
        deadline = time.monotonic() + timeout_s
        latest = (None, None)
        while time.monotonic() < deadline:
            fsm_code, fsm_id, mode_code, fsm_mode = self.check_loco_fsm()
            latest = (fsm_id, fsm_mode)
            sport = self.read_sport_state()
            sport_text = ""
            if sport is not None:
                topic, state = sport
                sport_text = f"; sport={topic}:{int(state.fsm_id)}/{int(state.fsm_mode)}"
            self.log(
                "POLL StandUp stable: "
                f"GetFsmId code={fsm_code} value={fsm_id}; "
                f"GetFsmMode code={mode_code} value={fsm_mode}{sport_text}"
            )
            if fsm_code == 0 and mode_code == 0:
                if fsm_id in LOCOMOTION_FSM_IDS:
                    return True, fsm_id, fsm_mode
                if fsm_id == 4 and fsm_mode == 0:
                    return True, fsm_id, fsm_mode
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT StandUp stable: latest_fsm={latest[0]} latest_mode={latest[1]}")
        return False, latest[0], latest[1]

    def wait_loco_available(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        latest = None
        while time.monotonic() < deadline:
            code, version = self.loco.GetServerApiVersion()
            latest = (code, decode_rpc_value(version))
            self.log(f"POLL loco available: GetServerApiVersion code={code} value={decode_rpc_value(version)}")
            if code == 0:
                return True
            time.sleep(self.args.poll_s)
        self.log(f"TIMEOUT loco available: latest={latest}")
        return False

    def release_mode(self) -> None:
        self.snapshot("before ReleaseMode")
        code, result, name = self.check_motion()
        if code == 0 and not name:
            self.log("MotionSwitcher already released (name is empty).")
            return
        status = self.call("MotionSwitcher.ReleaseMode()", self.motion_switcher.ReleaseMode)
        if isinstance(status, tuple) and status[0] != 0:
            raise RuntimeError(f"ReleaseMode failed: {status}")
        self.poll_motion_name("", self.args.mode_timeout_s, "ReleaseMode name empty")
        self.snapshot("after ReleaseMode")

    def select_ai_service(self) -> None:
        deadline = time.monotonic() + self.args.mode_timeout_s
        status = None
        while time.monotonic() < deadline:
            status = self.call(
                f"MotionSwitcher.SelectMode({self.args.target_mode!r})",
                lambda: self.motion_switcher.SelectMode(self.args.target_mode),
            )
            if isinstance(status, tuple) and status[0] == 0:
                break
            self.log(f"SelectMode retry after code={status}; MotionSwitcher may still be releasing.")
            time.sleep(self.args.select_retry_s)
        if not isinstance(status, tuple) or status[0] != 0:
            raise RuntimeError(f"SelectMode failed: {status}")
        if not self.poll_motion_name(self.args.target_mode, self.args.mode_timeout_s, "SelectMode target"):
            raise RuntimeError(f"MotionSwitcher did not report {self.args.target_mode!r}.")
        if self.args.start_service:
            code = self.call(
                f"RobotState.ServiceSwitch({self.args.start_service!r}, True)",
                lambda: self.robot_state.ServiceSwitch(self.args.start_service, True),
            )
            if code != 0:
                raise RuntimeError(f"ServiceSwitch failed with code {code}.")
        if not self.wait_loco_available(self.args.mode_timeout_s):
            raise RuntimeError("Loco RPC did not become available.")
        ok, fsm_id = self.poll_loco_fsm(range(0, 1000), self.args.mode_timeout_s, "first valid Loco FSM")
        if not ok:
            raise RuntimeError(f"Loco FSM did not become valid; latest={fsm_id}.")
        self.snapshot("after SelectMode/ServiceSwitch")

    def request_zero(self) -> None:
        self.call("Loco.SetVelocity(0,0,0,0.2)", lambda: self.loco.SetVelocity(0.0, 0.0, 0.0, 0.2))
        code = self.call("Loco.ZeroTorque()", self.loco.ZeroTorque)
        if code != 0:
            raise RuntimeError(f"ZeroTorque failed with code {code}.")
        ok, fsm_id = self.poll_loco_fsm([0], self.args.mode_timeout_s, "FSM 0 ZeroTorque")
        if not ok:
            raise RuntimeError(f"FSM 0 was not confirmed; latest={fsm_id}.")

    def request_damp(self) -> None:
        self.call("Loco.SetVelocity(0,0,0,0.2)", lambda: self.loco.SetVelocity(0.0, 0.0, 0.0, 0.2))
        code = self.call("Loco.Damp()", self.loco.Damp)
        if code != 0:
            raise RuntimeError(f"Damp failed with code {code}.")
        ok, fsm_id = self.poll_loco_fsm([1], self.args.mode_timeout_s, "FSM 1 Damp")
        if not ok:
            raise RuntimeError(f"FSM 1 was not confirmed; latest={fsm_id}.")

    def request_standup(self) -> None:
        code = self.call("Loco.StandUp()", self.loco.StandUp)
        if code != 0:
            raise RuntimeError(f"StandUp failed with code {code}.")
        ok, fsm_id = self.poll_loco_fsm([4] + list(LOCOMOTION_FSM_IDS), self.args.stand_timeout_s, "FSM 4 StandUp")
        if not ok:
            raise RuntimeError(f"StandUp FSM was not confirmed; latest={fsm_id}.")
        stable, stable_fsm, stable_mode = self.poll_standup_stable(self.args.stand_timeout_s)
        if not stable:
            raise RuntimeError(
                "StandUp did not reach a stable standing readback; "
                f"latest_fsm={stable_fsm}, latest_mode={stable_mode}."
            )

    def request_locomotion(self) -> None:
        for fsm_id in self.args.start_fsm_ids:
            code = self.call(f"Loco.SetFsmId({fsm_id})", lambda fsm_id=fsm_id: self.loco.SetFsmId(fsm_id))
            if code != 0:
                continue
            ok, observed = self.poll_loco_fsm(LOCOMOTION_FSM_IDS, self.args.mode_timeout_s, "locomotion FSM")
            if ok:
                self.call("Loco.SetVelocity(0,0,0,0.2)", lambda: self.loco.SetVelocity(0.0, 0.0, 0.0, 0.2))
                self.log(f"LOCOMOTION confirmed as FSM {observed}.")
                return
        raise RuntimeError(f"Locomotion was not confirmed via candidates {self.args.start_fsm_ids}.")

    def recover(self) -> None:
        if self.args.recover == "none":
            self.log("Recovery skipped by --recover none.")
            return
        self.log(f"BEGIN recovery mode={self.args.recover}")
        try:
            code, result, name = self.check_motion()
            if code != 0 or name != self.args.target_mode:
                self.select_ai_service()
            else:
                self.wait_loco_available(self.args.mode_timeout_s)
        except BaseException as exc:
            self.log(f"Recovery native service select failed: {exc}")
        if self.args.recover == "service":
            return
        try:
            if self.args.recover in {"damp", "stand", "locomotion"}:
                self.request_damp()
            if self.args.recover in {"stand", "locomotion"}:
                self.request_standup()
            if self.args.recover == "locomotion":
                self.request_locomotion()
        except BaseException as exc:
            self.log(f"Recovery mode action failed: {exc}")
        self.snapshot("after recovery")

    def run_path(self) -> None:
        self.snapshot("initial")
        if self.args.path == "snapshot":
            return
        if self.args.path == "baseline-damp-stand-loco":
            self.request_damp()
            self.request_standup()
            self.request_locomotion()
            return
        if self.args.path == "zero-direct-stand-loco":
            self.request_zero()
            self.request_standup()
            self.request_locomotion()
            return
        if self.args.path == "zero-direct-loco":
            self.request_zero()
            self.request_locomotion()
            return
        if self.args.path == "release-select-service":
            self.release_mode()
            self.select_ai_service()
            return
        if self.args.path == "release-direct-stand-loco":
            self.release_mode()
            self.select_ai_service()
            self.request_standup()
            self.request_locomotion()
            return
        if self.args.path == "release-direct-loco":
            self.release_mode()
            self.select_ai_service()
            self.request_locomotion()
            return
        raise ValueError(f"Unsupported path: {self.args.path}")


def parse_start_fsm_ids(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or [801, 802, 500]


def main() -> None:
    parser = argparse.ArgumentParser(description="Timestamp G1 mode switch paths on a hoist-protected robot.")
    parser.add_argument("net", help="Unitree DDS network interface, e.g. enp11s0.")
    parser.add_argument(
        "--path",
        choices=[
            "snapshot",
            "baseline-damp-stand-loco",
            "zero-direct-stand-loco",
            "zero-direct-loco",
            "release-select-service",
            "release-direct-stand-loco",
            "release-direct-loco",
        ],
        default="snapshot",
    )
    parser.add_argument("--confirm-real-robot", default="", help=f"Must be {CONFIRM_TOKEN} for non-snapshot paths.")
    parser.add_argument("--target-mode", default="ai")
    parser.add_argument("--start-service", default="ai_sport")
    parser.add_argument("--loco-service", default="sport")
    parser.add_argument("--start-fsm-ids", type=parse_start_fsm_ids, default=parse_start_fsm_ids("801,802,500"))
    parser.add_argument("--sport-topic", action="append", default=[])
    parser.add_argument("--rpc-timeout-s", type=float, default=1.0)
    parser.add_argument("--poll-s", type=float, default=0.1)
    parser.add_argument("--select-retry-s", type=float, default=1.0)
    parser.add_argument("--mode-timeout-s", type=float, default=6.0)
    parser.add_argument("--stand-timeout-s", type=float, default=12.0)
    parser.add_argument("--recover", choices=["none", "service", "damp", "stand", "locomotion"], default="locomotion")
    args = parser.parse_args()

    if args.path != "snapshot" and args.confirm_real_robot != CONFIRM_TOKEN:
        raise SystemExit(f"Refusing --path {args.path} without --confirm-real-robot {CONFIRM_TOKEN}.")

    ChannelFactoryInitialize(0, args.net)
    print("=====================================")
    print("  G1 Mode Switch Path Measurement")
    print("=====================================")
    print(f"net={args.net}")
    print(f"path={args.path}")
    print(f"target_mode={args.target_mode}")
    print(f"start_service={args.start_service or '<disabled>'}")
    print(f"loco_service={args.loco_service}")
    print(f"start_fsm_ids={args.start_fsm_ids}")
    print(f"recover={args.recover}")
    print("=====================================", flush=True)

    experiment = Experiment(args)
    try:
        experiment.run_path()
        experiment.snapshot("final")
        if args.path == "release-select-service" and args.recover != "none":
            experiment.log("release-select-service ends in service/FSM0 by design; running requested recovery.")
            experiment.recover()
        print("RESULT: SUCCESS", flush=True)
    except BaseException as exc:
        experiment.log(f"RESULT: FAILED with {type(exc).__name__}: {exc}")
        experiment.recover()
        raise


if __name__ == "__main__":
    main()
