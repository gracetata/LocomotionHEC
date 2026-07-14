"""Check Unitree G1 high-level locomotion state before low-level policy handoff.

Core functions:
    print_motion_switcher_state reads MotionSwitcher mode ownership, and
    print_loco_state reads G1 loco FSM, balance, swing-height, stand-height,
    and phase getters. run_stand_action can optionally request a high-level
    stand/damp/zero action after explicit operator confirmation.

Inputs/outputs:
    Input is a Unitree DDS network interface. Outputs are human-readable status
    lines from MotionSwitcherClient and LocoClient RPC calls. Optional stand
    actions return a process error when Unitree RPC reports a non-zero code.

Usage:
    python check_g1_loco_state.py eth0
    python check_g1_loco_state.py eth0 --stand-action high --confirm-stand I_UNDERSTAND
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Tuple

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


CONFIRM_TOKEN = "I_UNDERSTAND"


def print_motion_switcher_state() -> None:
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()
    code, result = motion_switcher.CheckMode()
    print("=====================================")
    print("  Unitree MotionSwitcher")
    print("=====================================")
    print(f"CheckMode code : {code}")
    print(f"CheckMode data : {result}")


def print_loco_value(name: str, getter: Callable[[], Tuple[int, object]]) -> None:
    try:
        code, value = getter()
    except Exception as exc:
        print(f"{name:<14}: exception={exc}")
        return
    print(f"{name:<14}: code={code} value={value}")


def print_loco_state(loco_client: LocoClient) -> None:
    print("=====================================")
    print("  Unitree G1 Loco State")
    print("=====================================")
    print_loco_value("fsm_id", loco_client.GetFsmId)
    print_loco_value("fsm_mode", loco_client.GetFsmMode)
    print_loco_value("balance_mode", loco_client.GetBalanceMode)
    print_loco_value("swing_height", loco_client.GetSwingHeight)
    print_loco_value("stand_height", loco_client.GetStandHeight)
    print_loco_value("phase", loco_client.GetPhase)


def run_stand_action(loco_client: LocoClient, action: str, confirm_stand: str) -> None:
    if action == "none":
        return
    if confirm_stand != CONFIRM_TOKEN:
        raise SystemExit(f"Refusing STAND_ACTION={action} without --confirm-stand {CONFIRM_TOKEN}.")

    if action == "high":
        code = loco_client.SetStandHeight((1 << 32) - 1)
    elif action == "low":
        code = loco_client.SetStandHeight(0)
    elif action == "squat2stand":
        code = loco_client.SetFsmId(706)
    elif action == "damp":
        code = loco_client.SetFsmId(1)
    elif action == "zero":
        code = loco_client.SetFsmId(0)
    elif action == "stop":
        code = loco_client.SetVelocity(0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unsupported stand action: {action}")

    print("=====================================")
    print("  Requested Stand Action")
    print("=====================================")
    print(f"Action : {action}")
    print(f"Code   : {code}")
    if code != 0:
        raise SystemExit(f"Unitree stand action failed with code {code}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Unitree G1 loco state and optionally request a stand action.")
    parser.add_argument("net", type=str, help="Network interface for Unitree DDS, e.g. eth0.")
    parser.add_argument(
        "--stand-action",
        choices=["none", "high", "low", "squat2stand", "damp", "zero", "stop"],
        default="none",
        help="Optional high-level action. Defaults to read-only.",
    )
    parser.add_argument("--confirm-stand", default="", help="Must be I_UNDERSTAND for non-read-only actions.")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)
    print_motion_switcher_state()

    loco_client = LocoClient()
    loco_client.SetTimeout(5.0)
    loco_client.Init()
    print_loco_state(loco_client)
    run_stand_action(loco_client, args.stand_action, args.confirm_stand)
    if args.stand_action != "none":
        time.sleep(1.0)
        print_loco_state(loco_client)


if __name__ == "__main__":
    main()