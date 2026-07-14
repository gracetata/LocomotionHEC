"""Unitree G1 high-level locomotion RPC client helpers.

Core class:
    LocoClient wraps Unitree G1 sport/loco RPC APIs for high-level state reads
    and motion commands such as stand-height, damping, velocity, and arm tasks.

Inputs/outputs:
    Inputs are JSON RPC payloads sent through unitree_sdk2py Client._Call.
    Methods return Unitree RPC status codes and decoded data values when the
    firmware exposes the corresponding getter.

Usage:
    client = LocoClient(); client.SetTimeout(5.0); client.Init()
    client = LocoClient(service_name="sport")
    code, fsm_id = client.GetFsmId()
"""

import json

from ...rpc.client import Client
from .g1_loco_api import *

"""
" class SportClient
"""
class LocoClient(Client):
    def __init__(self, service_name: str = LOCO_SERVICE_NAME):
        super().__init__(service_name, False)
        self.service_name = service_name
        self.first_shake_hand_stage_ = -1
        self.continous_move_ = False

    def Init(self):
        # set api version
        self._SetApiVerson(LOCO_API_VERSION)

        # regist api
        self._RegistApi(ROBOT_API_ID_LOCO_GET_FSM_ID, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_GET_FSM_MODE, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_GET_BALANCE_MODE, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_GET_SWING_HEIGHT, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_GET_STAND_HEIGHT, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_GET_PHASE, 0) # deprecated

        self._RegistApi(ROBOT_API_ID_LOCO_SET_FSM_ID, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_SET_BALANCE_MODE, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_SET_SWING_HEIGHT, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_SET_STAND_HEIGHT, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_SET_VELOCITY, 0)
        self._RegistApi(ROBOT_API_ID_LOCO_SET_ARM_TASK, 0)

    # 7101
    def SetFsmId(self, fsm_id: int):
        p = {}
        p["data"] = fsm_id
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_LOCO_SET_FSM_ID, parameter)
        return code

    # 7102
    def SetBalanceMode(self, balance_mode: int):
        p = {}
        p["data"] = balance_mode
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_LOCO_SET_BALANCE_MODE, parameter)
        return code

    # 7104
    def SetStandHeight(self, stand_height: float):
        p = {}
        p["data"] = stand_height
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_LOCO_SET_STAND_HEIGHT, parameter)
        return code

    # 7105
    def SetVelocity(self, vx: float, vy: float, omega: float, duration: float = 1.0):
        p = {}
        velocity = [vx,vy,omega]
        p["velocity"] = velocity
        p["duration"] = duration
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_LOCO_SET_VELOCITY, parameter)
        return code
    
    # 7106
    def SetTaskId(self, task_id: float):
        p = {}
        p["data"] = task_id
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_LOCO_SET_ARM_TASK, parameter)
        return code

    def _GetData(self, api_id: int):
        payload = {}
        parameter = json.dumps(payload)
        code, data = self._Call(api_id, parameter)
        if code != 0:
            return code, None
        decoded = json.loads(data)
        return code, decoded.get("data")

    def GetFsmId(self):
        return self._GetData(ROBOT_API_ID_LOCO_GET_FSM_ID)

    def GetFsmMode(self):
        return self._GetData(ROBOT_API_ID_LOCO_GET_FSM_MODE)

    def GetBalanceMode(self):
        return self._GetData(ROBOT_API_ID_LOCO_GET_BALANCE_MODE)

    def GetSwingHeight(self):
        return self._GetData(ROBOT_API_ID_LOCO_GET_SWING_HEIGHT)

    def GetStandHeight(self):
        return self._GetData(ROBOT_API_ID_LOCO_GET_STAND_HEIGHT)

    def GetPhase(self):
        return self._GetData(ROBOT_API_ID_LOCO_GET_PHASE)

    def Damp(self):
        return self.SetFsmId(1)
    
    def Start(self):
        return self.SetFsmId(500)

    def Squat(self):
        return self.SetFsmId(2)

    def Squat2StandUp(self):
        return self.SetFsmId(706)

    def Lie2StandUp(self):
        return self.SetFsmId(702)

    def Sit(self):
        return self.SetFsmId(3)

    def StandUp(self):
        return self.SetFsmId(4)

    def StandUp2Squat(self):
        return self.SetFsmId(706)

    def ZeroTorque(self):
        return self.SetFsmId(0)

    def StopMove(self):
        return self.SetVelocity(0., 0., 0.)

    def HighStand(self):
        UINT32_MAX = (1 << 32) - 1
        return self.SetStandHeight(UINT32_MAX)

    def LowStand(self):
        UINT32_MIN = 0
        return self.SetStandHeight(UINT32_MIN)

    def Move(self, vx: float, vy: float, vyaw: float, continous_move: bool = None):
        if continous_move is None:
            continous_move = self.continous_move_
        duration = 864000.0 if continous_move else 1
        return self.SetVelocity(vx, vy, vyaw, duration)

    def BalanceStand(self):
        return self.SetBalanceMode(0)

    def ContinuousGait(self, flag: bool):
        return self.SetBalanceMode(1 if flag else 0)

    def SwitchMoveMode(self, flag: bool):
        self.continous_move_ = flag
        return 0

    def WaveHand(self, turn_flag: bool = False):
        return self.SetTaskId(1 if turn_flag else 0)

    def ShakeHand(self, stage: int = -1):
        if stage == 0:
            self.first_shake_hand_stage_ = False
            self.SetTaskId(2)
        elif stage == 1:
            self.first_shake_hand_stage_ = True
            self.SetTaskId(3)
        else:
            self.first_shake_hand_stage_ = not self.first_shake_hand_stage_
            return self.SetTaskId(3 if self.first_shake_hand_stage_ else 2)
    