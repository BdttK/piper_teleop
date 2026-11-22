"""piper 机械臂封装(基于 vendored piper_sdk)与 dry-run 假臂。"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# piper 关节硬限位(deg),与 SDK JointCtrl 文档一致
JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)


@dataclass
class ArmState:
    joints_deg: np.ndarray  # (6,)
    gripper_mm: float
    hz: float


class PiperArm:
    """单个 piper 臂(一条独立 CAN 总线)。"""

    is_fake = False

    def __init__(self, can_name: str):
        from piper_sdk import C_PiperInterface_V2

        self.can_name = can_name
        # start_sdk_*_limit: 让 SDK 在发送前按关节/夹爪限位截断指令
        self.piper = C_PiperInterface_V2(
            can_name,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True,
        )
        self._connected = False

    # ---- 生命周期 ----
    def connect(self) -> None:
        self.piper.ConnectPort()
        self._connected = True

    def close(self) -> None:
        if not self._connected:
            return
        try:
            self.piper.DisconnectPort()
        except Exception:
            logger.debug("[%s] DisconnectPort 异常", self.can_name, exc_info=True)
        self._connected = False

    # ---- 读 ----
    def read(self) -> ArmState:
        joint_msgs = self.piper.GetArmJointMsgs()
        joints = (
            np.array(
                [
                    joint_msgs.joint_1,
                    joint_msgs.joint_2,
                    joint_msgs.joint_3,
                    joint_msgs.joint_4,
                    joint_msgs.joint_5,
                    joint_msgs.joint_6,
                ],
                dtype=np.float32,
            )
            / 1000.0
        )
        gripper_msgs = self.piper.GetArmGripperMsgs()
        return ArmState(
            joints_deg=joints,
            gripper_mm=gripper_msgs.grippers_angle / 1000.0,
            hz=float(joint_msgs.Hz),
        )

    def wait_feedback(self, timeout_s: float = 3.0, min_hz: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.read().hz > min_hz:
                return True
            time.sleep(0.05)
        return False

    # ---- 使能/失能 ----
    def enable(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.piper.EnablePiper()
            status = self.piper.GetArmEnableStatus()
            if status and all(status):
                return True
            time.sleep(0.05)
        return False

    def disable(self, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.piper.DisablePiper()
            status = self.piper.GetArmEnableStatus()
            if status and not any(status):
                return True
            time.sleep(0.05)
        return False

    def is_enabled(self) -> bool:
        status = self.piper.GetArmEnableStatus()
        return bool(status) and all(status)

    # ---- 写 ----
    def send_command(
        self,
        joints_deg: np.ndarray,
        gripper_mm: float,
        speed_rate: int = 100,
        installation_pos: int = 0,
        gripper_effort_nm: float = 1.0,
    ) -> None:
        joints = np.asarray(joints_deg, dtype=float)
        self.piper.MotionCtrl_2(0x01, 0x01, int(speed_rate), 0x00, 0, installation_pos)
        self.piper.JointCtrl(*[int(round(v * 1000.0)) for v in joints])
        self.piper.GripperCtrl(
            int(round(float(gripper_mm) * 1000.0)),
            int(round(float(gripper_effort_nm) * 1000.0)),
            0x01,
            0x00,
        )

    def stop_hold(self) -> None:
        """急停并保持当前位置(退出前再调用 disable)。"""
        self.piper.MotionCtrl_1(0x01, 0x00, 0x00)

    def recover(self) -> None:
        """清除急停/错误 —— 注意:固件复位会让机械臂立刻失电下坠。

        不要在正常启动路径里调用;仅在人已扶住机械臂或它已落在桌面上时使用。
        """
        self.piper.MotionCtrl_1(0x02, 0x00, 0x00)


class FakeArm:
    """dry-run 用假臂:输出正弦轨迹,接口与 PiperArm 一致。"""

    is_fake = True

    def __init__(self, can_name: str, phase: float = 0.0):
        self.can_name = can_name
        self._t0 = time.monotonic()
        self._phase = phase
        self._enabled = False
        self.last_cmd: tuple[np.ndarray, float] | None = None

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self) -> ArmState:
        t = (time.monotonic() - self._t0) + self._phase
        base = np.array([0.0, 60.0, -60.0, 0.0, 30.0, 0.0], dtype=np.float32)
        amp = np.array([20.0, 20.0, 20.0, 15.0, 15.0, 30.0], dtype=np.float32)
        joints = base + amp * math.sin(t * 0.6)
        gripper = 50.0 + 40.0 * math.sin(t * 1.2)
        return ArmState(joints_deg=joints.astype(np.float32), gripper_mm=float(gripper), hz=200.0)

    def wait_feedback(self, timeout_s: float = 3.0, min_hz: float = 1.0) -> bool:
        return True

    def enable(self, timeout_s: float = 5.0) -> bool:
        self._enabled = True
        return True

    def disable(self, timeout_s: float = 3.0) -> bool:
        self._enabled = False
        return True

    def is_enabled(self) -> bool:
        return self._enabled

    def send_command(
        self,
        joints_deg: np.ndarray,
        gripper_mm: float,
        speed_rate: int = 100,
        installation_pos: int = 0,
        gripper_effort_nm: float = 1.0,
    ) -> None:
        self.last_cmd = (np.asarray(joints_deg, dtype=float), float(gripper_mm))

    def stop_hold(self) -> None:
        pass

    def recover(self) -> None:
        pass
