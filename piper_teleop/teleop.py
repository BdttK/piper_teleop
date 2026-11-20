"""双臂主从映射与安全处理。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .arm import JOINT_LIMITS_DEG
from .config import AppConfig, ArmConfig

logger = logging.getLogger(__name__)

NUM_DIMS = 14  # 左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪


@dataclass
class TeleopSnapshot:
    state: np.ndarray  # (14,) 从臂反馈
    action: np.ndarray  # (14,) 实际下发的目标
    timestamp: float


class SideTeleop:
    """单侧(左/右)主臂→从臂映射:方向/偏置、软限位、斜率限制、夹爪。"""

    def __init__(self, name: str, arm_cfg: ArmConfig, leader, follower, cfg: AppConfig):
        self.name = name
        self.cfg = arm_cfg
        self.leader = leader
        self.follower = follower
        self.limits = cfg.limits
        self.control_hz = cfg.control_hz
        self.speed_rate = cfg.speed_rate
        self.installation_pos = cfg.installation_pos
        self.sign = np.asarray(arm_cfg.joint_sign, dtype=np.float64)
        self.offset = np.asarray(arm_cfg.joint_offset_deg, dtype=np.float64)
        self.lower = np.array([lo for lo, _ in JOINT_LIMITS_DEG], dtype=np.float64) + cfg.limits.soft_margin_deg
        self.upper = np.array([hi for _, hi in JOINT_LIMITS_DEG], dtype=np.float64) - cfg.limits.soft_margin_deg
        self._last_cmd: np.ndarray | None = None

    def reset(self) -> None:
        self._last_cmd = None

    def compute_action(self, leader_state) -> np.ndarray:
        joints = self.sign * np.asarray(leader_state.joints_deg, dtype=np.float64) + self.offset
        joints = np.clip(joints, self.lower, self.upper)
        if self._last_cmd is not None:
            max_step = self.limits.max_joint_speed_deg_s / self.control_hz
            joints = self._last_cmd + np.clip(joints - self._last_cmd, -max_step, max_step)
            joints = np.clip(joints, self.lower, self.upper)
        gripper = self.cfg.gripper_sign * (leader_state.gripper_mm - self.cfg.gripper_zero_mm)
        gripper = float(np.clip(gripper, 0.0, self.limits.gripper_max_mm))
        self._last_cmd = joints.copy()
        return np.concatenate([joints, [gripper]]).astype(np.float32)

    def send(self, action: np.ndarray) -> None:
        self.follower.send_command(
            action[:6],
            action[6],
            speed_rate=self.speed_rate,
            installation_pos=self.installation_pos,
            gripper_effort_nm=self.cfg.gripper_effort_nm,
        )


class BimanualTeleop:
    """双臂遥操:控制线程周期性调用 step();共享最新快照供录制线程读取。"""

    def __init__(self, cfg: AppConfig, leaders: dict, followers: dict):
        missing = [s for s in ("left", "right") if s not in cfg.arms]
        if missing:
            raise ValueError(f"配置缺少 arm 侧: {missing}")
        self.cfg = cfg
        self.sides = {
            side: SideTeleop(side, cfg.arms[side], leaders[side], followers[side], cfg)
            for side in ("left", "right")
        }
        self.following = False
        self.error: str | None = None
        self._snapshot = TeleopSnapshot(
            state=np.zeros(NUM_DIMS, dtype=np.float32),
            action=np.zeros(NUM_DIMS, dtype=np.float32),
            timestamp=time.monotonic(),
        )

    def latest(self) -> TeleopSnapshot:
        return self._snapshot

    def step(self) -> None:
        if self.error is not None:
            return

        side_states: dict[str, np.ndarray] = {}
        side_actions: dict[str, np.ndarray] = {}
        for side, side_teleop in self.sides.items():
            try:
                leader_state = side_teleop.leader.read()
            except Exception as exc:
                self.request_abort(f"{side} 主臂 {side_teleop.leader.can_name} 读取异常: {exc}")
                return
            if not getattr(side_teleop.leader, "is_fake", False) and leader_state.hz < 1.0:
                self.request_abort(f"{side} 主臂反馈丢失 (Hz={leader_state.hz:.1f})")
                return

            action = side_teleop.compute_action(leader_state)

            try:
                follower_state = side_teleop.follower.read()
            except Exception as exc:
                self.request_abort(f"{side} 从臂 {side_teleop.follower.can_name} 读取异常: {exc}")
                return
            if not getattr(side_teleop.follower, "is_fake", False) and follower_state.hz < 1.0:
                self.request_abort(f"{side} 从臂反馈丢失 (Hz={follower_state.hz:.1f})")
                return

            if self.following:
                try:
                    side_teleop.send(action)
                except Exception as exc:
                    self.request_abort(f"{side} 从臂下发异常: {exc}")
                    return

            side_actions[side] = action
            side_states[side] = np.concatenate(
                [follower_state.joints_deg, [follower_state.gripper_mm]]
            ).astype(np.float32)

        self._snapshot = TeleopSnapshot(
            state=np.concatenate([side_states["left"], side_states["right"]]).astype(np.float32),
            action=np.concatenate([side_actions["left"], side_actions["right"]]).astype(np.float32),
            timestamp=time.monotonic(),
        )

    # ---- 安全 ----
    def stop_hold(self) -> None:
        self.following = False
        for side_teleop in self.sides.values():
            try:
                side_teleop.follower.stop_hold()
            except Exception:
                logger.debug("stop_hold 异常", exc_info=True)

    def request_abort(self, message: str) -> None:
        if self.error is None:
            logger.error(message)
            self.error = message
        self.stop_hold()

    def enable_followers(self) -> bool:
        ok = True
        for side_teleop in self.sides.values():
            follower = side_teleop.follower
            if not follower.enable(5.0):
                logger.error("使能从臂失败: %s", follower.can_name)
                ok = False
            side_teleop.reset()
        return ok

    def disable_followers(self) -> None:
        self.following = False
        for side_teleop in self.sides.values():
            side_teleop.follower.disable(3.0)

    def release_leaders(self) -> None:
        """主臂失能以便手动拖动(注意:无重力补偿,松手会下坠)。"""
        for side_teleop in self.sides.values():
            side_teleop.leader.disable(3.0)

    def start_following(self) -> None:
        """解除急停、重置指令斜率并开始主从跟随。"""
        for side_teleop in self.sides.values():
            try:
                side_teleop.follower.recover()
            except Exception:
                logger.debug("recover 异常", exc_info=True)
            side_teleop.reset()
        self.following = True

    def followers_enabled(self) -> bool:
        return all(s.follower.is_enabled() for s in self.sides.values())


class ControlLoop(threading.Thread):
    """固定频率执行 teleop.step() 的线程。"""

    def __init__(self, teleop: BimanualTeleop, control_hz: int):
        super().__init__(daemon=True, name="control-loop")
        self.teleop = teleop
        self.control_hz = control_hz
        self._stop_event = threading.Event()

    def run(self) -> None:
        period = 1.0 / self.control_hz
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            try:
                self.teleop.step()
            except Exception as exc:
                self.teleop.request_abort(f"控制循环异常: {exc}")
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def request_stop(self) -> None:
        self._stop_event.set()
