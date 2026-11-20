#!/usr/bin/env python
"""双臂 piper 主从遥操采集主程序(LeRobot v2.1 数据集格式)。

用法示例:
    conda activate piper_lerobot
    # 交互采集(先按 e 使能从臂,空格开始/结束 episode)
    python scripts/record_teleop.py
    # 无硬件自检:假臂 + 假相机,自动录 2 个 5 秒 episode
    python scripts/record_teleop.py --dry-run --fake-cameras --auto 2
    # 只测相机接真臂
    python scripts/record_teleop.py --task "pick the cup"
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from piper_teleop.arm import FakeArm, PiperArm
from piper_teleop.cameras import CameraSet, FakeCameraSet
from piper_teleop.config import load_config
from piper_teleop.keyboard import KeyReader
from piper_teleop.recorder import EpisodeRecorder
from piper_teleop.teleop import BimanualTeleop, ControlLoop

HELP_TEXT = """
---------------- 采集控制 ----------------
  e        使能从臂并开始主从跟随(主臂会先失能,可手拖)
  空格      开始 / 结束当前 episode(结束时保存)
  退格      丢弃当前 episode
  x / Esc  急停:从臂停住并保持
  q        退出(未保存的 episode 会被丢弃)
------------------------------------------
"""


class RecordThread(threading.Thread):
    """按数据集 fps 抽帧写入;主线程只负责 start/stop 指令。"""

    def __init__(self, cfg, recorder: EpisodeRecorder, cameras, teleop: BimanualTeleop):
        super().__init__(daemon=True, name="record-loop")
        self.cfg = cfg
        self.recorder = recorder
        self.cameras = cameras
        self.teleop = teleop
        self._lock = threading.Lock()
        self._command: tuple | None = None
        self._stop_event = threading.Event()
        self.status = "idle"
        self.frames = 0
        self.error: str | None = None

    # ---- 主线程接口 ----
    def begin(self) -> None:
        with self._lock:
            self._command = ("start",)

    def end(self, discard: bool = False) -> None:
        with self._lock:
            self._command = ("stop", discard)

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def recording(self) -> bool:
        return self.status == "recording"

    # ---- 线程内部 ----
    def _handle_command(self, command: tuple) -> None:
        kind = command[0]
        if kind == "start":
            if self.status == "recording":
                return
            self.recorder.start_episode(self.cfg.task)
            self.frames = 0
            self.status = "recording"
        elif kind == "stop" and self.status == "recording":
            discard = command[1]
            if discard or self.frames == 0:
                self.recorder.discard_episode()
                self.status = "discarded" if self.frames else "empty"
            else:
                frames = self.recorder.save_episode()
                self.status = f"saved:{frames}"
                print(
                    f"\n[采集] episode 已保存({frames} 帧),"
                    f"累计 {self.recorder.total_episodes} 个 episode"
                )
            self.frames = 0

    def run(self) -> None:
        period = 1.0 / self.cfg.fps
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                command = self._command
                self._command = None
            if command is not None:
                try:
                    self._handle_command(command)
                except Exception as exc:
                    logging.exception("录制命令执行失败")
                    self._fail(f"录制命令执行失败: {exc}")
            if self.status == "recording":
                try:
                    snapshot = self.teleop.latest()
                    images = self.cameras.read_all()
                    self.recorder.add_frame(snapshot.state, snapshot.action, images)
                    self.frames += 1
                except Exception as exc:
                    logging.exception("写帧失败")
                    self._fail(f"写帧异常: {exc}")
                    try:
                        self.recorder.discard_episode()
                    except Exception:
                        logging.debug("丢弃缓冲异常", exc_info=True)
                    with self._lock:
                        self._command = None
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
        if self.status == "recording":
            try:
                self.recorder.discard_episode()
            except Exception:
                pass
            self.status = "discarded"

    def _fail(self, message: str) -> None:
        self.error = message
        self.status = "error"
        self.frames = 0
        self.teleop.request_abort(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双臂 piper 主从遥操采集(LeRobot v2.1)")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "bimanual_piper.yaml")
    )
    parser.add_argument("--dry-run", action="store_true", help="使用假臂,不连接真实机械臂")
    parser.add_argument("--fake-cameras", action="store_true", help="使用合成相机画面")
    parser.add_argument(
        "--auto", type=int, default=0, metavar="N", help="自动录制 N 个 episode 后退出(测试用)"
    )
    parser.add_argument("--auto-seconds", type=float, default=5.0, help="自动模式单个 episode 时长")
    parser.add_argument("--task", default=None, help="覆盖配置中的任务描述")
    parser.add_argument("--resume", default=None, metavar="DIR", help="向已有数据集目录追加 episode")
    return parser.parse_args()


def build_arms(cfg, dry_run: bool) -> tuple[dict, dict]:
    leaders, followers = {}, {}
    for index, (side, arm_cfg) in enumerate(cfg.arms.items()):
        if dry_run:
            # 从臂相位错开,便于在数据里区分 state(从臂)与 action(主臂映射)
            leaders[side] = FakeArm(arm_cfg.leader, phase=2.0 * index)
            followers[side] = FakeArm(arm_cfg.follower, phase=2.0 * index + 1.0)
        else:
            leaders[side] = PiperArm(arm_cfg.leader)
            followers[side] = PiperArm(arm_cfg.follower)
    return leaders, followers


def connect_all(arms: dict) -> bool:
    for label, arm in arms.items():
        try:
            arm.connect()
        except Exception as exc:
            print(f"[错误] 连接 {label} ({arm.can_name}) 失败: {exc}")
            return False
    for label, arm in arms.items():
        if not arm.wait_feedback(2.0):
            print(f"[错误] {label} ({arm.can_name}) 无反馈,检查 CAN 接线/电源/接口名")
            return False
    return True


def run_auto(args, cfg, teleop: BimanualTeleop, record: RecordThread) -> int:
    if not teleop.followers_enabled():
        print("[自动] 使能从臂...")
        if not teleop.enable_followers():
            print("[错误] 从臂使能失败,检查急停/供电")
            return 1
    teleop.start_following()
    time.sleep(0.5)

    for index in range(args.auto):
        print(f"[自动] 录制 episode {index + 1}/{args.auto}({args.auto_seconds}s)")
        record.begin()
        time.sleep(args.auto_seconds)
        record.end()
        deadline = time.monotonic() + 30.0
        while record.recording and time.monotonic() < deadline:
            time.sleep(0.05)
        if record.error:
            print(f"[错误] 录制失败: {record.error}")
            return 1
        time.sleep(0.3)
    return 0


def run_interactive(cfg, teleop: BimanualTeleop, record: RecordThread) -> int:
    print(HELP_TEXT)
    last_status = 0.0
    with KeyReader() as keys:
        while True:
            if teleop.error:
                print(f"\n[错误] {teleop.error}\n程序退出。")
                return 1
            key = keys.poll(0.2)
            if key in ("q", "ctrl-c"):
                return 0
            elif key == "e":
                if not teleop.followers_enabled():
                    print("[安全] 使能从臂...")
                    if not teleop.enable_followers():
                        print("[错误] 从臂使能失败,检查急停/供电")
                        continue
                    print("[安全] 主臂即将失能(可手拖),请先托住主臂!")
                    teleop.release_leaders()
                teleop.start_following()
                print("[状态] 已开始主从跟随(空格开始录制)")
            elif key == "space":
                if record.recording:
                    record.end()
                elif not teleop.following:
                    print("[提示] 先按 e 使能从臂")
                else:
                    record.begin()
                    print(f"[采集] 开始录制: {cfg.task}")
            elif key == "backspace":
                if record.recording:
                    record.end(discard=True)
                    print("[采集] 已丢弃当前 episode")
            elif key in ("x", "esc"):
                print("[安全] 急停:从臂停住保持")
                teleop.stop_hold()

            now = time.monotonic()
            if now - last_status >= 1.0:
                last_status = now
                if record.recording:
                    rec = f"录制中 {record.frames} 帧"
                else:
                    rec = record.status
                follow = "开" if teleop.following else "关"
                print(f"[状态] 跟随={follow} 录制={rec}")


def shutdown(teleop, control, record, cameras, recorder, leaders, followers) -> None:
    print("\n[安全] 停止采集...")
    record.end(discard=True)
    record.request_stop()
    record.join(timeout=15.0)
    control.request_stop()
    control.join(timeout=3.0)
    try:
        teleop.stop_hold()
        time.sleep(0.2)
        teleop.disable_followers()
    except Exception:
        logging.debug("失能从臂异常", exc_info=True)
    for arm in list(leaders.values()) + list(followers.values()):
        arm.close()
    try:
        cameras.disconnect()
    except Exception:
        logging.debug("相机断开异常", exc_info=True)
    recorder.close()
    print(f"[完成] 数据集: {recorder.root}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.task:
        cfg.task = args.task

    print(f"配置: {args.config}")
    print(f"任务: {cfg.task}")
    print(f"fps={cfg.fps} control_hz={cfg.control_hz} 相机={list(cfg.cameras)}")

    leaders, followers = build_arms(cfg, args.dry_run)
    arms: dict = {}
    for side in cfg.arms:
        arms[f"主臂-{side}"] = leaders[side]
        arms[f"从臂-{side}"] = followers[side]

    print("[1/4] 连接机械臂...")
    if not connect_all(arms):
        for arm in arms.values():
            arm.close()
        return 1

    print("[2/4] 连接相机...")
    camera_cls = FakeCameraSet if args.fake_cameras else CameraSet
    cameras = camera_cls(cfg.cameras)
    try:
        cameras.connect()
    except Exception as exc:
        print(f"[错误] 相机连接失败: {exc}")
        for arm in arms.values():
            arm.close()
        return 1

    print("[3/4] 初始化遥操与数据集...")
    teleop = BimanualTeleop(cfg, leaders, followers)
    recorder = EpisodeRecorder(cfg, resume_root=args.resume)
    print(f"  数据集: {recorder.root}")

    print("[4/4] 启动采集...")
    control = ControlLoop(teleop, cfg.control_hz)
    record = RecordThread(cfg, recorder, cameras, teleop)
    control.start()
    record.start()

    exit_code = 0
    try:
        if args.auto > 0:
            exit_code = run_auto(args, cfg, teleop, record)
        else:
            if not sys.stdin.isatty():
                print("[提示] 当前不是交互终端,请用 --auto N 自动模式")
                return 1
            exit_code = run_interactive(cfg, teleop, record)
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C")
    finally:
        shutdown(teleop, control, record, cameras, recorder, leaders, followers)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
