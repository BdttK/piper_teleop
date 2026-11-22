#!/usr/bin/env python3
"""Replay one episode of ``RoboSynChallenge/cobotmagic_Real_items_handover`` on the
dual-Piper station, so the recorded motion can be compared against the real arms.

Stages (each one is strictly safer than the next):

    # 1. Offline: parse the episode, print the motion profile. No hardware.
    python scripts/replay_dataset_episode.py --episode 1

    # 2. Connect the arms read-only and compare the dataset's first frame with
    #    the current pose on the station (detects swapped left/right arms).
    python scripts/replay_dataset_episode.py --episode 1 --execute --check-only

    # 3. Move the arms to the episode's first frame and hold there while showing
    #    the dataset video next to the live cameras.
    python scripts/replay_dataset_episode.py --episode 1 --execute --check-only --show-video

    # 4. Full slow replay (arms move!).
    python scripts/replay_dataset_episode.py --episode 1 --execute --speed 0.5

The dataset stores 32-dim state/action in radians + normalized gripper; only the
6 joints and the gripper per arm are sent to the motors (``action_space.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "datasets" / "cobotmagic_Real_items_handover"
DEFAULT_DEPLOYMENT = Path("/home/agilex/gk/FastWAM-RL/deployment")
DEFAULT_CONFIG = DEFAULT_DEPLOYMENT / "configs" / "cobotmagic_real.yaml"

CAM_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
JOINT_NAMES = ["j1_waist", "j2_shoulder", "j3_elbow", "j4_forearm", "j5_wrist", "j6_rotate"]

logger = logging.getLogger("replay_episode")


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_episode(data_root: Path, episode: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(action (T,32), state (T,32), fps)`` for one episode."""
    info = json.loads((data_root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    parquet = data_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"missing {parquet}; download it from the dataset repo first")
    df = pd.read_parquet(parquet).sort_values("frame_index").reset_index(drop=True)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    return action, state, fps


def actions_to_jg(actions: np.ndarray) -> np.ndarray:
    """32-dim dataset action -> 14-dim ``[lj(6), lg, rj(6), rg]``."""
    return np.concatenate(
        [actions[:, 0:7], actions[:, 16:23]], axis=1
    )


def video_path(data_root: Path, episode: int, key: str) -> Path:
    return (
        data_root / "videos" / "chunk-000" / f"observation.images.{key}"
        / f"episode_{episode:06d}.mp4"
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_deg(v: np.ndarray) -> str:
    return " ".join(f"{x:7.1f}" for x in v)


def print_pose(title: str, joints_deg: np.ndarray, grip: float) -> None:
    print(f"{title:<12}{_fmt_deg(joints_deg)}   grip {grip:5.2f}")


def print_comparison(ds_jg: np.ndarray, live_jg: np.ndarray) -> None:
    """Compare the dataset frame with the measured station pose.

    The arms may simply be parked somewhere else, so this is a *hint*, not a
    verdict. Two hypotheses are scored - arms as wired vs. left/right swapped -
    and the lower mean joint error is reported.
    """
    ds_l, ds_lg, ds_r, ds_rg = ds_jg[0:6], ds_jg[6], ds_jg[7:13], ds_jg[13]
    live_l, live_lg, live_r, live_rg = live_jg[0:6], live_jg[6], live_jg[7:13], live_jg[13]

    print("\n关节位置 (deg) —— 数据集 vs 真机当前")
    print(f"{'':<12}" + " ".join(f"{n:>7}" for n in JOINT_NAMES) + "   gripper")
    print_pose("dataset L", np.degrees(ds_l), ds_lg)
    print_pose("current L", np.degrees(live_l), live_lg)
    print_pose("dataset R", np.degrees(ds_r), ds_rg)
    print_pose("current R", np.degrees(live_r), live_rg)

    direct = float(
        np.mean(np.abs(np.degrees(ds_l - live_l))) + np.mean(np.abs(np.degrees(ds_r - live_r)))
    )
    swapped = float(
        np.mean(np.abs(np.degrees(ds_l - live_r))) + np.mean(np.abs(np.degrees(ds_r - live_l)))
    )
    print(f"\n平均关节差: 正常对应 {direct:6.1f}°   左右互换 {swapped:6.1f}°")
    if direct <= swapped:
        print("=> 当前左右臂映射与数据集一致 (数据集 left ↔ 你的 left)")
    else:
        print("=> 注意: 互换后更接近 —— 检查 CAN 左右配置或加 --swap-arms 重放")


def print_profile(action: np.ndarray, state: np.ndarray, fps: int) -> None:
    jg = actions_to_jg(action)
    dur = len(action) / fps
    print(f"frames={len(action)}  fps={fps}  duration={dur:.1f}s  "
          f"joints in radians, gripper normalized [0,1]")
    for side, lo in (("L", 0), ("R", 7)):
        jmin, jmax = np.degrees(jg[:, lo:lo + 6].min(0)), np.degrees(jg[:, lo:lo + 6].max(0))
        gmin, gmax = jg[:, lo + 6].min(), jg[:, lo + 6].max()
        print(f"  arm {side} joint range (deg): min {_fmt_deg(jmin)}")
        print(f"                             max {_fmt_deg(jmax)}   gripper [{gmin:.2f}, {gmax:.2f}]")
    speed = np.abs(np.diff(np.degrees(jg[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]), axis=0)).max() * fps
    print(f"  peak joint speed: {speed:.1f} deg/s")
    print(f"  state/action joint mismatch: max {np.degrees(np.abs(state - action).max()):.2f}°")


# --------------------------------------------------------------------------- #
# Video display
# --------------------------------------------------------------------------- #
class VideoTrack:
    """Sequential decoder for one episode video; ``get(idx)`` returns frame idx."""

    def __init__(self, path: Path, fps: int):
        import av

        self.fps = fps
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.time_base = float(self.stream.time_base)
        self._start()

    def _start(self, seconds: float = 0.0) -> None:
        self.container.seek(int(seconds / self.time_base), stream=self.stream)
        self._gen = self.container.decode(self.stream)
        self._last = None
        self._last_t = -1.0

    def get(self, idx: int) -> np.ndarray | None:
        target = idx / self.fps
        if self._last_t > target + 0.5:  # rewound
            self._start(max(0.0, target - 0.2))
        while self._last is None or self._last_t < target:
            try:
                frame = next(self._gen)
            except StopIteration:
                break
            t = float(frame.pts * self.time_base) if frame.pts is not None else self._last_t
            self._last = frame.to_ndarray(format="rgb24")
            self._last_t = t
        return self._last

    def close(self) -> None:
        self.container.close()


class LiveCameras(threading.Thread):
    """Grabs the three station cameras in the background."""

    def __init__(self, group, fps: float = 30.0):
        super().__init__(daemon=True)
        self.group = group
        self.period = 1.0 / max(1.0, fps)
        self.frames: dict[str, np.ndarray] | None = None
        self.error: Exception | None = None
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self.frames = self.group.read()
                self.error = None
            except Exception as exc:  # keep the control loop alive
                self.error = exc
            self._stop_evt.wait(self.period)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=2.0)


class FrameSink:
    """Shows BGR frames in an ``ffplay`` window (the env's cv2 is headless).

    Frames are pushed through a small drop-oldest queue and written to ffplay's
    stdin by a helper thread, so a slow display never stalls the control loop.
    """

    def __init__(self, width: int, height: int, fps: float = 30.0, title: str = "replay"):
        import subprocess

        self._queue: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=2)
        self._proc = subprocess.Popen(
            [
                "ffplay", "-loglevel", "quiet", "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}", "-framerate", str(max(1.0, fps)),
                "-i", "pipe:0", "-window_title", title,
            ],
            stdin=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._proc.stdin.write(item.tobytes())
            except (BrokenPipeError, OSError, ValueError):
                return

    def push(self, img_bgr: np.ndarray) -> None:
        if not self.alive:
            return
        while True:
            try:
                self._queue.put_nowait(img_bgr)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=2.0)
        except Exception:
            self._proc.kill()


def _cell(img: np.ndarray | None, size=(320, 240)) -> np.ndarray:
    import cv2

    if img is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(np.ascontiguousarray(img), size, interpolation=cv2.INTER_AREA)


def compose_view(ds_frames, live_frames, hud: str) -> np.ndarray:
    import cv2

    top = np.concatenate([_cell(ds_frames.get(k)) for k in CAM_KEYS], axis=1)
    bottom = np.concatenate([_cell(live_frames.get(k) if live_frames else None) for k in CAM_KEYS], axis=1)
    for img, label in ((top, "DATASET"), (bottom, "LIVE")):
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    canvas = np.concatenate([top, bottom], axis=0)
    cv2.rectangle(canvas, (0, canvas.shape[0] - 22), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(canvas, hud, (8, canvas.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--deployment-dir", type=Path, default=DEFAULT_DEPLOYMENT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--execute", action="store_true", help="connect the real arms (default: offline analysis only)")
    p.add_argument("--read-only", action="store_true",
                   help="only read the current pose, never enable or move the arms")
    p.add_argument("--check-only", action="store_true",
                   help="move to the episode's first frame and hold; do not replay the trajectory")
    p.add_argument("--hold-seconds", type=float, default=30.0,
                   help="在 --check-only 结束时保持对比窗口的秒数 (0 = 直到按 q)")
    p.add_argument("--show-video", action="store_true", help="show dataset video next to the live cameras")
    p.add_argument("--speed", type=float, default=0.5, help="replay speed vs. dataset (0.1-2.0)")
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=-1, help="-1 = end of episode")
    p.add_argument("--max-step-deg", type=float, default=2.0, help="max joint step per control cycle")
    p.add_argument("--swap-arms", action="store_true", help="send left commands to the right arm and vice versa")
    p.add_argument("--no-goto-start", action="store_true", help="replay from the measured pose without homing to frame 0")
    p.add_argument("--return-home", action="store_true",
                   help="slowly return to robot.home_pose_deg from the config when done")
    p.add_argument("--home-only", action="store_true",
                   help="only move slowly to robot.home_pose_deg and exit")
    p.add_argument("--disable-at-end", action="store_true")
    p.add_argument("--hold-on-exit", type=float, default=30.0,
                   help="keep the motors enabled for this many seconds before the process exits; "
                        "on exit the firmware drops torque and the arms sag (0 = exit immediately, "
                        "-1 = hold until Enter/Ctrl+C)")
    p.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    return p.parse_args()


def wait_feedback(robot, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(h.feedback_hz > 1.0 for h in robot.read_health()):
            return True
        time.sleep(0.05)
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    action, state, fps = load_episode(args.data_root, args.episode)
    jg_all = actions_to_jg(action)
    T = len(jg_all)
    i0 = max(0, min(args.start_frame, T - 1))
    i1 = T if args.end_frame < 0 else min(args.end_frame + 1, T)
    if i1 - i0 < 2:
        raise SystemExit("frame range too short")

    print(f"\n=== episode {args.episode} of {args.data_root.name} ===")
    print_profile(action, state, fps)

    if not args.execute:
        print("\n(离线模式) 加 --execute 连接真机。建议顺序:")
        print("  1) --execute --check-only --show-video   比较起始位置")
        print("  2) --execute --speed 0.5                 真机重放")
        return 0

    sys.path.insert(0, str(args.deployment_dir))
    from cameras.camera_group import build_camera_group
    from robot_safety import HealthWatchdog, limit_joint_step, move_home
    from robots.cobotmagic_piper import build_robot

    cfg = yaml.safe_load(args.config.read_text())
    robot = build_robot(cfg["robot"], 14)
    camera_group = None
    camera_thread = None
    tracks: list[VideoTrack] = []

    print("\n连接机械臂 ...")
    robot.connect(enable=not args.read_only)
    try:
        if not wait_feedback(robot):
            print("!! 没有关节反馈,检查 CAN (ip -br link show type can) 与机械臂供电")
            return 1

        live = robot.read_proprio()
        print_comparison(jg_all[i0], live)
        print("arm health: " + " | ".join(h.describe() for h in robot.read_health()))
        if args.read_only:
            print("\n只读模式结束,未使能、未发送任何指令。")
            return 0

        if args.home_only:
            from robot_safety import home_targets

            print("返回 home 姿态 (慢速) ...")
            ok = move_home(robot, home_targets(robot, cfg), tol_deg=2.0, timeout=180.0,
                           control_hz=min(20.0, args.control_hz), max_step_deg=1.5)
            print("已回到 home。" if ok else "!! 未能回到 home 姿态")
            return 0

        # Safe-limit clamping: the dataset sits on some hard stops (j2/j3 = 0°,
        # j6 near ±120°); commanding the exact stop trips the firmware fault.
        lo_l, hi_l = robot.left.cfg.safe_joint_limits()
        lo_r, hi_r = robot.right.cfg.safe_joint_limits()
        jg_cmd = jg_all.copy()
        jg_cmd[:, 0:6] = np.clip(jg_cmd[:, 0:6], lo_l, hi_l)
        jg_cmd[:, 7:13] = np.clip(jg_cmd[:, 7:13], lo_r, hi_r)
        clipped = np.abs(jg_cmd - jg_all) * 180 / np.pi
        if clipped.max() > 0.05:
            worst = np.unravel_index(np.argmax(clipped), clipped.shape)
            print(f"!! 数据集姿态触及限位,已按安全余量裁剪 (frame {worst[0]}, "
                  f"joint {worst[1]}, {clipped[worst]:.1f}°)")

        if args.swap_arms:
            jg_cmd = np.concatenate([jg_cmd[:, 7:14], jg_cmd[:, 0:7]], axis=1)

        if not args.yes:
            ans = input("\n回车: 慢速移动到 episode 首帧; 其他键+q 取消: ")
            if ans.strip().lower().startswith("q"):
                print("已取消")
                return 0

        if args.show_video:
            camera_group = build_camera_group(cfg["cameras"])
            camera_group.connect()
            camera_thread = LiveCameras(camera_group, fps=cfg["cameras"].get("fps", 30))
            camera_thread.start()
            for key in CAM_KEYS:
                path = video_path(args.data_root, args.episode, key)
                if path.exists():
                    tracks.append(VideoTrack(path, fps))
                else:
                    logger.warning("video missing: %s", path)

        watchdog = HealthWatchdog()

        if not args.no_goto_start:
            target = {
                "left": (jg_cmd[i0, 0:6].copy(), float(jg_cmd[i0, 6])),
                "right": (jg_cmd[i0, 7:13].copy(), float(jg_cmd[i0, 13])),
            }
            print("移动到首帧姿态 (慢速) ...")
            if not move_home(robot, target, tol_deg=2.0, timeout=180.0,
                             control_hz=min(20.0, args.control_hz), max_step_deg=1.5):
                print("!! 未能到达首帧姿态,放弃重放")
                return 1
            print("\n移动完成,实测 vs 数据集首帧:")
            print_comparison(jg_cmd[i0], robot.read_proprio())

        if args.check_only:
            print("已停在首帧姿态。对照视频/真机检查位置关系,确认无误后去掉 --check-only 重放。")
            if args.show_video and camera_thread is not None:
                _hold_with_view(robot, jg_cmd[i0], camera_thread, tracks, i0, fps,
                                args.hold_seconds)
            return 0

        _replay(robot, watchdog, jg_cmd, i0, i1, fps, args, camera_thread, tracks)
        print("重放结束,机械臂保持当前位置。")
        if args.return_home:
            from robot_safety import home_targets

            print("返回 home 姿态 (慢速) ...")
            if not move_home(robot, home_targets(robot, cfg), tol_deg=2.0, timeout=180.0,
                             control_hz=min(20.0, args.control_hz), max_step_deg=1.5):
                print("!! 未能回到 home 姿态")
            else:
                print("已回到 home。")

    except KeyboardInterrupt:
        print("\n中断,保持当前位置 (力矩未释放)")
        robot.safety_stop()
        return 130
    finally:
        if camera_thread is not None:
            camera_thread.stop()
        if camera_group is not None:
            camera_group.disconnect()
        for track in tracks:
            track.close()
        if not args.disable_at_end:
            _hold_enabled(robot, args.hold_on_exit)
        robot.disconnect(disable=args.disable_at_end)
    return 0


def _hold_enabled(robot, seconds: float) -> None:
    """Keep the arms powered for a grace period before the process exits.

    The Piper firmware cuts power to every motor when the CAN command stream
    stops, so the moment this process ends the arms go limp. Holding here gives
    the operator time to support them or set them down first.
    """
    if seconds == 0:
        return
    try:
        if not (robot.left.is_enabled() and robot.right.is_enabled()):
            return
        jg = robot.read_proprio()
    except Exception:
        return
    label = "直到回车/Ctrl+C" if seconds < 0 else f"{seconds:.0f}s"
    print(f"\n保持使能 {label} —— 进程退出后固件会立即断电,机械臂会因重力下坠。")
    print("  现在可以去扶住/托住机械臂;回车或 Ctrl+C 立即释放退出。")
    release = threading.Event()

    if sys.stdin.isatty():
        def _listen() -> None:
            try:
                input()
            except Exception:
                pass
            release.set()

        threading.Thread(target=_listen, daemon=True).start()

    deadline = None if seconds < 0 else time.monotonic() + seconds
    try:
        while (deadline is None or time.monotonic() < deadline) and not release.is_set():
            robot.send_joint_command(jg[0:6], float(jg[6]), jg[7:13], float(jg[13]))
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


def _replay(robot, watchdog, jg_cmd, i0, i1, fps, args, camera_thread, tracks) -> None:
    from robot_safety import limit_joint_step

    speed = float(np.clip(args.speed, 0.1, 2.0))
    step_s = 1.0 / max(1.0, args.control_hz)
    max_step = math.radians(args.max_step_deg)
    view = args.show_video and camera_thread is not None
    sink = FrameSink(960, 480, fps=min(30.0, 1.0 / step_s),
                     title="replay: DATASET (top) vs LIVE (bottom)") if view else None
    t0 = time.monotonic()
    last_log = 0.0
    lag_since = None
    err_max = 0.0
    done = 0
    clamp_count = 0

    print(f"重放中: speed={speed:.2f}x control={args.control_hz:.0f}Hz  "
          "Ctrl+C 停止 (保持位置)")
    while True:
        faults = watchdog.check(robot)
        if faults:
            print("!! 故障,停止: " + "; ".join(faults))
            robot.safety_stop()
            return

        pos = i0 + (time.monotonic() - t0) * speed * fps
        if pos >= i1 - 1:
            break
        k = int(pos)
        frac = pos - k
        target = jg_cmd[k] * (1.0 - frac) + jg_cmd[k + 1] * frac

        fb = robot.read_proprio()
        lj, lclamped = limit_joint_step(fb[0:6], target[0:6], max_step)
        rj, rclamped = limit_joint_step(fb[7:13], target[7:13], max_step)
        if lclamped or rclamped:
            clamp_count += 1
            if clamp_count <= 3 or clamp_count % 200 == 0:
                logger.warning("命令被限幅 %d 次 (跟不上重放速度),考虑降低 --speed", clamp_count)
        robot.send_joint_command(lj, float(target[6]), rj, float(target[13]))

        err_l = float(np.degrees(np.abs(target[0:6] - fb[0:6]).max()))
        err_r = float(np.degrees(np.abs(target[7:13] - fb[7:13]).max()))
        err_max = max(err_max, err_l, err_r)
        done += 1
        lag = max(err_l, err_r)
        if lag > 20.0:
            lag_since = lag_since or time.monotonic()
            if time.monotonic() - lag_since > 3.0:
                print(f"!! 机械臂跟不上(滞后 {lag:.0f}° > 20° 持续 3s),停止")
                robot.safety_stop()
                return
        else:
            lag_since = None

        now = time.monotonic()
        if now - last_log > 1.0:
            print(f"  frame {k:4d}/{i1 - 1}  t={k / fps:5.1f}s  跟踪误差 L {err_l:4.1f}° R {err_r:4.1f}°  "
                  f"grip L {target[6]:.2f} R {target[13]:.2f}")
            last_log = now

        if sink is not None:
            hud = (f"frame {k}/{i1 - 1}  speed {speed:.2f}x  err L {err_l:.1f} R {err_r:.1f} deg")
            sink.push(compose_view(_dataset_frames(tracks, k), camera_thread.frames, hud))
            if not sink.alive:
                print("显示窗口已关闭,继续重放但不显示画面")
                sink.close()
                sink = None

        time.sleep(step_s)
    print(f"重放完成: {done} 个控制周期, 最大跟踪误差 {err_max:.1f}°")
    if sink is not None:
        sink.close()


def _hold_with_view(robot, target, camera_thread, tracks, idx: int, fps: int,
                    seconds: float = 0.0) -> None:
    """Hold the pose while showing the comparison view.

    The target is re-sent every cycle: Piper drops torque (and the arms sag onto
    their limits) if no motion command arrives for a while.
    """
    sink = FrameSink(960, 480, fps=10, title="dataset (top) vs live (bottom)")
    print("窗口: 关闭 ffplay 窗口或等待自动结束" + (f" ({seconds:.0f}s)" if seconds > 0 else ""))
    deadline = time.monotonic() + seconds if seconds > 0 else None
    from robot_safety import HealthWatchdog

    watchdog = HealthWatchdog()
    while (deadline is None or time.monotonic() < deadline) and sink.alive:
        robot.send_joint_command(target[0:6], float(target[6]), target[7:13], float(target[13]))
        faults = watchdog.check(robot)
        if faults:
            print("!! 保持时出现故障: " + "; ".join(faults))
            robot.safety_stop()
            break
        fb = robot.read_proprio()
        err_l = float(np.degrees(np.abs(target[0:6] - fb[0:6]).max()))
        err_r = float(np.degrees(np.abs(target[7:13] - fb[7:13]).max()))
        sink.push(compose_view(_dataset_frames(tracks, idx), camera_thread.frames,
                               f"first frame {idx}  hold  err L {err_l:.1f} R {err_r:.1f} deg"))
        time.sleep(0.1)
    sink.close()


def _dataset_frames(tracks, idx: int) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for key, track in zip(CAM_KEYS, tracks):
        out[key] = track.get(idx)
    return out


if __name__ == "__main__":
    sys.exit(main())
