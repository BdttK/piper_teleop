#!/usr/bin/env python3
"""Generate a height-compensated LeRobot v2.1 copy of a cobotmagic episode.

The source dataset was recorded with the table at a certain height. If the
station's table is lower, every end-effector target has to move down by the
same amount **relative to the robot base**. This script recomputes the joint
trajectory with Piper forward/inverse kinematics so the end-effector path is
shifted by ``--dz`` while keeping the gripper orientation as close as possible.

Only the joints + `ee_pos` + `ee_rot6d` in `observation.state`/`action` change.
Videos are copied unchanged; all meta files are rewritten for the new episode.

Usage:
    python scripts/offset_dataset_z.py                       # dry-run, prints stats
    python scripts/offset_dataset_z.py --write               # generate the copy
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "data" / "datasets" / "cobotmagic_Real_items_handover"

# Piper DH parameters (dh_is_offset=1, same table as piper_sdk's C_PiperForwardKinematics).
_A = [0.0, 0.0, 285.03, -21.98, 0.0, 0.0]
_ALPHA = [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2]
_THETA = [0.0, -math.pi * 172.22 / 180, -102.78 / 180 * math.pi, 0.0, 0.0, 0.0]
_D = [123.0, 0.0, 0.0, 250.75, 0.0, 91.0]

# Commandable joint range = hard stop minus the driver's 1.5 deg safety margin.
LIMITS_RAD = np.radians(np.array([
    [-150.0, 148.5], [1.5, 178.5], [-168.5, -1.5],
    [-98.5, 98.5], [-68.5, 68.5], [-118.5, 118.5],
]))
JOINT_NAMES = ["j1_waist", "j2_shoulder", "j3_elbow", "j4_forearm", "j5_wrist", "j6_rotate"]
ARM_OFFSETS = {"left": 0, "right": 16}
CAM_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #
def fk(q: np.ndarray) -> np.ndarray:
    """Piper forward kinematics. Joints in rad; returns a 4x4 pose in mm."""
    T = np.eye(4)
    for i in range(6):
        th = q[i] + _THETA[i]
        ct, st = math.cos(th), math.sin(th)
        ca, sa = math.cos(_ALPHA[i]), math.sin(_ALPHA[i])
        T = T @ np.array([
            [ct, -st, 0, _A[i]],
            [st * ca, ct * ca, -sa, -sa * _D[i]],
            [st * sa, ct * sa, ca, ca * _D[i]],
            [0, 0, 0, 1],
        ])
    return T


def rotvec(R: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    return ang / (2.0 * math.sin(ang)) * np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1],
    ])


def rot6d(R: np.ndarray) -> np.ndarray:
    return R[:, :2].reshape(-1)


def ik(q0: np.ndarray, p_target: np.ndarray, R_target: np.ndarray,
       rot_weight: float, iters: int = 40) -> tuple[np.ndarray, float, float]:
    """Damped least-squares IK. Returns (q, pos_err_mm, rot_err_deg)."""
    q = q0.copy()
    for _ in range(iters):
        T = fk(q)
        e = np.concatenate([p_target - T[:3, 3], rot_weight * rotvec(R_target @ T[:3, :3].T) * 1000.0])
        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2:
            break
        J = np.zeros((6, 6))
        h = 1e-6
        for j in range(6):
            qp = q.copy()
            qp[j] += h
            Tp = fk(qp)
            J[:3, j] = (Tp[:3, 3] - T[:3, 3]) / h
            J[3:, j] = rot_weight * rotvec(Tp[:3, :3] @ T[:3, :3].T) * 1000.0 / h
        dq = J.T @ np.linalg.solve(J @ J.T + 1e3 * np.eye(6), e)
        m = float(np.abs(dq).max())
        if m > 0.03:
            dq *= 0.03 / m
        dq[((q <= LIMITS_RAD[:, 0] + 1e-9) & (dq < 0)) |
           ((q >= LIMITS_RAD[:, 1] - 1e-9) & (dq > 0))] = 0.0
        q = np.clip(q + dq, LIMITS_RAD[:, 0], LIMITS_RAD[:, 1])
    T = fk(q)
    return q, float(np.linalg.norm(p_target - T[:3, 3])), \
        float(np.degrees(np.linalg.norm(rotvec(R_target @ T[:3, :3].T))))


# --------------------------------------------------------------------------- #
# Compensation
# --------------------------------------------------------------------------- #
def compensate(values: np.ndarray, dz_mm: float, rot_weight: float) -> tuple[np.ndarray, dict]:
    """Shift the end-effector path of a (T,32) state/action array down by dz_mm."""
    out = values.copy()
    report: dict[str, list] = {}
    for side, o in ARM_OFFSETS.items():
        rows = []
        for i in range(len(values)):
            q0 = values[i, o:o + 6]
            T = fk(q0)
            p_target = T[:3, 3] + np.array([0.0, 0.0, dz_mm])
            q_new, pos_err, rot_err = ik(q0, p_target, T[:3, :3], rot_weight)
            out[i, o:o + 6] = q_new
            Tn = fk(q_new)
            out[i, o + 7:o + 10] = Tn[:3, 3] / 1000.0  # FK-consistent ee_pos (m)
            out[i, o + 10:o + 16] = rot6d(Tn[:3, :3])
            rows.append((i, pos_err, rot_err, float(np.degrees(np.abs(q_new - q0).max()))))
        a = np.asarray(rows)
        report[side] = {
            "pos_err_mm_mean": round(float(a[:, 1].mean()), 3),
            "pos_err_mm_max": round(float(a[:, 1].max()), 3),
            "rot_err_deg_max": round(float(a[:, 2].max()), 3),
            "joint_delta_deg_max": round(float(a[:, 3].max()), 3),
            "frames_over_2mm": [int(x) for x in a[a[:, 1] > 2.0, 0]],
        }
    return out, report


def print_report(report: dict, dz_mm: float, n: int) -> None:
    print(f"\n补偿 dz={dz_mm:+.1f}mm, {n} 帧")
    for side in ("left", "right"):
        r = report[side]
        bad = r["frames_over_2mm"]
        span = f"{bad[0]}..{bad[-1]}" if bad else "-"
        print(f"  {side:5s}: 位置误差 均值 {r['pos_err_mm_mean']:.2f} 最大 {r['pos_err_mm_max']:.2f} mm"
              f" | 姿态偏差 ≤ {r['rot_err_deg_max']:.2f}° | 关节改动 ≤ {r['joint_delta_deg_max']:.1f}°")
        print(f"          >2mm 帧 {len(bad)} 个" + (f" (范围 {span})" if bad else ""))


# --------------------------------------------------------------------------- #
# Dataset writing
# --------------------------------------------------------------------------- #
def stats_for(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [len(a)],
    }


def write_dataset(src: Path, dst: Path, episode: int, dz_mm: float, rot_weight: float) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    ep_meta = next(e for e in episodes if e["episode_index"] == episode)
    ep_stats = [json.loads(l) for l in (src / "meta" / "episodes_stats.jsonl").read_text().splitlines() if l.strip()]
    ep_stat = next(e for e in ep_stats if e["episode_index"] == episode)

    parquet = src / info["data_path"].format(episode_chunk=0, episode_index=episode)
    df = pd.read_parquet(parquet).sort_values("frame_index").reset_index(drop=True)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)

    new_state, rep_s = compensate(state, dz_mm, rot_weight)
    new_action, rep_a = compensate(action, dz_mm, rot_weight)
    report = {"source_episode": episode, "dz_mm": dz_mm, "rot_weight": rot_weight,
              "state": rep_s, "action": rep_a}
    print_report(rep_s, dz_mm, len(state))
    print("  (action 同 state，省略)")

    new_ep = 0  # the copy contains one episode, renumbered from 0
    df["observation.state"] = list(new_state.astype(np.float32))
    df["action"] = list(new_action.astype(np.float32))
    df["episode_index"] = new_ep
    df["index"] = np.arange(len(df), dtype=np.int64)

    out_parquet = dst / info["data_path"].format(episode_chunk=0, episode_index=new_ep)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    for key in CAM_KEYS:
        v_src = src / info["video_path"].format(episode_chunk=0, video_key=f"observation.images.{key}",
                                                episode_index=episode)
        v_dst = dst / info["video_path"].format(episode_chunk=0, video_key=f"observation.images.{key}",
                                                episode_index=new_ep)
        v_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v_src, v_dst)

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    info["total_episodes"] = 1
    info["total_frames"] = len(df)
    info["total_videos"] = len(CAM_KEYS)
    info["splits"] = {"train": "0:1"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    shutil.copy2(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")

    ep_meta_new = dict(ep_meta, episode_index=new_ep, length=len(df))
    (dst / "meta" / "episodes.jsonl").write_text(json.dumps(ep_meta_new) + "\n")

    stats = dict(ep_stat["stats"])
    stats["observation.state"] = stats_for(new_state)
    stats["action"] = stats_for(new_action)
    stats["episode_index"] = {"min": [new_ep], "max": [new_ep], "mean": [new_ep], "std": [0.0], "count": [1]}
    stats["index"] = {"min": [0], "max": [len(df) - 1], "mean": [float(np.mean(np.arange(len(df))))],
                      "std": [float(np.std(np.arange(len(df))))], "count": [len(df)]}
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    (dst / "meta" / "episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": new_ep, "stats": stats}) + "\n")
    (dst / "z_offset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n已生成: {dst}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=None,
                   help="output dataset root (default: <src>_z-15mm)")
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--dz", type=float, default=-0.015, help="vertical offset in meters (default -0.015)")
    p.add_argument("--rot-weight", type=float, default=0.02,
                   help="orientation weight in IK (0 = position only, higher = stiffer orientation)")
    p.add_argument("--write", action="store_true", help="actually write the dataset (default: dry-run)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dz_mm = args.dz * 1000.0
    info = json.loads((args.src / "meta" / "info.json").read_text())
    parquet = args.src / info["data_path"].format(episode_chunk=0, episode_index=args.episode)
    df = pd.read_parquet(parquet)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)

    new_state, rep = compensate(state, dz_mm, args.rot_weight)
    print(f"episode {args.episode}: {len(state)} 帧, dz={dz_mm:+.1f}mm, rot_weight={args.rot_weight}")
    print_report(rep, dz_mm, len(state))
    delta = new_state - state
    print(f"  state 关节改动最大 {np.degrees(np.abs(delta[:, :6]).max()):.1f}°(左) "
          f"{np.degrees(np.abs(delta[:, 16:22]).max()):.1f}°(右)")
    print(f"  ee_pos z 实际平均下移 {((new_state[:, 9] - state[:, 9]).mean() * 1000):+.2f}mm (左) "
          f"{((new_state[:, 25] - state[:, 25]).mean() * 1000):+.2f}mm (右)")

    if not args.write:
        print("\n(dry-run) 加 --write 生成数据集副本")
        return 0
    dst = args.dst or args.src.parent / f"{args.src.name}_z{int(round(dz_mm)):+d}mm"
    if dst.exists():
        print(f"!! {dst} 已存在,先删除或指定 --dst")
        return 1
    write_dataset(args.src, dst, args.episode, dz_mm, args.rot_weight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
