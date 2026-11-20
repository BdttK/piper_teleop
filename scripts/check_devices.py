#!/usr/bin/env python
"""只读自检:CAN 接口、RealSense 相机、四台 piper 的连接与关节读数。

不会使能或运动任何机械臂,可以安全地在采集前执行。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from piper_teleop.arm import PiperArm
from piper_teleop.config import load_config


def list_can_interfaces() -> None:
    print("== CAN 接口 ==")
    try:
        result = subprocess.run(
            ["ip", "-br", "link", "show", "type", "can"], capture_output=True, text=True
        )
    except FileNotFoundError:
        print("  未找到 ip 命令")
        return
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        print("  未检测到 CAN 接口(检查 gs_usb 驱动与 USB 连接)")
        return
    for line in lines:
        iface = line.split()[0]
        up = "UP" in line.split()
        details = subprocess.run(
            ["ip", "-details", "link", "show", iface], capture_output=True, text=True
        ).stdout
        match = re.search(r"bitrate (\d+)", details)
        bitrate = match.group(1) if match else "-"
        warning = ""
        if "ERROR-PASSIVE" in details or "BUS-OFF" in details:
            warning = "  <-- 总线异常:检查接线/供电/波特率"
        print(f"  {iface:20s} {'UP' if up else 'DOWN':4s} bitrate={bitrate}{warning}")


def list_cameras() -> None:
    print("== RealSense 相机 ==")
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  未安装 pyrealsense2(conda 环境里执行 install.sh)")
        return
    devices = list(rs.context().query_devices())
    if not devices:
        print("  未检测到 RealSense 设备")
        return
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"  {name}  SN={serial}")


def check_arms(cfg) -> None:
    print("== piper 机械臂(只读,不使能/不运动) ==")
    seen: set[str] = set()
    for side, arm_cfg in cfg.arms.items():
        for role, can_name in (("主臂", arm_cfg.leader), ("从臂", arm_cfg.follower)):
            if can_name in seen:
                continue
            seen.add(can_name)
            arm = None
            try:
                arm = PiperArm(can_name)
                arm.connect()
                if not arm.wait_feedback(2.0):
                    print(f"  {role} {can_name:18s} 无反馈(检查接口名/CAN 接线/电源)")
                    continue
                state = arm.read()
                joints = " ".join(f"{value:7.1f}" for value in state.joints_deg)
                enabled = "是" if arm.is_enabled() else "否"
                print(
                    f"  {role} {can_name:18s} Hz={state.hz:5.1f} "
                    f"关节(deg)=[{joints}] 夹爪={state.gripper_mm:6.1f}mm 使能={enabled}"
                )
            except Exception as exc:
                print(f"  {role} {can_name:18s} 连接失败: {exc}")
            finally:
                if arm is not None:
                    arm.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="CAN/相机/piper 只读自检")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "bimanual_piper.yaml")
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    list_can_interfaces()
    print()
    list_cameras()
    print()
    check_arms(cfg)
    print("\n自检完成(未使能/未运动任何机械臂)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
