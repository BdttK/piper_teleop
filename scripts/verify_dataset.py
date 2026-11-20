#!/usr/bin/env python
"""校验数据集是否为 LeRobot v2.1,并抽查 parquet 与视频可读。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 LeRobot v2.1 数据集")
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "--video-backend",
        default=None,
        help="视频解码后端,默认 torchcodec,失败时可试 pyav",
    )
    args = parser.parse_args()

    root = Path(args.dataset_dir).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"[错误] 不是数据集目录(缺 meta/info.json): {root}")
        return 1

    info = json.loads(info_path.read_text())
    version = info["codebase_version"]
    print(f"codebase_version : {version}")
    if version != "v2.1":
        print(f"[警告] 版本不是 v2.1(当前 {version})")
    print(f"fps              : {info['fps']}")
    print(f"robot_type       : {info.get('robot_type')}")
    for key, feature in info["features"].items():
        print(f"  feature {key:34s} {feature['dtype']:8s} shape={feature['shape']}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/check", root=root, video_backend=args.video_backend)
    print(
        f"episodes={ds.meta.total_episodes} frames={ds.meta.total_frames} "
        f"tasks={list(ds.meta.tasks.values())}"
    )

    episodes_file = root / "meta" / "episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_file.open(encoding="utf-8")]
    print("episode lengths  :", [ep["length"] for ep in episodes])

    offset = 0
    for ep in episodes:
        for frame_index in (offset, offset + ep["length"] - 1):
            item = ds[frame_index]
            for key in ds.meta.camera_keys:
                assert tuple(item[key].shape[-2:]) == (480, 640), (key, item[key].shape)
            assert tuple(item["observation.state"].shape) == (14,)
            assert tuple(item["action"].shape) == (14,)
        offset += ep["length"]

    videos = sorted((root / "videos").rglob("*.mp4"))
    print(f"videos           : {len(videos)} 个 mp4")
    print("[OK] 数据集校验通过(parquet + 视频 + meta 均可读)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
