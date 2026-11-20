"""LeRobot v2.1 数据集录制封装(lerobot v0.3.3)。"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import AppConfig

logger = logging.getLogger(__name__)

STATE_NAMES = (
    [f"left_joint_{i}" for i in range(1, 7)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(1, 7)]
    + ["right_gripper"]
)


def build_features(cfg: AppConfig) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
    }
    for name, cam in cfg.cameras.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class EpisodeRecorder:
    """管理 LeRobotDataset:建库/追加 episode/保存/丢弃。"""

    def __init__(self, cfg: AppConfig, resume_root: str | None = None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        self.features = build_features(cfg)
        self.task = cfg.task
        threads = cfg.dataset.image_writer_threads

        if resume_root:
            self.root = Path(resume_root).expanduser().resolve()
            self.dataset = LeRobotDataset(
                cfg.dataset.repo_id, root=self.root, batch_encoding_size=1
            )
            self.dataset.start_image_writer(0, threads)
            logger.info("追加到已有数据集: %s", self.root)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = cfg.dataset.repo_id.split("/")[-1]
            self.root = cfg.out_root / f"{name}_{stamp}"
            self.dataset = LeRobotDataset.create(
                repo_id=cfg.dataset.repo_id,
                fps=cfg.fps,
                features=self.features,
                root=self.root,
                robot_type=cfg.dataset.robot_type,
                use_videos=cfg.dataset.use_videos,
                image_writer_threads=threads,
                # 每个 episode 立即编码,异常中断时不丢已保存的数据
                batch_encoding_size=1,
            )
            logger.info("创建新数据集: %s", self.root)

        self.frame_count = 0
        self.episode_active = False

    def start_episode(self, task: str | None = None) -> None:
        if task:
            self.task = task
        self.frame_count = 0
        self.episode_active = True

    def add_frame(
        self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray]
    ) -> None:
        frame = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }
        for name, image in images.items():
            frame[f"observation.images.{name}"] = image
        self.dataset.add_frame(frame, task=self.task)
        self.frame_count += 1

    def save_episode(self) -> int:
        frames = self.frame_count
        self.dataset.save_episode()
        self.episode_active = False
        self.frame_count = 0
        return frames

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
        self.episode_active = False
        self.frame_count = 0

    @property
    def total_episodes(self) -> int:
        return self.dataset.meta.total_episodes

    def close(self) -> None:
        self.dataset.stop_image_writer()
