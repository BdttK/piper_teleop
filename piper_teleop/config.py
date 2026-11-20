"""配置加载:configs/*.yaml -> AppConfig。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ArmConfig:
    leader: str
    follower: str
    joint_sign: list[float] = field(default_factory=lambda: [1.0] * 6)
    joint_offset_deg: list[float] = field(default_factory=lambda: [0.0] * 6)
    gripper_sign: float = 1.0
    gripper_zero_mm: float = 0.0
    gripper_effort_nm: float = 1.0

    def validate(self) -> None:
        if len(self.joint_sign) != 6 or len(self.joint_offset_deg) != 6:
            raise ValueError(f"{self.leader}: joint_sign/joint_offset_deg 必须为 6 维")


@dataclass
class CameraConfig:
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    rotation: int = 0


@dataclass
class DatasetConfig:
    repo_id: str = "local/piper_bimanual"
    out_root: str = "data"
    use_videos: bool = True
    image_writer_threads: int = 4
    robot_type: str = "bimanual_piper"


@dataclass
class LimitsConfig:
    max_joint_speed_deg_s: float = 180.0
    soft_margin_deg: float = 2.0
    gripper_max_mm: float = 100.0


@dataclass
class AppConfig:
    fps: int
    control_hz: int
    task: str
    watchdog_timeout_s: float
    dataset: DatasetConfig
    limits: LimitsConfig
    arms: dict[str, ArmConfig]
    cameras: dict[str, CameraConfig]
    speed_rate: int = 100
    installation_pos: int = 0

    @property
    def out_root(self) -> Path:
        path = Path(self.dataset.out_root)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def validate(self) -> None:
        if not self.arms:
            raise ValueError("配置缺少 arms")
        if not self.cameras:
            raise ValueError("配置缺少 cameras")
        if self.fps <= 0 or self.control_hz <= 0:
            raise ValueError("fps/control_hz 必须为正")
        for name, arm in self.arms.items():
            try:
                arm.validate()
            except ValueError as exc:
                raise ValueError(f"arms.{name}: {exc}") from exc


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = AppConfig(
        fps=int(raw.get("fps", 30)),
        control_hz=int(raw.get("control_hz", 200)),
        task=str(raw.get("task", "default task")),
        watchdog_timeout_s=float(raw.get("watchdog_timeout_s", 0.5)),
        dataset=DatasetConfig(**(raw.get("dataset") or {})),
        limits=LimitsConfig(**(raw.get("limits") or {})),
        arms={k: ArmConfig(**v) for k, v in (raw.get("arms") or {}).items()},
        cameras={k: CameraConfig(**v) for k, v in (raw.get("cameras") or {}).items()},
        speed_rate=int(raw.get("speed_rate", 100)),
        installation_pos=int(raw.get("installation_pos", 0)),
    )
    cfg.validate()
    return cfg
