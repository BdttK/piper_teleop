"""多路 RealSense 相机(复用 vendored lerobot 的 RealSenseCamera)与假相机。"""

from __future__ import annotations

import logging
import time

import numpy as np

from .config import CameraConfig

logger = logging.getLogger(__name__)

_ROTATION_NAMES = {0: "NO_ROTATION", 90: "ROTATE_90", 180: "ROTATE_180", 270: "ROTATE_270"}


class CameraSet:
    """一组 RealSense RGB 相机,异步取最新帧。"""

    def __init__(self, configs: dict[str, CameraConfig]):
        from lerobot.cameras.configs import ColorMode, Cv2Rotation
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        self.cameras: dict = {}
        for name, cam in configs.items():
            if cam.rotation not in _ROTATION_NAMES:
                raise ValueError(f"相机 {name}: rotation 只能是 0/90/180/270,当前 {cam.rotation}")
            cfg = RealSenseCameraConfig(
                serial_number_or_name=cam.serial,
                fps=cam.fps,
                width=cam.width,
                height=cam.height,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation[_ROTATION_NAMES[cam.rotation]],
                use_depth=False,
            )
            self.cameras[name] = RealSenseCamera(cfg)
        self._last: dict[str, np.ndarray | None] = {}

    def connect(self) -> None:
        self._last = {name: None for name in self.cameras}
        for name, cam in self.cameras.items():
            cam.connect()
            logger.info("相机已连接: %s", name)

    def read_all(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for name, cam in self.cameras.items():
            try:
                image = cam.async_read(timeout_ms=500)
                self._last[name] = image
            except TimeoutError:
                image = self._last[name]
                if image is None:
                    raise RuntimeError(f"相机 {name} 超时且无历史帧") from None
            frames[name] = image
        return frames

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                logger.debug("相机断开异常", exc_info=True)


class FakeCameraSet:
    """合成随机的渐变画面,无相机时用于验证录制链路。"""

    def __init__(self, configs: dict[str, CameraConfig]):
        self.shapes = {name: (cam.height, cam.width) for name, cam in configs.items()}
        self._t0 = time.monotonic()

    def connect(self) -> None:
        pass

    def read_all(self) -> dict[str, np.ndarray]:
        t = time.monotonic() - self._t0
        frames: dict[str, np.ndarray] = {}
        for i, (name, (height, width)) in enumerate(self.shapes.items()):
            x = np.linspace(0, 200, width, dtype=np.float32)[None, :]
            y = np.linspace(0, 200, height, dtype=np.float32)[:, None]
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0] = np.clip(x + 40 * i + 30 * np.sin(t), 0, 255).astype(np.uint8)
            image[..., 1] = np.clip(y + 20 * np.cos(t), 0, 255).astype(np.uint8)
            image[..., 2] = np.uint8((i * 60 + 30 * t) % 255)
            frames[name] = image
        return frames

    def disconnect(self) -> None:
        pass
