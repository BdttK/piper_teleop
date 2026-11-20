# piper_teleop — 双臂 piper 主从遥操数据采集

4 台 piper 机械臂(2 主臂 + 2 从臂,各占一条独立 USB-CAN)+ 3 台 Intel RealSense D435
(左右腕部 + 前方场景)的主从遥操采集,数据保存为 **LeRobot v2.1** 数据集格式。

依赖仓库以 **git submodule** 固定在 `third_party/`(版本锁定,不带内容):

```
third_party/lerobot     # huggingface/lerobot v0.3.3,唯一对应 LeRobot 数据集 v2.1 的版本线(v0.4.0 起为 v3.0)
third_party/piper_sdk   # agilexrobotics/piper_sdk 0.7.0a1
```

克隆时记得带上子模块:

```bash
git clone --recurse-submodules <repo-url> piper_teleop
# 已克隆过:git submodule update --init
```

`install.sh` 发现 `third_party/` 缺失时会自行 clone(不带子模块也能装)。

## 目录结构

```
piper_teleop/
├── install.sh                    # 创建 conda 环境 + 克隆/安装全部依赖
├── configs/bimanual_piper.yaml   # CAN 映射 / 相机序列号 / 数据集与安全参数
├── scripts/
│   ├── setup_can.sh              # 按 USB 口把 4 条 CAN 固定命名并激活
│   ├── check_devices.py          # 只读自检(CAN/相机/4 臂读数)
│   ├── record_teleop.py          # 采集主程序
│   └── verify_dataset.py         # 数据集校验(v2.1/parquet/视频)
├── piper_teleop/                 # 采集代码包(arm/cameras/teleop/recorder/...)
├── third_party/                  # 依赖仓库(git submodule,版本锁定)
└── data/                         # 数据集输出目录
```

## 安装

要求:Linux + conda(需联网)。

```bash
cd /home/agilex/gk/piper_teleop
bash install.sh
conda activate piper_lerobot
```

安装内容:新 conda 环境 `piper_lerobot`(Python 3.12)、CPU 版 torch/torchvision、
`lerobot v0.3.3`(含 pyrealsense2)、`piper_sdk`、pyyaml、以及 torchcodec 解码所需的
conda FFmpeg 7 运行库(通过 `activate.d` 脚本注入 `LD_LIBRARY_PATH`)。结束时自动校验
数据集版本为 `v2.1`。安装/更新环境后请重新 `conda activate piper_lerobot`。

## 硬件配置

### 1. CAN 总线(4 条,1Mbps)

接口名按 USB 口固定,与配置文件一致:

| 接口名 | 用途 |
| --- | --- |
| `can_leader_left` | 左主臂 |
| `can_follower_left` | 左从臂 |
| `can_leader_right` | 右主臂 |
| `can_follower_right` | 右从臂 |

```bash
# 查看每个 CAN 适配器的 USB bus-info(逐个插拔并记录)
bash scripts/setup_can.sh --list

# 把 bus-info 填进 scripts/setup_can.sh 顶部的 USB_PORTS 映射,然后:
sudo bash scripts/setup_can.sh
```

如果现场想用别的接口名,改 `configs/bimanual_piper.yaml` 里 `arms.*.leader/follower` 即可。
`check_devices.py` 里出现 `ERROR-PASSIVE` 说明总线物理层有问题(接线/供电/波特率)。

### 2. 相机

3 台 D435 按**序列号**在 `configs/bimanual_piper.yaml` 中绑定到名字:

| 名字 | 位置 | 当前序列号 |
| --- | --- | --- |
| `left_wrist` | 左从臂腕部 | 327122071526 |
| `right_wrist` | 右从臂腕部 | 327122072690 |
| `front` | 前方场景 | 335622071161 |

腕部相机安装朝向不同,可把 `rotation` 改为 `90/180/270`。数据集中的相机特征名为
`observation.images.<名字>`。

## 采集流程

```bash
conda activate piper_lerobot
python scripts/check_devices.py     # 先自检:4 臂读数 + 相机 + CAN
python scripts/record_teleop.py     # 交互采集
```

交互按键:

| 按键 | 功能 |
| --- | --- |
| `e` | 主臂失能(可手拖)+ 使能从臂,开始主从跟随 |
| 空格 | 开始 / 结束当前 episode(结束时保存) |
| 退格 | 丢弃当前 episode |
| `x` / `Esc` | 急停:从臂停住保持 |
| `q` | 退出(未保存的 episode 会被丢弃) |

数据输出:`data/piper_bimanual_<时间戳>/`
(由 `LeRobotDataset.create` 生成标准 v2.1 布局:`data/` parquet、`videos/` mp4、`meta/`)。

其他常用参数:

```bash
python scripts/record_teleop.py --task "pick the red cup"   # 覆盖任务描述
python scripts/record_teleop.py --resume data/piper_bimanual_20261119_153000  # 向已有数据集追加
python scripts/record_teleop.py --dry-run --fake-cameras --auto 2            # 无硬件自检

# 校验数据集(版本/帧数/视频可读)
python scripts/verify_dataset.py data/piper_bimanual_<时间戳>
```

## 数据格式(LeRobot v2.1)

- `observation.state` / `action`:`float32 (14,)`,顺序为
  `left_joint_1..6, left_gripper, right_joint_1..6, right_gripper`(关节 deg,夹爪 mm);
  `action` 是经限幅/限速后**实际下发**给从臂的目标。
- 相机:`observation.images.left_wrist / right_wrist / front`,RGB 视频。
- 默认 `fps=30`,`robot_type="bimanual_piper"`。

回读校验:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/piper_bimanual", root="data/piper_bimanual_<时间戳>")
print(ds.meta.info["codebase_version"])  # v2.1
print(ds.meta.total_episodes, ds.meta.total_frames)
```

如果解码视频时报 torchcodec / FFmpeg 错误,先确认已 `conda activate piper_lerobot`
(安装脚本会注入 conda 内 FFmpeg 库路径),或改用 lerobot 的 PyAV 后端:
`LeRobotDataset(..., video_backend="pyav")`。

## 安全与标定

- **主臂失能后没有重力补偿**,会下坠。开始前先用手托住主臂,或在低位启动。
- 从臂只在你按 `e` 后使能;退出、急停、异常都会先"停住保持"再失能。
- 主臂反馈丢失(>1Hz 无数据)、CAN 异常、写帧异常都会自动触发急停并退出。
- 主从方向/零点:若从臂镜像安装导致方向相反,用 `joint_sign`(逐关节 ±1)修正;
  零点偏差用 `joint_offset_deg`;夹爪零点用 `gripper_zero_mm` 标定。
- 速度/限位:默认最大关节指令速度 180°/s,软限位在 piper 硬限位内收 2°,
  夹爪行程上限 100mm —— 均在 `configs/bimanual_piper.yaml` 调整。

## 常见问题

- **接口名不对/插拔后变了**:重新执行 `sudo bash scripts/setup_can.sh`。
- **打开接口时报 permission 错误**:RealSense 需要 udev 规则,通常系统已配置;
  否则重插设备或安装 Intel librealsense udev 规则后重启。
- **从臂不跟随**:确认按过 `e`、`GetArmEnableStatus` 全为 true、急停未被按下;
  先用 `check_devices.py` 看 4 臂反馈 Hz。
- **数据帧与时间戳不同步**:录制线程按固定 fps 抽帧;相机超时会复用上一帧并继续,
  持续超时会报错退出。
