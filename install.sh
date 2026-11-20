#!/usr/bin/env bash
# 一键安装:新建 conda 环境并安装 lerobot v0.3.3(LeRobot v2.1 数据集格式)与 piper_sdk。
# 第三方仓库固定在 third_party/ 的 git submodule;缺失时这里会按固定版本自行 clone。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-piper_lerobot}"
PY_VER="${PY_VER:-3.12}"
LEROBOT_URL="https://github.com/huggingface/lerobot"
LEROBOT_TAG="v0.3.3"   # 唯一对应 LeRobot 数据集 v2.1 的版本线(v0.4.0 起为 v3.0)
PIPER_URL="https://github.com/agilexrobotics/piper_sdk"

# ---------- 1. conda 环境 ----------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '!/^#/ {print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python="$PY_VER" -y
fi
conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel

# ---------- 2. 克隆依赖仓库(vendored) ----------
mkdir -p third_party
if [ ! -d third_party/lerobot/.git ]; then
    git clone --depth 1 -b "$LEROBOT_TAG" "$LEROBOT_URL" third_party/lerobot
fi
if [ ! -d third_party/piper_sdk/.git ]; then
    git clone --depth 1 "$PIPER_URL" third_party/piper_sdk
fi

# ---------- 3. 安装依赖 ----------
# 先装 CPU 版 torch(满足 lerobot 的 >=2.2.1,<2.8 约束;数据采集不需要 CUDA)
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.1,<2.8" "torchvision>=0.21,<0.23"

python -m pip install -e "third_party/lerobot[intelrealsense]"
python -m pip install -e third_party/piper_sdk
python -m pip install -r requirements.txt

# ---------- 3.5 torchcodec 解码所需的 FFmpeg 7 共享库 ----------
conda install -n "$ENV_NAME" -c conda-forge -y "ffmpeg=7"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/10_ffmpeg_libs.sh" <<'EOF'
# torchcodec 需要能加载 conda 环境内的 FFmpeg 7 共享库(libavutil.so.59 等)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---------- 4. 校验 ----------
python - <<'PY'
import lerobot
import piper_sdk
import pyrealsense2
import torch
import yaml
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from torchcodec.decoders import VideoDecoder  # noqa: F401  # 验证 FFmpeg 运行库可加载

print(f"lerobot={lerobot.__version__} dataset={CODEBASE_VERSION} torch={torch.__version__}")
assert CODEBASE_VERSION == "v2.1", f"数据集版本不是 v2.1: {CODEBASE_VERSION}"
print("环境安装完成。")
PY

echo
echo "安装完成。使用前先激活环境: conda activate $ENV_NAME"
