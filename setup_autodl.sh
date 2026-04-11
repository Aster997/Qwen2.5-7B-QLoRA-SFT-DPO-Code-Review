#!/bin/bash
# ═══════════════════════════════════════════════════════════
# AutoDL vGPU 48GB 环境一键搭建脚本
# 适配：A100/A800 vGPU，CUDA 12.1+，PyTorch 2.x（镜像预装）
# 运行方式：bash setup_autodl.sh
# ═══════════════════════════════════════════════════════════

set -e  # 任何命令失败即退出

echo "==== [0/6] 检查基础环境 ===="
nvidia-smi
python --version
pip --version

# AutoDL 学术加速源（国内镜像，速度快）
PIP_OPTS="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

echo ""
echo "==== [1/6] 升级基础工具 ===="
pip install $PIP_OPTS --upgrade pip setuptools wheel

echo ""
echo "==== [2/6] 安装 LLaMA-Factory ===="
# 克隆到 /root/autodl-tmp 避免占用系统盘
if [ ! -d "/root/autodl-tmp/LLaMA-Factory" ]; then
    cd /root/autodl-tmp
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    # 备用（若 GitHub 慢）：
    # git clone https://gitee.com/hiyouga/LLaMA-Factory.git
else
    echo "LLaMA-Factory 已存在，跳过克隆"
fi
cd /root/autodl-tmp/LLaMA-Factory
pip install $PIP_OPTS -e ".[torch,metrics]"

echo ""
echo "==== [3/6] 安装核心训练依赖 ===="
pip install $PIP_OPTS \
    "transformers>=4.43.0" \
    "peft>=0.12.0" \
    "trl>=0.9.0" \
    "accelerate>=0.33.0" \
    "datasets>=2.20.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=3.20.0" \
    "tensorboard>=2.17.0" \
    "scipy>=1.13.0"

echo ""
echo "==== [4/6] 安装 bitsandbytes（NF4 量化必须） ===="
# bitsandbytes >= 0.43 支持 CUDA 12.x
pip install $PIP_OPTS "bitsandbytes>=0.43.3"
# 验证
python -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__} OK')"

echo ""
echo "==== [5/6] 安装 Flash Attention 2（vGPU A100/A800 支持 FA2）===="
# 优先尝试预编译 wheel（秒装），失败再本地编译（约 15 分钟）
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda.replace('.',''))")
echo "检测到 torch=$TORCH_VER, cuda=$CUDA_VER"

FA2_INSTALLED=false
pip install $PIP_OPTS flash-attn --no-build-isolation && FA2_INSTALLED=true || true

if [ "$FA2_INSTALLED" = false ]; then
    echo "预编译 wheel 安装失败，尝试源码编译（约 10-20 分钟）..."
    pip install $PIP_OPTS flash-attn --no-build-isolation --no-cache-dir
fi

python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} OK')"

echo ""
echo "==== [6/6] 安装数据脚本依赖 ===="
cd /root/autodl-tmp
# 克隆你的项目（替换为你的实际 git 地址）
# git clone https://github.com/YOUR_USERNAME/qwen-code-reviewer.git
# cd qwen-code-reviewer

pip install $PIP_OPTS \
    "openai>=1.35.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "numpy>=1.26.0"

echo ""
echo "==== 注册数据集到 LLaMA-Factory ===="
# 项目的 dataset_info.json 已包含 LLaMA-Factory 所有默认条目 + 自定义 v3 条目
# 可以直接覆盖 LLaMA-Factory 的 data/dataset_info.json
DATA_DIR="/root/autodl-tmp/LLaMA-Factory/data"
PROJECT_DATA="/root/autodl-tmp/qwen-code-reviewer/data"

if [ -d "$PROJECT_DATA" ]; then
    echo "正在注册数据集..."
    # 覆盖 dataset_info.json（包含全部默认+自定义条目）
    cp "$PROJECT_DATA/dataset_info.json" "$DATA_DIR/dataset_info.json"
    echo "  dataset_info.json 已更新"

    # 拷贝训练数据到 LLaMA-Factory data 目录
    if [ -d "$PROJECT_DATA/final" ]; then
        cp "$PROJECT_DATA/final/"*.json "$DATA_DIR/"
        echo "  final/*.json 已拷贝到 $DATA_DIR"
    else
        echo "  警告：$PROJECT_DATA/final 不存在，请先运行 build_final_dataset.py 生成训练集"
    fi
else
    echo "警告：项目 data 目录不存在，请先上传项目文件后重新运行此段"
    echo "  手动执行："
    echo "    cp /path/to/project/data/dataset_info.json $DATA_DIR/dataset_info.json"
    echo "    cp /path/to/project/data/final/*.json $DATA_DIR/"
fi

echo ""
echo "==== 环境验证 ===="
python -c "
import torch
import transformers
import peft
import trl
import bitsandbytes as bnb
import flash_attn
print(f'torch:          {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'GPU:            {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'transformers:   {transformers.__version__}')
print(f'peft:           {peft.__version__}')
print(f'trl:            {trl.__version__}')
print(f'bitsandbytes:   {bnb.__version__}')
print(f'flash_attn:     {flash_attn.__version__}')
"

echo ""
echo "========================================================"
echo " 环境搭建完成！"
echo " 下一步："
echo "   1. 上传数据集到 /root/autodl-tmp/qwen-code-reviewer/data/final/"
echo "   2. 把 dataset_info.json 拷贝到 LLaMA-Factory/data/"
echo "   3. 把 json 数据文件也拷到 LLaMA-Factory/data/"
echo "   4. cd /root/autodl-tmp/LLaMA-Factory"
echo "   5. llamafactory-cli train /root/autodl-tmp/qwen-code-reviewer/configs/qwen_qlora_sft_v3.yaml"
echo "========================================================"
