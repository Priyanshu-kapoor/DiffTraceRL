#!/usr/bin/env bash
set -euo pipefail

# Bring up a Linux training environment that matches the upstream dLLM-RL stack
# closely enough for the Trado critic pipeline to run.
#
# Assumptions:
# - this script is run on a Linux machine, not on native Windows
# - conda is already installed
# - the shell can see the target NVIDIA GPU(s)
#
# You can override these before running the script:
#   ENV_NAME=dllm-rl
#   PYTHON_VERSION=3.10
#   PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV_NAME="${ENV_NAME:-dllm-rl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH. Install Miniconda/Anaconda first."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found in PATH. Install NVIDIA drivers before training."
  exit 1
fi

# The conda shell hook is required so this script can activate the env in-place.
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y --name "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel ninja packaging

# Use the CUDA-enabled PyTorch wheel set explicitly instead of relying on
# whatever pip would choose by default on this machine.
python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch==2.6.0 torchvision==0.21.0

# The upstream repo already documents this Linux wheel, so keep the default here
# aligned with that recommendation.
python -m pip install --no-cache-dir "${FLASH_ATTN_WHEEL_URL}"

python -m pip install -r requirements.txt

# These are the most common gaps for the Trado training path. They are included
# in requirements.txt, but installing them once more here keeps the setup
# instructions explicit and easier to debug if one package failed earlier.
python -m pip install deepspeed xformers triton flashinfer-python

python scripts/check_linux_training_env.py

echo
echo "Environment setup finished."
echo "Next step:"
echo "  python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_with_value.yaml experiment.preflight_only=true"
