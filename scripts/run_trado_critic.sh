#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the standalone critic launcher.
# Usage:
#   bash scripts/run_trado_critic.sh configs/rl_trado_critic_mbpp_with_value.yaml

CONFIG_PATH="${1:?pass a config path}"

python rl_trado_critic.py "config=${CONFIG_PATH}"

