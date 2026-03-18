#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the standalone critic launcher.
# Usage:
#   bash scripts/run_trado_critic.sh configs/rl_trado_critic_mbpp_with_value.yaml

CONFIG_PATH="${1:?pass a config path}"
shift || true

# Pass all extra OmegaConf overrides through to the launcher so smoke tests and
# one-off machine-specific runs do not require editing tracked config files.
python rl_trado_critic.py "config=${CONFIG_PATH}" "$@"
