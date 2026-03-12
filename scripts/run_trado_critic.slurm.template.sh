#!/usr/bin/env bash
# Example Slurm wrapper. Fill in the cluster-specific resource requests.
#
# sbatch scripts/run_trado_critic.slurm.template.sh configs/rl_trado_critic_mbpp_with_value.yaml

#SBATCH --job-name=trado-critic
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG_PATH="${1:?pass a config path}"

python rl_trado_critic.py "config=${CONFIG_PATH}"
