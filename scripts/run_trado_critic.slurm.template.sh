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
shift || true

# Activate your training environment here before launching, for example:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate dllm-rl

# Run the launcher preflight first so the batch job fails fast on missing
# models, missing data, or GPU/config mismatches.
bash scripts/run_trado_critic.sh "$CONFIG_PATH" experiment.preflight_only=true "$@"
bash scripts/run_trado_critic.sh "$CONFIG_PATH" "$@"
