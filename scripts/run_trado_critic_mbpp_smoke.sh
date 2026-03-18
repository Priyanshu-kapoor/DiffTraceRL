#!/usr/bin/env bash
set -euo pipefail

# Small Linux smoke test for the standalone Trado critic pipeline.
#
# This script intentionally uses CLI overrides instead of editing the tracked
# config so professor-side bring-up is easy to repeat and easy to diff.
#
# Required environment variables:
#   PRETRAINED_MODEL   - policy/base Trado checkpoint or HF model id
# Optional environment variables:
#   VALUE_BASE_MODEL   - value-model base; defaults to PRETRAINED_MODEL
#   PROJECT_NAME       - output directory name for the smoke run

CONFIG_PATH="${1:-configs/rl_trado_critic_mbpp_with_value.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

: "${PRETRAINED_MODEL:?Set PRETRAINED_MODEL to a real Trado checkpoint or HF id.}"
VALUE_BASE_MODEL="${VALUE_BASE_MODEL:-${PRETRAINED_MODEL}}"
PROJECT_NAME="${PROJECT_NAME:-rl_trado_critic_mbpp_smoke}"

COMMON_ARGS=(
  "model.pretrained_model=${PRETRAINED_MODEL}"
  "model.value_base_model=${VALUE_BASE_MODEL}"
  "experiment.project=${PROJECT_NAME}"
  "experiment.start_from_scratch=true"
  "experiment.total_step=1"
  "experiment.eval_every=1"
  "experiment.save_every=1"
  "rollout.num_task_per_step=2"
  "rollout.num_response_per_task=1"
  "rollout.max_active=16"
  "evaluation.num_response_per_task=1"
  "evaluation.max_active=16"
  "execute.num_chunk=2"
  "critic.num_steps_per_epoch=2"
  "critic.batch_size=4"
  "training.gradient_accumulation_steps=1"
)

echo "[smoke] running launcher preflight first"
bash scripts/run_trado_critic.sh "$CONFIG_PATH" "${COMMON_ARGS[@]}" experiment.preflight_only=true "$@"

echo "[smoke] launching 1-step MBPP critic RL run"
bash scripts/run_trado_critic.sh "$CONFIG_PATH" "${COMMON_ARGS[@]}" experiment.preflight_only=false "$@"
