#!/bin/bash
#SBATCH --job-name=trado-base-eval
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --output=/dcs/large/u5671205/baseline_eval/slurm-%j.out
#SBATCH --error=/dcs/large/u5671205/baseline_eval/slurm-%j.err

echo "Job started on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

source /etc/profile.d/modules.sh
module load CUDA/12.4.1
source /etc/profile.d/conda.sh
conda activate dllm-rl

export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

cd /dcs/pg24/u5671205/ai-project/repos/DiffTraceRL

mkdir -p /dcs/large/u5671205/baseline_eval
ln -sfn /dcs/large/u5671205/baseline_eval baseline_eval

python eval_base_model.py \
    config=configs/rl_trado_critic_mbpp_with_value.yaml \
    model.pretrained_model=/dcs/large/u5671205/models/TraDo-4B-Instruct \
    experiment.project=baseline_eval \
    experiment.project_dir=/dcs/large/u5671205/baseline_eval \
    experiment.function=evaluation \
    experiment.current_epoch=1 \
    experiment.start_from_scratch=true \
    evaluation.eval_dataset=mbpp_critic_eval \
    evaluation.num_response_per_task=8 \
    evaluation.max_token=512 \
    evaluation.top_k=0 \
    evaluation.remasking_strategy=low_confidence_dynamic

echo "Job finished: $(date)"
