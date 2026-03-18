# Trado Critic Runs

This document explains the critic-training workflow that uses workspace-level
data under `ai-project/data` and repo-local runtime outputs under
`ai-project/repos/dLLM-RL/<project>/`.

## Workspace layout

Expected layout:

```text
ai-project/
  data/
    mbpp_dataset/
    humaneval_dataset/
    codecontests_dataset/
    mbpp_critic.jsonl
    humaneval_critic.jsonl
    codecontests_pilot_1000_critic.jsonl
    mbpp_masking_trajectories.jsonl
    humaneval_masking_trajectories.jsonl
    codecontests_pilot_1000_masking_trajectories.jsonl
  repos/
    dLLM-RL/
```

## 1. Download raw datasets

Run from `repos/dLLM-RL`:

```bash
python data/download_raw_code_dataset.py --dataset mbpp
python data/download_raw_code_dataset.py --dataset humaneval
python data/download_raw_code_dataset.py --dataset codecontests
```

## 2. Prepare normalized critic-source JSONL

```bash
python data/prepare_critic_code_data.py --dataset mbpp
python data/prepare_critic_code_data.py --dataset humaneval
python data/prepare_critic_code_data.py --dataset codecontests --pilot-size 1000
```

## 3. Build GPT-2 masking trajectories

```bash
python data/build_entropy_masking_trajectories.py --input-jsonl ../../data/mbpp_critic.jsonl --output-jsonl ../../data/mbpp_masking_trajectories.jsonl
python data/build_entropy_masking_trajectories.py --input-jsonl ../../data/humaneval_critic.jsonl --output-jsonl ../../data/humaneval_masking_trajectories.jsonl
python data/build_entropy_masking_trajectories.py --input-jsonl ../../data/codecontests_pilot_1000_critic.jsonl --output-jsonl ../../data/codecontests_pilot_1000_masking_trajectories.jsonl
```

## 4. Launch critic RL training

Set model paths first in the config you want to run, then launch:

```bash
python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_with_value.yaml experiment.preflight_only=true
python rl_trado_critic.py config=configs/rl_trado_critic_humaneval_with_value.yaml experiment.preflight_only=true
python rl_trado_critic.py config=configs/rl_trado_critic_codecontests_with_value.yaml experiment.preflight_only=true
```

If preflight succeeds, run the full training command:

```bash
python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_with_value.yaml
python rl_trado_critic.py config=configs/rl_trado_critic_humaneval_with_value.yaml
python rl_trado_critic.py config=configs/rl_trado_critic_codecontests_with_value.yaml
```

## 5. Resume a stopped run

Edit the config before restarting:
- set `experiment.start_from_scratch: false`
- set `experiment.current_epoch` to the last finished epoch

Then rerun the same command.

## Outputs

Each run writes to its own project directory:
- checkpoints under `<project>/ckpt/`
- rollout files and reward JSON under `<project>/temp_data/`
- result summaries under `<project>/results/`
