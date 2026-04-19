# Trado Critic Runs

This document explains the standalone Trado critic-training workflow.

Important environment note:
- full RL training is meant for Linux, not native Windows
- local native Windows is fine for CPU-safe checks, dataset prep, and launcher preflight debugging
- the first real GPU-backed smoke run should happen on the professor's Linux machine

Runtime layout:
- workspace-level data lives under `ai-project/data`
- repo-local outputs live under `ai-project/repos/dLLM-RL/<project>/`

## Which model to use

For the first smoke run, use a **Trado checkpoint**, not GPT-2.

- Recommended first choice: `Gen-Verse/TraDo-4B-Instruct`
- Larger alternative if the machine has enough GPU memory: `Gen-Verse/TraDo-8B-Instruct`

Important:
- GPT-2 is used only in the offline trajectory-building step to score tokens by entropy
- the actual critic RL pipeline runs on a Trado model
- for the first smoke run, set both `PRETRAINED_MODEL` and `VALUE_BASE_MODEL` to the **same Trado checkpoint**

That means the simplest valid first setup is:

```bash
export PRETRAINED_MODEL=Gen-Verse/TraDo-4B-Instruct
export VALUE_BASE_MODEL=Gen-Verse/TraDo-4B-Instruct
```

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

## 1. Prepare the Linux training environment

Run this on the Linux training machine from `repos/dLLM-RL`:

```bash
bash scripts/setup_linux_training_env.sh
```

If you want to inspect the environment without reinstalling anything:

```bash
python scripts/check_linux_training_env.py
```

That check must pass before any critic RL run.

## 2. Download raw datasets

Run from `repos/dLLM-RL`:

```bash
python data/download_raw_code_dataset.py --dataset mbpp
python data/download_raw_code_dataset.py --dataset humaneval
python data/download_raw_code_dataset.py --dataset codecontests
```

## 3. Prepare normalized critic-source JSONL

```bash
python data/prepare_critic_code_data.py --dataset mbpp
python data/prepare_critic_code_data.py --dataset humaneval
python data/prepare_critic_code_data.py --dataset codecontests --pilot-size 1000
```

## 4. Build GPT-2 masking trajectories

```bash
python data/build_entropy_masking_trajectories.py --input-jsonl ../../data/mbpp_critic.jsonl --output-jsonl ../../data/mbpp_masking_trajectories.jsonl
python data/build_entropy_masking_trajectories.py --input-jsonl ../../data/humaneval_critic.jsonl --output-jsonl ../../data/humaneval_masking_trajectories.jsonl
python data/build_entropy_masking_trajectories.py --input-jsonl ../../data/codecontests_pilot_1000_critic.jsonl --output-jsonl ../../data/codecontests_pilot_1000_masking_trajectories.jsonl
```

## 5. Run launcher preflight

Set real model paths first, then run launcher preflight. You can either edit the
config file or override the model paths on the command line.

```bash
python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_with_value.yaml model.pretrained_model=Gen-Verse/TraDo-4B-Instruct model.value_base_model=Gen-Verse/TraDo-4B-Instruct experiment.preflight_only=true
python rl_trado_critic.py config=configs/rl_trado_critic_humaneval_with_value.yaml experiment.preflight_only=true
python rl_trado_critic.py config=configs/rl_trado_critic_codecontests_with_value.yaml experiment.preflight_only=true
```

Preflight must succeed before rollout or training.

## 6. Run the first real smoke test

Start with MBPP. The helper script below runs preflight first and then launches
a 1-step smoke run using CLI overrides instead of editing the tracked config.

```bash
export PRETRAINED_MODEL=Gen-Verse/TraDo-4B-Instruct
export VALUE_BASE_MODEL=Gen-Verse/TraDo-4B-Instruct
bash scripts/run_trado_critic_mbpp_smoke.sh
```

If the model is already downloaded locally, you can replace the Hugging Face id
with the local checkpoint path.

Recommended exact first-run order:

```bash
bash scripts/setup_linux_training_env.sh
python scripts/check_linux_training_env.py
python data/download_raw_code_dataset.py --dataset mbpp
python data/prepare_critic_code_data.py --dataset mbpp
python data/build_entropy_masking_trajectories.py --input-jsonl ../../data/mbpp_critic.jsonl --output-jsonl ../../data/mbpp_masking_trajectories.jsonl
export PRETRAINED_MODEL=Gen-Verse/TraDo-4B-Instruct
export VALUE_BASE_MODEL=Gen-Verse/TraDo-4B-Instruct
python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_with_value.yaml model.pretrained_model=Gen-Verse/TraDo-4B-Instruct model.value_base_model=Gen-Verse/TraDo-4B-Instruct experiment.preflight_only=true
bash scripts/run_trado_critic_mbpp_smoke.sh
```

The smoke run is considered successful if it:
- starts rollout
- builds the expert cache
- writes critic/value/policy artifacts
- writes an evaluation result file

## 7. Run the full training command

Once the MBPP smoke run works, use the normal launcher:

```bash
python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_with_value.yaml
python rl_trado_critic.py config=configs/rl_trado_critic_humaneval_with_value.yaml
python rl_trado_critic.py config=configs/rl_trado_critic_codecontests_with_value.yaml
```

Recommended order:
1. MBPP smoke and short run
2. HumanEval preflight and smoke
3. CodeContests preflight and smoke

## 8. Resume a stopped run

Edit the config before restarting:
- set `experiment.start_from_scratch: false`
- set `experiment.current_epoch` to the last finished epoch

Then rerun the same command.

## Outputs

Each run writes to its own project directory:
- checkpoints under `<project>/ckpt/`
- rollout files and reward JSON under `<project>/temp_data/`
- result summaries under `<project>/results/`
