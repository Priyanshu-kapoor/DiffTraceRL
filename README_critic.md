# Trado Critic Runs

This document explains the standalone Trado critic-training workflow.

Important environment note:
- full RL training is meant for Linux, not native Windows
- local native Windows is fine for CPU-safe checks, dataset prep, and launcher preflight debugging
- the first real GPU-backed smoke run should happen on the Linux machine

Runtime layout:
- workspace-level data lives under `ai-project/data`
- repo-local outputs live under `ai-project/repos/dLLM-RL/<project>/`

## 0. Download the model from HuggingFace

The training pipeline needs the Trado-4B-Instruct weights on disk before it can run.
Download them once to a local directory and then point config paths at that directory.

**Option A — huggingface-cli (recommended)**
```bash
pip install -U huggingface_hub          # already in requirements.txt
huggingface-cli download Gen-Verse/TraDo-4B-Instruct \
    --repo-type model \
    --local-dir /path/to/models/TraDo-4B-Instruct
```

**Option B — Python snapshot download**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Gen-Verse/TraDo-4B-Instruct",
    local_dir="/path/to/models/TraDo-4B-Instruct",
)
```

**Option C — git lfs (requires git-lfs installed)**
```bash
git lfs install
git clone https://huggingface.co/Gen-Verse/TraDo-4B-Instruct \
    /path/to/models/TraDo-4B-Instruct
```

After downloading, note the full path — you will use it as `model.pretrained_model`
and `model.value_base_model` in the config (both point to the same checkpoint for
the first run).

**Future model (LLaDA 2.0):** when ready to switch, download `GSAI-ML/LLaDA-8B-Instruct`
(or its 2.0 successor) the same way and set `model.model_base: llada` in the config.
The training scripts already support LLaDA.

## GPU requirements and automatic config selection

The training scripts automatically select the right DeepSpeed accelerate config
based on the number of visible GPUs — no manual config change is needed when
moving between setups:

| Setup | DeepSpeed config auto-selected |
|-------|-------------------------------|
| 1× TITAN RTX 24 GB | `1_node_1_gpu_deepspeed_zero3.yaml` |
| 2× L40S 48 GB | `1_node_2_gpus_deepspeed_zero3.yaml` |

All configs use ZeRO-3 with CPU offload for both parameters and the optimizer,
so Trado-4B fits comfortably on a single 24 GB GPU.

To override the auto-selection, set `experiment.deepspeed_file` in the config to
the desired filename (without the `.yaml` extension).

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

Set real model paths first, then run launcher preflight.
Use `configs/rl_trado_critic_mbpp_linux.yaml` — it has `deepspeed_file: auto`
so it works on both 1-GPU and 2-GPU setups without any edits.

```bash
MODEL=/path/to/models/TraDo-4B-Instruct
python rl_trado_critic.py \
    config=configs/rl_trado_critic_mbpp_linux.yaml \
    model.pretrained_model=$MODEL \
    model.value_base_model=$MODEL \
    experiment.preflight_only=true
```

Preflight must succeed (no errors) before rollout or training.

## 6. Run the first real smoke test

```bash
MODEL=/path/to/models/TraDo-4B-Instruct
export PRETRAINED_MODEL=$MODEL
export VALUE_BASE_MODEL=$MODEL
bash scripts/run_trado_critic_mbpp_smoke.sh configs/rl_trado_critic_mbpp_linux.yaml
```

Recommended exact first-run order:

```bash
# 1. environment
bash scripts/setup_linux_training_env.sh
python scripts/check_linux_training_env.py

# 2. model download (once)
huggingface-cli download Gen-Verse/TraDo-4B-Instruct \
    --repo-type model --local-dir /path/to/models/TraDo-4B-Instruct

# 3. datasets
python data/download_raw_code_dataset.py --dataset mbpp
python data/prepare_critic_code_data.py --dataset mbpp
python data/build_entropy_masking_trajectories.py \
    --input-jsonl ../../data/mbpp_critic.jsonl \
    --output-jsonl ../../data/mbpp_masking_trajectories.jsonl

# 4. preflight
MODEL=/path/to/models/TraDo-4B-Instruct
python rl_trado_critic.py \
    config=configs/rl_trado_critic_mbpp_linux.yaml \
    model.pretrained_model=$MODEL \
    model.value_base_model=$MODEL \
    experiment.preflight_only=true

# 5. smoke run (1 step)
export PRETRAINED_MODEL=$MODEL
export VALUE_BASE_MODEL=$MODEL
bash scripts/run_trado_critic_mbpp_smoke.sh configs/rl_trado_critic_mbpp_linux.yaml
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
