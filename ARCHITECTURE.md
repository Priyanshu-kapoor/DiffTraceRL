# dLLM-RL Engineering Handbook
**Complete Technical Architecture Document**

---

## 1. Executive Summary

dLLM-RL is a reinforcement learning training framework for **discrete diffusion language models (dLLMs)** — a class of language models that generate text by iteratively *unmasking* tokens rather than generating them left-to-right like GPT. The project's primary contribution is **TraceRL**, a trajectory-aware RL method that treats each unmasking step as an optimization unit, plus a novel **relativistic adversarial critic** (IRL component) that provides dense per-token reward signals by learning to distinguish expert unmasking trajectories from policy trajectories.

**Core functionality:** Given a code generation task (MBPP, HumanEval, CodeContests), the system (1) rolls out candidate solutions using the policy model, (2) trains a binary critic to score partial unmasking prefixes as expert-like or policy-like, (3) converts critic scores into dense per-token rewards, (4) trains a diffusion value model for variance reduction, and (5) updates the policy via a clipped policy-gradient objective. This alternating loop produces a code-generating dLLM that improves over training.

**High-level architecture:**
```
Config (OmegaConf YAML)
  → rl_trado_critic.py (orchestrator)
    → [each epoch]:
        vLLM rollout (JetEngine)
        → code execution + scalar reward
        → critic init + critic training (BCE discriminator)
        → dense token reward generation
        → value model training (TraceRL + GAE)
        → policy model training (TraceRL + KL + clipping)
        → evaluation
```

**Main technologies:** Python 3.10, PyTorch 2.6, HuggingFace Transformers 4.52, Accelerate + DeepSpeed ZeRO-3, vLLM/JetEngine for fast rollout, Flash Attention 2, PEFT/LoRA, OmegaConf, Weights & Biases.

---

## 2. Repository Tree Overview

```
dLLM-RL/
│
├── rl_trado_critic.py          [CRITICAL]   Main orchestrator for TraceRL+IRL pipeline
├── rl.py                       [CRITICAL]   Orchestrator for standard TraceRL (no critic)
├── rl_v.py                     [Important]  Orchestrator variant with value model only
├── multinode_rl.py             [Important]  Multi-node version of rl.py
├── multinode_rl_v.py           [Important]  Multi-node version of rl_v.py
├── eval.py                     [Important]  Standalone evaluation runner
├── eval_v.py                   [Important]  Evaluation with value model
├── multinode_eval.py           [Utility]    Multi-node evaluation
├── generate.py                 [Important]  block_diffusion_generate() — inference entry
├── trado_critic_utils.py       [CRITICAL]   Shared utilities for critic pipeline
│
├── models/                     [CRITICAL]   All model definitions
│   ├── __init__.py             [Important]  Exports SDARForCausalLM, LLaDAModelLM, etc.
│   ├── sdar/
│   │   ├── modeling_sdar.py    [CRITICAL]   SDARForCausalLM — the base model (Qwen3-arch)
│   │   └── configuration_sdar.py [CRITICAL] SDARConfig — all hyperparameters
│   ├── llada/                  [Important]  LLaDA model (full-attention dLLM)
│   ├── dream/                  [Important]  Dream model (Diffusion LM)
│   ├── mmada/                  [Important]  MMaDA multimodal model
│   ├── sampling.py             [Important]  Shared sampling utilities
│   └── lr_schedulers.py        [Important]  get_scheduler() — cosine/linear/polynomial
│
├── train/                      [CRITICAL]   All training scripts
│   ├── train_trado_policy.py   [CRITICAL]   Policy gradient update (TraceRL)
│   ├── train_trado_value.py    [CRITICAL]   Value model training (TraceRL + GAE)
│   ├── train_trado_critic.py   [CRITICAL]   Critic binary classifier training
│   ├── init_trado_value_model.py [CRITICAL] Creates value model checkpoint from base
│   ├── init_trado_critic_model.py [CRITICAL] Creates critic model checkpoint from base
│   ├── prompting_utils.py      [CRITICAL]   UniversalPrompting — tokenizes prompt+response
│   └── utils.py                [Important]  get_config(), AverageMeter, flatten_omega_conf
│
├── data/                       [CRITICAL]   Dataset preparation
│   ├── build_trado_expert_trajectory_cache.py [CRITICAL] GPT-2→Trado step-map alignment
│   ├── build_entropy_masking_trajectories.py  [Important] GPT-2 trajectory generation
│   ├── prepare_critic_code_data.py            [Important] MBPP/HumanEval normalization
│   ├── download_raw_code_dataset.py           [Important] Downloads raw datasets
│   └── download_data.py                       [Utility]   General dataset downloader
│
├── sample/                     [CRITICAL]   Rollout/sampling scripts
│   └── trado_rl_rollout.py     [CRITICAL]   vLLM/JetEngine rollout, multi-GPU dispatch
│
├── reward/                     [CRITICAL]   Reward computation
│   ├── rl_critic_reward.py     [CRITICAL]   Dense per-token reward from critic scores
│   ├── rl_code_reward.py       [Important]  Scalar code correctness reward (baseline)
│   ├── rl_execute.py           [Important]  Parallel code execution sandbox
│   └── rl_reward.py            [Utility]    Math reward (non-code tasks)
│
├── configs/                    [Important]  All OmegaConf YAML configs
│   ├── rl_trado_critic_mbpp_with_value.yaml   [CRITICAL] Main production config
│   ├── rl_trado_critic_mbpp_linux.yaml        [CRITICAL] Linux 1-2 GPU config (ours)
│   ├── rl_trado_with_value.yaml               [Important] TraceRL without critic
│   ├── rl_trado.yaml                          [Important] Minimal TraceRL
│   └── ...                     [Important]  Configs for LLaDA, Dream, SDAR variants
│
├── accelerate_configs/         [Important]  DeepSpeed/Accelerate YAML configs
│   ├── 1_node_1_gpu_deepspeed_zero3.yaml     [CRITICAL] Single GPU (ours)
│   ├── 1_node_2_gpus_deepspeed_zero3.yaml    [CRITICAL] Dual GPU (ours)
│   ├── 1_node_4_gpus_deepspeed_zero3.yaml    [Important] 4-GPU original default
│   └── ...                     Various multi-node configs
│
├── scripts/                    [Important]  Shell helpers
│   ├── run_trado_critic.sh             [Important] Main Linux launch wrapper
│   ├── run_trado_critic_mbpp_smoke.sh  [Important] Linux smoke test
│   ├── setup_linux_training_env.sh     [Important] Conda env setup
│   └── check_linux_training_env.py     [Utility]   Environment preflight check
│
├── smoke_data_windows/         [Important]  Windows smoke test data + config
│   ├── smoke_config.yaml                     Config for Windows testing
│   ├── test_mbpp_critic.jsonl               3 MBPP examples
│   └── test_mbpp_masking_trajectories.jsonl  Corresponding trajectories
│
├── test_critic_windows.py      [Important]  End-to-end CPU/Windows smoke test
├── README.md                   [Important]  Main project README
├── README_critic.md            [CRITICAL]   Critic pipeline setup guide
├── requirements.txt            [Important]  Python dependencies
└── ARCHITECTURE.md             [This file]
```

---

## 3. System Architecture

### Data Pipeline

```
Raw Datasets (HuggingFace / local JSONL)
  ↓  data/download_raw_code_dataset.py
  ↓  data/prepare_critic_code_data.py
Normalized JSONL  (data/mbpp_critic.jsonl)
  [fields: task_id, question, reference_code, test_list/test_input, test_method]

  ↓  data/build_entropy_masking_trajectories.py  [GPT-2 token-entropy ordering]
Expert Masking Trajectories  (data/mbpp_masking_trajectories.jsonl)
  [fields: task_id, trajectory=[{step, positions},...], response]

  ↓  data/build_trado_expert_trajectory_cache.py  [GPT-2→Trado token alignment]
Expert Cache  (<project>/temp_data/expert_trado_cache.pt)
  [fields per record: task_id, prompt, response, trado_input_ids, trado_step_map]
```

### Training Control Flow

```
rl_trado_critic.py::main()
  │
  ├─ run_preflight_checks()        validate paths, GPU count, model refs
  ├─ write_runtime_config()        merge YAML + CLI → temp resolved YAML
  ├─ ensure_repo_dataset()         materialize data/*.json for rollout scripts
  ├─ build_expert_cache()          [once] build expert_trado_cache.pt
  │
  └─ while epoch <= total_step:
       │
       ├─ sample/trado_rl_rollout.py      [subprocess, JetEngine vLLM]
       │    outputs: temp_data/outputs-*.json
       │
       ├─ reward/rl_execute.py            [subprocess, parallel code sandbox]
       │    outputs: same file + correctness field
       │
       ├─ train/init_trado_critic_model.py [subprocess, CPU or GPU]
       │    outputs: <project>/ckpt/optimized_critic/  (fresh backbone+head)
       │
       ├─ train/train_trado_critic.py     [subprocess, single GPU, no accelerate]
       │    inputs:  expert cache, policy rollout outputs, replay buffer
       │    outputs: updated optimized_critic checkpoint, updated replay buffer
       │
       ├─ reward/rl_critic_reward.py      [subprocess]
       │    inputs:  rollout outputs, optimized_critic checkpoint
       │    outputs: temp_data/<optimization_data>.json  [vector rewards]
       │
       ├─ train/train_trado_value.py      [accelerate launch, DeepSpeed ZeRO-3]
       │    inputs:  <optimization_data>.json, optimized_value checkpoint
       │    outputs: updated optimized_value checkpoint
       │              temp_data/<optimization_data>.pt  [fragment tensors for policy]
       │
       ├─ train/train_trado_policy.py     [accelerate launch, DeepSpeed ZeRO-3]
       │    inputs:  <optimization_data>.pt (from value training), optimized checkpoint
       │    outputs: updated optimized checkpoint
       │
       └─ [every eval_every epochs]
            sample/trado_rl_rollout.py → rl_execute.py → reward/rl_code_reward.py
            outputs: <project>/results/results-eval-*.txt
```

### External Dependencies

| Component | Used For | Alternative |
|---|---|---|
| JetEngine / vLLM | Fast parallel rollout generation | CPU generation (slow) |
| DeepSpeed ZeRO-3 | Sharding 4B model across GPUs | FSDP (not implemented) |
| Flash Attention 2 | Fast attention kernels | SDPA fallback |
| Liger Kernel | Fused SwiGLU / RMSNorm | PyTorch fallback |
| bitsandbytes | 4-bit quantization in smoke_test | Not used in production |
| PEFT/LoRA | LoRA adapters in smoke_test | Full fine-tuning in production |
| Weights & Biases | Training metrics logging | Disabled in smoke_test |
| OmegaConf | Hierarchical YAML config | — |

---

## 4. End-to-End Execution Flow

**Entry point:** `python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_linux.yaml`

```
1. get_config()
   → OmegaConf.from_cli() reads CLI overrides
   → OmegaConf.load(cli_conf.config) reads YAML
   → OmegaConf.merge() produces final config

2. main()
   → assert config.model.model_base == "trado"
   → assert config.dataset.data_type == "code"
   → run_preflight_checks(config)
       → visible_gpu_count() via torch.cuda.device_count()
       → select_deepspeed_config() picks matching accelerate YAML
       → validate_model_references() loads tokenizer to verify paths
       → checks normalized_data_path and expert_trajectory_path exist
   → write_runtime_config() → /tmp/rl_trado_critic_XXXXX/resolved_config.yaml
   → config.config = runtime_config_path  (all children reuse this)
   → ensure_repo_dataset()
       → load_jsonl(normalized_data_path)
       → writes data/{train_dataset}.json and data/{eval_dataset}.json

3. Epoch 1 (start_from_scratch=True):
   → initialize_run_artifacts(): mkdir project/{results,temp_data,ckpt}
   → init_value_model(epoch=1):
       subprocess: python train/init_trado_value_model.py config=...
       → _get_value_model(SDARForCausalLM) creates ValueModel subclass
       → load pretrained weights from config.model.value_base_model
       → zero-initialize value_head.weight
       → save_checkpoint() → <project>/ckpt/optimized_value/

4. sample_train(epoch=1):
   subprocess: python sample/trado_rl_rollout.py config=...
   → Loads data/{train_dataset}.json
   → visible_gpu_count() → splits tasks across GPU groups
   → For each GPU group, spawns _llm_worker_run() in a Process:
       → LLM(model_path, tensor_parallel_size=tp)
       → llm.generate_streaming(prompts, SamplingParams(...))
       → Returns [{text, step_map}, ...]
   → Collects results, restores original order
   → Writes: <project>/temp_data/outputs-rl-{model_ref}-{dataset}.json

5. execute_train(epoch=1):
   subprocess: python reward/rl_execute.py config=...
   → Reads rollout outputs JSON
   → Runs each code response in a sandbox (subprocess with timeout)
   → Adds "correctness": [[True/False per test], ...] to each entry
   → Writes updated JSON to same path

6. init_critic(epoch=1):
   subprocess: python train/init_trado_critic_model.py config=...
   → _get_critic_model(SDARForCausalLM) creates CriticModel subclass
   → Loads config.model.value_base_model (same base as value model)
   → Random-initialize critic_head.weight (NOT zero — needs gradient signal)
   → save_checkpoint() → <project>/ckpt/optimized_critic/

7. train_critic(epoch=1):
   subprocess: python train/train_trado_critic.py config=...
   → load_rollout_policy_samples() reads rollout outputs
   → update_replay_buffer(): append new policy samples, trim to buffer_size
   → load expert_trado_cache.pt
   → build_prefix_descriptors() for expert and policy
   → For num_steps_per_epoch steps:
       → Sample batch_size/2 expert + batch_size/2 policy descriptors
       → build_training_batch(): extended [L0+2*L1] format with visible tail mask
       → CriticModel.forward() → [batch_size] scalar logits
       → F.binary_cross_entropy_with_logits(logits, labels=[1,1,...,0,0,...])
       → loss.backward(); optimizer.step()
   → save_checkpoint() → <project>/ckpt/optimized_critic/

8. generate_critic_reward(epoch=1):
   subprocess: python reward/rl_critic_reward.py config=...
   → Reads rollout outputs
   → Tokenizes (prompt, response) pairs via UniversalPrompting
   → build_prefix_examples(): one example per unique step per response
   → Load CriticModel from optimized_critic checkpoint
   → score_examples(): forward pass → sigmoid(logits) ∈ [0,1]
   → For each response:
       sorted_scores = scores by ascending step_id
       token_rewards[j] = score[step_j] - score[step_{j-1}]  (delta)
   → Normalize within response groups (z-score across responses per task)
   → Scale token_rewards so sum = normalized_return
   → Write: <project>/temp_data/{optimization_data}.json

9. train_value(epoch=1):
   accelerate launch ... train/train_trado_value.py config=...
   → Loads {optimization_data}.json
   → UniversalPrompting: tokenize → input_ids [B, L0+L1], start_pos=L0
   → collect_training_data(): creates fragment rows
       For each sample b ∈ [0,B):
           step_map_b → run one_round_vectorized() per unmasking round
           each round → one fragment row: extended_input_ids [L0+2*L1], p_mask [L0+L1]
   → TrainDataset: stores all fragment rows + step_map_all + resp_input_ids_all
   → compute_old_value_parallel(): value model forward on all rows → old_values
   → compute_returns_and_advantages_from_fragments(): GAE
       For each sequence s:
           union all p_masks → V_resp[L1] from old_values
           GAE backward recursion → R_resp[L1], A_resp[L1]
   → save_dataset_tensors(): writes {optimization_data}.pt
   → Training loop (max 3 steps in smoke_test):
       loss = -mean(A * log p(token)) clipped by ε

10. train_policy(epoch=1):
    accelerate launch ... train/train_trado_policy.py config=...
    → torch.load({optimization_data}.pt) → fragment tensors + advantages
    → Load policy model (4-bit+LoRA in smoke_test, full bf16 in production)
    → compute_old_logprobs_parallel(): reference log-probs for KL term
    → Training loop (max 3 steps in smoke_test):
        new_logprobs = policy_forward(extended_input_ids)
        ratio = exp(new_logprobs - old_logprobs)
        clipped_ratio = clip(ratio, 1-ε, 1+ε)
        policy_loss = -mean(min(ratio*A, clipped_ratio*A))  [PPO-clip]
        kl_loss = β * mean(new_logprobs - old_logprobs)     [KL penalty]
        loss = policy_loss + kl_loss
    → save LoRA adapters (smoke_test) or full checkpoint (production)
```

---

## 5. Configuration System

### Config Loading Mechanism

All scripts use the same pattern:
```python
cli_conf = OmegaConf.from_cli()          # e.g. smoke_test=true
yaml_conf = OmegaConf.load(cli_conf.config)
conf = OmegaConf.merge(yaml_conf, cli_conf)  # CLI overrides YAML
```

CLI overrides use dot notation: `model.pretrained_model=Gen-Verse/TraDo-4B-Instruct`

### Complete Parameter Reference

| Parameter | Meaning | Used In | Effect |
|---|---|---|---|
| `experiment.project` | Output directory name | All | Creates `<project>/ckpt/`, `<project>/temp_data/`, `<project>/results/` |
| `experiment.total_step` | Number of training epochs | rl_trado_critic.py | Loop bound |
| `experiment.current_epoch` | Starting epoch (1 = fresh) | All | Resume from checkpoint |
| `experiment.start_from_scratch` | Init value model and dirs | rl_trado_critic.py | If False, skips init |
| `experiment.eval_every` | Epochs between evaluations | rl_trado_critic.py | Evaluation frequency |
| `experiment.save_every` | Epochs between extra snapshots | train_trado_value/policy | Snapshot frequency |
| `experiment.train_value_every` | Epochs between value training | train_trado_value.py | Can skip value training |
| `experiment.deepspeed_file` | Accelerate config name (or "auto") | rl_trado_critic.py, rl.py | Selects GPU sharding strategy |
| `experiment.num_node` | Number of machines | rl_trado_critic.py | Multi-node setup |
| `experiment.node_index` | This machine's rank | rl_trado_critic.py | Multi-node setup |
| `model.pretrained_model` | Path or HF ID of base model | train_trado_policy.py, rollout | Policy checkpoint |
| `model.value_base_model` | Path or HF ID for value/critic init | init scripts | Base for value/critic |
| `model.optimized_name` | Policy checkpoint subdir | train_trado_policy.py | Save location |
| `model.optimized_value_name` | Value checkpoint subdir | train_trado_value.py | Save location |
| `model.optimized_critic_name` | Critic checkpoint subdir | train_trado_critic.py | Save location |
| `model.model_base` | "trado", "llada", "dream", etc. | rl.py | Selects script family |
| `dataset.train_dataset` | Dataset JSON name (no extension) | rollout, execute | Reads data/{name}.json |
| `dataset.optimization_data` | RL reward data name (no extension) | value + policy train | Reads/writes temp_data/{name}.json/.pt |
| `dataset.data_type` | "code" or "math" | rl.py, rl_execute.py | Selects reward computation |
| `rollout.tensor_parallel_size` | GPUs per vLLM worker | trado_rl_rollout.py | Tensor parallelism in LLM |
| `rollout.num_task_per_step` | Tasks to roll out per epoch | trado_rl_rollout.py | Throughput |
| `rollout.num_response_per_task` | Responses per task | trado_rl_rollout.py | Diversity for reward normalization |
| `rollout.max_token` | Max generation length | trado_rl_rollout.py | Generation cap |
| `rollout.block_size` | Unmasking block size | trado_rl_rollout.py | Trado decoding parameter |
| `rollout.temperature` | Sampling temperature | trado_rl_rollout.py | Entropy of rollout |
| `training.block_size` | Block size for training attention | train_trado_value/policy | Must match rollout |
| `training.max_prompt_len` | Max prompt tokens | UniversalPrompting | Filters too-long prompts |
| `training.max_gen_length` | Max response tokens | UniversalPrompting | Pads/truncates responses |
| `training.batch_size_lm` | Per-GPU training batch size | train_trado_value/policy | VRAM usage |
| `training.gradient_accumulation_steps` | Grad accumulation | train_trado_value/policy | Effective batch size |
| `training.mixed_precision` | "bf16", "fp16", or "no" | Accelerator | Compute precision |
| `training.gradient_checkpointing_enable` | Recompute activations | train scripts | VRAM vs speed tradeoff |
| `training.eps` | PPO clip ratio ε | train_trado_policy.py | Policy update stability |
| `training.beta` | KL penalty coefficient | train_trado_policy.py | Deviation from reference |
| `training.gam` | GAE discount γ | train_trado_value.py | Return horizon |
| `training.lam` | GAE λ parameter | train_trado_value.py | Bias-variance tradeoff |
| `training.shrink` | Step collapse factor | train_trado_value.py | Reduces fragment count |
| `training.method` | "TraceRL", "random_masking", "coupled" | train_trado_policy.py | RL algorithm |
| `optimizer.params.policy_learning_rate` | LR for policy | train_trado_policy.py | — |
| `optimizer.params.value_learning_rate` | LR for value head | train_trado_value.py | — |
| `critic.batch_size` | Critic training batch size | train_trado_critic.py | — |
| `critic.learning_rate` | Critic optimizer LR | train_trado_critic.py | — |
| `critic.num_steps_per_epoch` | Critic gradient steps | train_trado_critic.py | Training thoroughness |
| `critic.buffer_size` | Max replay buffer entries | train_trado_critic.py | Memory of past policy |
| `critic.freeze_backbone` | Freeze SDAR backbone in critic | train_trado_critic.py | Speeds up critic training |
| `critic.pooling` | How to pool tokens for critic score | train_trado_critic.py | "visible_response_mean" |
| `critic.reward_mode` | How to turn critic scores to rewards | rl_critic_reward.py | "delta_sigmoid" |
| `smoke_test` | Enable smoke test mode | train_trado_policy/value.py | 4-bit/LoRA, 3-step limit |

---

## 6. Data Pipeline

### Stage 1: Raw → Normalized JSONL

```
HuggingFace datasets (google-research-datasets/mbpp, etc.)
  ↓ data/download_raw_code_dataset.py
  [raw format varies by dataset]
  ↓ data/prepare_critic_code_data.py
data/mbpp_critic.jsonl
  [each line: {
    "task_id": "Mbpp/1",
    "question": "Write a function to...",
    "reference_code": "def fn(...):\n  ...",
    "test_list": ["assert fn(1) == 2", ...],   # for function-type
    "test_input": ["1\n"], "test_output": ["2\n"],  # for stdio-type
    "test_method": "function" | "stdio",
    "prefix": "",          # Optional code prefix
    "test_time_limit": 3   # Seconds
  }]
```

### Stage 2: GPT-2 Masking Trajectories

```
data/mbpp_critic.jsonl  +  GPT-2 (via HuggingFace)
  ↓ data/build_entropy_masking_trajectories.py
data/mbpp_masking_trajectories.jsonl
  [each line: {
    "task_id": "Mbpp/1",
    "response": "def fn(...):\n  ...",
    "trajectory": [
      {"step": 0, "positions": [5, 12, 23]},  # Token indices revealed at step 0
      {"step": 1, "positions": [1, 8, 19]},
      ...
    ]
  }]
```
The trajectory orders token reveals by ascending entropy — lowest-entropy (most predictable) tokens revealed first.

### Stage 3: Expert Cache (GPT-2 → Trado token alignment)

```
data/mbpp_masking_trajectories.jsonl  +  TraDo-4B tokenizer
  ↓ data/build_trado_expert_trajectory_cache.py
<project>/temp_data/expert_trado_cache.pt
  {
    "records": [
      {
        "task_id": "Mbpp/1",
        "prompt": "<system>...<|im_start|>user\nWrite a function...<|im_end|>...",
        "response": "def fn(...):\n  ...",
        "trado_input_ids": [1234, 5678, ...],   # shape [L1]
        "trado_step_map": [0, 0, 1, 2, 1, ...], # shape [L1], step per token
        "test_list": [...],
        "test_method": "function"
      },
      ...
    ]
  }
```

### Stage 4: Rollout Output

```
data/{train_dataset}.json  +  Policy model
  ↓ sample/trado_rl_rollout.py  (vLLM/JetEngine)
<project>/temp_data/outputs-rl-{model_ref}-{dataset}.json
  [
    {
      "question": "...",
      "prompt": "<formatted prompt>",
      "full_output": ["def fn...", "def fn..."],    # num_response_per_task responses
      "step_map": [[0,0,1,2,...], [0,1,1,2,...]],  # per token, per response
      "extracted_output": ["def fn...", ...],
      "correctness": [[True, False, True], ...],   # after rl_execute.py
      "response_length": [128, 134, ...]
    },
    ...
  ]
```

### Stage 5: Critic Reward Output

```
<project>/temp_data/outputs-*.json  +  Critic model
  ↓ reward/rl_critic_reward.py
<project>/temp_data/{optimization_data}.json
  [
    {
      "prompt": "...",
      "response": "def fn(...):\n  ...",
      "step_map": [0, 0, 1, 2, 1, ...],    # [L1] step per response token
      "reward": [0.02, 0.02, 0.08, 0.15, 0.08, ...]  # [L1] scaled delta rewards
    },
    ...
  ]
```
`use_vector_reward = True` is detected because `reward` is a list, not a scalar.

### Stage 6: Value Training Input (Fragment Tensors)

```
{optimization_data}.json  +  UniversalPrompting
  ↓ train/train_trado_value.py::collect_training_data()
TrainDataset:
  extended_input_ids:  [N_rows, L0 + 2*L1]   # N_rows = sum of rounds per sample
  p_mask:              [N_rows, L0 + L1]      # True at positions updated this round
  tok_idx_ext:         [N_rows, L0 + 2*L1]   # Position IDs (accounting for padding)
  labels:              [N_rows, L0 + L1]
  seq_ids:             [N_rows]               # Which original sequence (0..B-1)
  step_map_all:        [B, L1]                # Original step map per sequence
  resp_input_ids_all:  [B, L1]                # Original response tokens
  per_token_reward_all:[B, L1]                # Per-token rewards from critic
  old_values:          [N_rows, L0+L1]        # Filled by compute_old_value_parallel()
  adv:                 [N_rows, L0+L1]        # Filled by compute_returns_and_advantages()
```

Saved as `{optimization_data}.pt` for policy training.

---

## 7. Model Architecture

### SDAR / TraDo Model (SDARForCausalLM)

This is a Qwen3-family transformer adapted for **discrete diffusion** (block-attention unmasking). The architecture is identical to a causal LM in terms of layers, but inference uses a non-autoregressive unmasking procedure.

```
SDARForCausalLM
  ├── model: SDARModel
  │   ├── embed_tokens: Embedding(vocab_size=151936, hidden_size=4096)
  │   ├── layers: ModuleList[32 × SDARDecoderLayer]
  │   │   ├── self_attn: SDARAttention
  │   │   │   ├── q_proj: Linear(4096, 4096, bias=False)
  │   │   │   ├── k_proj: Linear(4096, 1024, bias=False)   # GQA: 8 KV heads
  │   │   │   ├── v_proj: Linear(4096, 1024, bias=False)
  │   │   │   ├── o_proj: Linear(4096, 4096, bias=False)
  │   │   │   ├── q_norm: SDARRMSNorm(head_dim=128)        # Per-head QK norm
  │   │   │   └── k_norm: SDARRMSNorm(head_dim=128)
  │   │   ├── mlp: SDARMLP
  │   │   │   ├── gate_proj: Linear(4096, 22016, bias=False)
  │   │   │   ├── up_proj:   Linear(4096, 22016, bias=False)
  │   │   │   └── down_proj: Linear(22016, 4096, bias=False)
  │   │   ├── input_layernorm:    SDARRMSNorm(4096)
  │   │   └── post_attention_layernorm: SDARRMSNorm(4096)
  │   └── norm: SDARRMSNorm(4096)
  └── lm_head: Linear(4096, 151936, bias=False)            # Tied to embed_tokens
```

**TraDo-4B parameter breakdown:**
- Embedding: 151936 × 4096 = 622M
- Per layer: ~134M (attn ~67M + MLP ~67M)
- 32 layers: ~4.3B total

### Key Architectural Points

**Grouped Query Attention (GQA):** 32 query heads, 8 key-value heads. Each KV head is shared by 4 query heads. This reduces KV cache memory by 4×.

**Per-head QK normalization:** Unique to this architecture — RMSNorm applied to each query and key head separately before attention. Stabilizes training at large scale.

**RoPE (Rotary Position Embedding):** Applied to queries and keys. `rope_theta=10000.0`. Supports dynamic scaling for long sequences.

**Block Attention for Diffusion:** At inference time, the attention mask is structured so that tokens in block `k` of the response can attend to all prompt tokens and all previously revealed blocks. This is the "block-attention" in TraDo. The same block-attention pattern is used during training via `make_basic_block_attention()`.

**Extended Input Format `[L0 + 2*L1]`:**
This is the key innovation for training diffusion models:
```
[  prompt (L0)  |  response (L1)  |  masked response tail (L1)  ]
```
- The first response segment contains the full response tokens
- The second segment contains the response with tokens at step ≥ current_step masked with `[MASK]` token
- The model predicts what the masked tokens should be, using the full response as "ground truth context"
- The `p_mask` identifies which positions in the second segment are being predicted

### Value Model Extension

```python
class ValueModel(SDARForCausalLM):
    value_head: Linear(hidden_size=4096, 1, bias=False)  # zero-initialized
    
    def forward(...) -> Tensor[B, L0+L1]:
        last_hidden_state = self.model(...)     # [B, L0+2*L1, 4096]
        # Take first L0+L1 positions (original tokens, not masked tail)
        condensed = cat([hidden[:,:L0], hidden[:,L0+L1:]], dim=1)  # [B, L0+L1, 4096]
        return value_head(condensed).squeeze(-1)  # [B, L0+L1]
```

### Critic Model Extension

```python
class CriticModel(SDARForCausalLM):
    critic_head: Linear(hidden_size=4096, 1, bias=False)  # random-initialized
    
    def forward(..., visible_tail_mask: Tensor[B, L1]) -> Tensor[B]:
        hidden = self.model(input_ids[L0+2*L1], ...)  # [B, L0+2*L1, 4096]
        tail_hidden = hidden[:, L0+L1:, :]  # [B, L1, 4096] — the masked tail
        # Pool only visible (unmasked) response positions
        mask = visible_tail_mask.unsqueeze(-1)  # [B, L1, 1]
        pooled = (tail_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(1)  # [B, 4096]
        return critic_head(pooled).squeeze(-1)  # [B] — one score per prefix
```

---

## 8. Diffusion Process Logic

### What "Discrete Diffusion" Means Here

Unlike continuous diffusion (DDPM/DDIM for images), TraDo uses **absorbing-state discrete diffusion**: tokens are masked (absorbed into a `[MASK]` state) and then iteratively revealed. There is no Gaussian noise.

**Forward process (masking):** Given a complete response `x_0`, mask tokens in a specific order defined by `step_map`. At step `k`, all tokens with `step_map[j] >= k` are masked:
```
x_k[j] = MASK_TOKEN  if step_map[j] >= k
x_k[j] = x_0[j]      if step_map[j] < k
```
The step_map is determined by GPT-2 token entropy ordering for expert trajectories, or by the model's own decoding order during rollout.

**Reverse process (unmasking):** Starting from all-masked `x_K`, reveal tokens block by block. Each step, the model scores all masked positions and reveals those with highest confidence (or lowest entropy). This is implemented in `sample/trado_rl_rollout.py` via JetEngine.

**Block structure:** The response is divided into non-overlapping blocks of size `block_size` (e.g., 4 tokens). Within each block, all tokens with the same minimum step_id are revealed simultaneously. This allows parallel unmasking within a block.

### Training Objective

**Value Model Loss** (`train/train_trado_value.py`, around line 1000+):
```
L_value = (V_predicted - R_target)^2
```
where `R_target` comes from GAE (Generalized Advantage Estimation):
```
R_j = r_j + γ * R_{j+1}     (discounted return)
δ_t = r_t - V_t + γ * V_{t+1}  (TD residual)
A_t = δ_t + (γλ) * A_{t+1}    (GAE advantage)
```
These are computed at the *step level* (one value per unmasking step, not per token), then broadcast back to token level via `step_map`.

**Policy Loss** (`train/train_trado_policy.py`, around line 900+):
```
ratio_j   = exp(log π_new(x_j | x_{<j}) - log π_old(x_j | x_{<j}))
L_clip    = -min(ratio_j * A_j, clip(ratio_j, 1-ε, 1+ε) * A_j)
L_KL      = β * (log π_new - log π_old)
L_policy  = mean over p_mask positions of (L_clip + L_KL)
```
where `p_mask` selects the specific tokens being trained in each fragment row.

**Critic Loss** (`train/train_trado_critic.py`):
```
L_critic = BCE(σ(critic_score), label)
label = 1.0  for expert prefix
label = 0.0  for policy prefix
```

**Reward Generation** (`reward/rl_critic_reward.py`):
```
score_k    = σ(critic(prefix_up_to_step_k))  ∈ [0,1]
delta_k    = score_k - score_{k-1}           (incremental improvement)
reward_j   = delta_{step_map[j]}             (token j gets its step's delta)
```
Then within each task's response group (num_response_per_task responses):
```
return_i   = sum_j(reward_j)                 (total return for response i)
z_return_i = (return_i - mean) / std          (normalize across group)
scale_i    = z_return_i / return_i
reward_j_scaled = reward_j * scale_i         (token rewards scaled to sum = z_return)
```

---

## 9. Training Pipeline

### Value Training Deep Dive

`train/train_trado_value.py::main()` — runs under `accelerate launch` with DeepSpeed ZeRO-3.

```
1. Load {optimization_data}.json
2. UniversalPrompting → input_ids [B, L0+L1], start_pos=L0
3. Build step_map_list [B, L1] and reward_scalar_list [B, L1]
4. apply shrink: collapse every `shrink` adjacent steps → fewer fragment rows
5. make_basic_block_attention(L0+2*L1, start_pos, block_size) → [1,1,N,N]
   (precomputed once, reused for all batches)
6. collect_training_data():
   for each sample b in [0,B):
       step_b = step_map[b].clone()  # will be modified in-place (set to -1 when selected)
       round r = 0:
           one_round_vectorized():
               for each block bi in [0, NB):
                   min_step = min(step_b[bi*block_size:(bi+1)*block_size] where >= 0)
                   selected positions = those with step_b == min_step in this block
               extended_input_ids = [full_response | response_with_unselected_masked]
               p_mask = True at selected positions in L0:L0+L1 range
           append (extended_input_ids, p_mask, ...) as row
           set step_b[selected] = -1
       round r+1: repeat until step_b all -1
   → N_rows fragment rows total across all B samples
7. TrainDataset created with all N_rows rows
8. compute_old_value_parallel():
   for each batch from dataloader:
       values = value_model(extended_input_ids, attention_mask, tok_idx_ext)  [B, L0+2*L1]
       values_condensed = cat(values[:,:L0], values[:,L0+L1:])  [B, L0+L1]
       old_values[ids] = values_condensed * p_mask   # store, zero elsewhere
9. compute_returns_and_advantages_from_fragments(): fills dataset.Return, dataset.adv
10. save_dataset_tensors() → {optimization_data}.pt
11. Training loop:
    for batch in dataloader:
        values = value_model(...)  [B, L0+L1]  (new forward pass)
        targets = dataset.Return[ids]           [B, L0+L1]
        loss = mean((values - targets)^2 * p_mask) / p_mask.sum()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
    if step >= 3 and _SMOKE_TEST: break
12. save checkpoint → optimized_value/
```

### Policy Training Deep Dive

`train/train_trado_policy.py::main()` — also runs under `accelerate launch`.

```
1. Load {optimization_data}.pt (saved by value training)
   → extended_input_ids, p_mask, tok_idx_ext, labels, adv, start_pos
2. In smoke_test: 4-bit + LoRA (only LoRA params are trained)
   In production: full bf16 model, all params trained
3. compute_old_logprobs_parallel():
   for each batch:
       logits = policy_model(extended_input_ids, ...)  [B, L0+2*L1, vocab]
       logits_condensed = cat(logits[:,:L0], logits[:,L0+L1:])  [B, L0+L1, vocab]
       log_probs = log_softmax(logits)
       old_logprobs = log_probs.gather(-1, labels)  [B, L0+L1]  # per-token log prob
4. Training loop:
    for batch in dataloader:
        new_logits = policy_model(extended_input_ids)
        new_logprobs = log_softmax(new_logits).gather(-1, labels)
        ratio = (new_logprobs - old_logprobs).exp()
        advantages = adv[ids]
        # PPO clip
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-eps, 1+eps) * advantages
        policy_loss = -min(surr1, surr2)[p_mask].mean()
        # KL penalty
        kl = (new_logprobs - old_logprobs)[p_mask].mean()
        loss = policy_loss + beta * kl
        accelerator.backward(loss)
        grad_clip(); optimizer.step(); lr_scheduler.step()
    if step >= 3 and _SMOKE_TEST: break
5. save LoRA adapters (smoke_test) or full model (production)
```

### Critic Training Deep Dive

`train/train_trado_critic.py::main()` — runs as plain Python, no accelerate.

```
1. load_rollout_policy_samples(): read rollout outputs JSON
   → list of {task_id, prompt, response, step_map} per (task, response) pair
2. update_replay_buffer():
   load existing buffer from critic.buffer_path (or empty)
   append new policy_samples
   trim to critic.buffer_size (FIFO)
   save buffer
3. load expert_trado_cache.pt → list of expert records
4. build_prefix_descriptors(expert_records):
   for each expert record:
       for each unique step k in trado_step_map:
           yield PrefixDescriptor(task_id, prompt, response, step_map, step=k)
   → O(L1 × N_expert) descriptors
5. build_prefix_descriptors(replay_buffer) similarly
6. Load CriticModel from optimized_critic checkpoint
   if critic.freeze_backbone: freeze all except critic_head
7. AdamW on critic_head params only (or all if not frozen)
8. for step in range(critic.num_steps_per_epoch):
   sample half_batch expert + half_batch policy descriptors (random)
   labels = [1.0] * half_batch + [0.0] * half_batch
   batch = build_training_batch(descriptors, tokenizer, ...):
       for each descriptor:
           build extended format [L0+2*L1] with tail masked up to step k
           visible_tail_mask = (step_map <= k)
   logits = critic_model(batch.extended_input_ids,
                         batch.attention_mask,
                         batch.tok_idx_ext,
                         batch.visible_tail_mask)  # [batch_size]
   loss = F.binary_cross_entropy_with_logits(logits, labels_tensor)
   loss.backward(); optimizer.step(); optimizer.zero_grad()
9. save_checkpoint(critic_model, ...) → optimized_critic/
```

---

## 10. Inference / Sampling Pipeline

`sample/trado_rl_rollout.py` — called as subprocess by `rl_trado_critic.py`.

```
1. Load config, read data/{train_dataset}.json
2. Determine visible GPU count, split into groups of tensor_parallel_size GPUs
3. Shuffle all tasks, assign chunks to groups
4. For each GPU group, spawn Process(_llm_worker_run, gpu_list, tasks):
   a. Set CUDA_VISIBLE_DEVICES=",".join(gpu_list)
   b. Create LLM(model_path, tensor_parallel_size=len(gpu_list))
   c. Create SamplingParams(temperature, top_k, top_p, max_tokens,
                            remasking_strategy, block_length, denoising_steps)
   d. outputs = llm.generate_streaming(prompts, sp, max_active=max_active)
   e. For each output:
      - text = output.text
      - step_map = output.first_unmask_times  (per-token reveal order)
      - Extract code from markdown block if present
   f. queue.put(results_for_my_chunk)
5. Parent collects from all queues, restores original task order
6. Write JSON to rollout_outputs_path(config, "train")
```

**JetEngine / vLLM** performs speculative block-attention decoding. The `block_length` and `denoising_steps` parameters control how many tokens per step and how many denoising passes per block.

---

## 11. Loss Functions

### Policy Loss (PPO-Clip + KL)
**Location:** `train/train_trado_policy.py`, training loop  
**Formula:**
```
L = -E_{j∈p_mask}[min(r_j * A_j, clip(r_j, 1-ε, 1+ε) * A_j)] + β * KL(π_new || π_old)
r_j = π_new(x_j) / π_old(x_j)  (probability ratio)
```
**Why:** PPO-clip prevents large policy updates that could destabilize training. The KL term provides an additional soft constraint keeping the new policy close to the reference.

### Value Loss (MSE)
**Location:** `train/train_trado_value.py`, training loop  
**Formula:**
```
L = E_{j∈p_mask}[(V_θ(x_j) - R_j)^2]
```
where `R_j` is the GAE return. **Why:** Minimize TD error to get accurate state value estimates that reduce variance in advantage estimation.

### Critic Loss (Binary Cross-Entropy)
**Location:** `train/train_trado_critic.py`  
**Formula:**
```
L = -[y * log(σ(f(x))) + (1-y) * log(1 - σ(f(x)))]
y = 1 for expert prefixes, 0 for policy prefixes
```
**Why:** Train a discriminator that assigns high scores to "expert-like" partial responses and low scores to "policy-like" ones. The score differential `σ(f_k) - σ(f_{k-1})` becomes the per-step reward.

---

## 12. Scheduler / LR Schedule

**Location:** `models/lr_schedulers.py`

Default: cosine decay with warmup.

```
t = current_step / total_steps  (progress ∈ [0,1])
lr(t) = lr_base × [min_lr_scale + (1 - min_lr_scale) × 0.5 × (1 + cos(π × t))]
```

With `min_lr_scale=1.0` (smoke_test config default): LR stays constant (no decay).  
With `min_lr_scale=0.0` (standard): LR decays to zero at end of training.  
With `warmup_steps > 0`: LR linearly ramps from 0 to lr_base over first `warmup_steps`.

**Config:** `lr_scheduler.scheduler: "cosine"`, `lr_scheduler.params.warmup_steps: 0`, `lr_scheduler.params.min_lr_scale: 1.0`

---

## 13. Utilities and Helper Functions

### `trado_critic_utils.py`

| Function | Purpose |
|---|---|
| `load_jsonl(path)` | Read a JSONL file → list of dicts |
| `write_jsonl(path, records)` | Write list of dicts → JSONL |
| `rollout_outputs_path(config, fn)` | Compute path for rollout output JSON |
| `build_prefix_examples(...)` | Build one example per reveal step per response |
| `make_basic_block_attention(...)` | Build Trado block-attention mask for [L0+2*L1] |
| `score_examples(model, examples, ...)` | Batch-forward critic on prefix examples |

### `train/utils.py`

| Function | Purpose |
|---|---|
| `get_config()` | OmegaConf CLI+YAML merge |
| `AverageMeter` | Running average tracker for loss logging |
| `flatten_omega_conf(conf)` | Convert OmegaConf to flat dict for W&B |

### `train/prompting_utils.py` — `UniversalPrompting`

Central tokenization class. Handles:
- Left-padding prompts (so all prompts end at the same position)
- Right-padding responses (so all responses start at the same position)
- Tracking `start_pos` (prompt length) and `drop_num` (filtered samples)
- System prompt construction for Trado's chat template

Input: `(list[str] prompts, list[str] responses)`  
Output: `(input_ids [B, L0+L1], label_ids [B, L0+L1], start_pos int, drop_num int)`

---

## 14. Class and Function Reference

### Critical Chain: `rl_trado_critic.py`

| Function | Calls | Called By | Side Effects |
|---|---|---|---|
| `main()` | `run_preflight_checks`, `write_runtime_config`, epoch loop functions | `__main__` | Creates project dirs, writes runtime config |
| `select_deepspeed_config(config)` | `visible_gpu_count`, `OmegaConf.load` | `run_preflight_checks`, `run_accelerate` | None |
| `run_accelerate(config, script, **overrides)` | `select_deepspeed_config`, `subprocess.run` | `train_value`, `train_policy` | Launches accelerate subprocess |
| `run_python(config, subdir, script, **overrides)` | `subprocess.run` | Most other step functions | Launches Python subprocess |
| `build_expert_cache(config, epoch)` | `run_python` | `main()` once | Writes expert_trado_cache.pt |
| `train_critic(config, epoch)` | `run_python` | epoch loop | Updates optimized_critic checkpoint |
| `generate_critic_reward(config, epoch)` | `run_python` | epoch loop | Writes optimization_data.json |

### Critical Chain: `train/train_trado_value.py`

| Function | Calls | Key Output |
|---|---|---|
| `main()` | Everything below | Full training run |
| `collect_training_data(input_ids, step_map_list, rewards)` | `one_round_vectorized` × many | Fragment rows as tensors |
| `one_round_vectorized(input_ids_b, step_map_b, ...)` | Pure tensor ops | One fragment row per call |
| `make_basic_block_attention(N, start_pos, block_size)` | Pure tensor ops | [1,1,N,N] attention bias |
| `compute_old_value_parallel(...)` | `value_model.forward` | Fills `dataset.old_values` |
| `compute_returns_and_advantages_from_fragments(dataset, γ, λ)` | Pure tensor ops | Fills `dataset.Return`, `dataset.adv` |
| `save_dataset_tensors(dataset, save_dir, name, ...)` | `torch.save` | Writes `.pt` file for policy |

### Critical Chain: `data/build_trado_expert_trajectory_cache.py`

| Function | Purpose |
|---|---|
| `dense_step_map_from_trajectory(traj, n_tokens)` | Sparse → dense step_id array |
| `build_char_step_map(text, offsets, token_step_map)` | Token steps → character steps |
| `trado_steps_for_formatted_response(...)` | Tries multiple code formats, aligns GPT-2/Trado |
| `build_expert_cache_entry(record, traj, tokenizer)` | One record → cache entry dict |

---

## 15. Dependency Graph

```
rl_trado_critic.py  [ORCHESTRATOR]
  ├── trado_critic_utils.py
  │     ├── torch, transformers
  ├── rl.py (parallel standard pipeline)
  │
  ├── sample/trado_rl_rollout.py
  │     ├── JetEngine/vLLM
  │     └── models/ (loaded via HF)
  │
  ├── reward/rl_execute.py
  │
  ├── train/init_trado_critic_model.py
  │     ├── models/sdar/modeling_sdar.py → SDARForCausalLM
  │     └── train/utils.py → get_config()
  │
  ├── train/train_trado_critic.py
  │     ├── models/sdar/modeling_sdar.py
  │     ├── trado_critic_utils.py
  │     └── transformers.AutoTokenizer
  │
  ├── reward/rl_critic_reward.py
  │     ├── models/sdar/modeling_sdar.py
  │     ├── trado_critic_utils.py
  │     └── train/prompting_utils.py → UniversalPrompting
  │
  ├── train/train_trado_value.py   [accelerate]
  │     ├── models/sdar/modeling_sdar.py
  │     ├── train/init_sdar_value_model.py → _get_value_model()
  │     ├── train/prompting_utils.py
  │     ├── models/lr_schedulers.py
  │     └── accelerate + deepspeed
  │
  └── train/train_trado_policy.py  [accelerate]
        ├── models/sdar/modeling_sdar.py
        ├── train/prompting_utils.py
        ├── models/lr_schedulers.py
        ├── peft (LoRA in smoke_test)
        ├── bitsandbytes (4-bit in smoke_test)
        └── accelerate + deepspeed
```

---

## 16. Checkpoint Format

### Policy / Value / Critic Checkpoints

All saved using `model.save_pretrained(save_dir, safe_serialization=True)`:
```
<project>/ckpt/optimized/
  ├── config.json              # Model config (SDARConfig)
  ├── model.safetensors        # All weights (or shards model-00001-of-N.safetensors)
  ├── tokenizer.json
  ├── tokenizer_config.json
  ├── special_tokens_map.json
  ├── modeling_sdar.py         # Copied for trust_remote_code
  └── configuration_sdar.py   # Copied for trust_remote_code
```

For the **value model**: the checkpoint includes `value_head.weight` alongside standard model weights. When loading, `_get_value_model()` must be used (not plain `AutoModelForCausalLM`) because it defines the `ValueModel` subclass.

For the **critic model**: same, but `critic_head.weight` instead.

For **LoRA checkpoints** (smoke_test policy):
```
<project>/ckpt/optimized/
  ├── adapter_config.json      # LoRA config (r=8, alpha=16, target_modules=...)
  └── adapter_model.safetensors  # Only LoRA delta weights (~64KB vs ~8GB full)
```

### Replay Buffer
```
<project>/temp_data/critic_policy_buffer.pt
  {
    "samples": [
      {"task_id": "Mbpp/1", "prompt": "...", "response": "...", "step_map": [...]},
      ...
    ]
  }
```

### Expert Cache
```
<project>/temp_data/expert_trado_cache.pt
  {
    "records": [
      {
        "task_id": str, "prompt": str, "response": str,
        "trado_input_ids": List[int],  "trado_step_map": List[int],
        "test_list": List[str], "test_method": str
      },
      ...
    ]
  }
```

---

## 17. Logging and Outputs

### Weights & Biases
- Enabled in production via `config.wandb.entity` (set to your W&B org)
- Disabled in smoke_test via `os.environ["WANDB_MODE"] = "disabled"`
- Logs: loss per step, epoch, learning rate
- Run name = `config.experiment.project`

### File Outputs Per Run

```
<project>/
  results/
    results-rl-{model_ref}-{train_dataset}.txt    # Value training losses
    results-rl-{policy_ref}-{train_dataset}.txt   # Policy training losses
    results-eval-{model_ref}-{eval_dataset}.txt   # Evaluation code correctness
  temp_data/
    outputs-rl-{model_ref}-{dataset}.json         # Rollout + execution results
    {optimization_data}.json                       # Critic rewards (vector)
    {optimization_data}.pt                         # Fragment tensors (value→policy)
    expert_trado_cache.pt                          # Expert cache (built once)
    critic_policy_buffer.pt                        # Replay buffer (grows over epochs)
  ckpt/
    optimized/                   # Current best policy
    optimized_value/             # Current best value model
    optimized_critic/            # Current best critic
    epoch-10-value/              # Snapshot at epoch 10 (save_every=10)
    epoch-20-value/
    ...
```

---

## 18. Key Design Decisions

### Why Block Attention?
TraDo generates text in blocks of `block_size` tokens simultaneously. Block attention means each block can see all previous blocks but not future ones. This enables parallel generation within a block, unlike pure autoregressive generation. Training uses the same attention mask as inference for consistency.

### Why Extended Format `[L0 + 2*L1]`?
The model needs to (a) see the full response to compute log-probabilities for training, but (b) also see the partially-masked response to simulate the inference condition. The double-response format puts both views in a single forward pass, avoiding the need for two separate forward passes.

### Why Fragment Rows (One Row Per Unmasking Round)?
TraceRL's core insight: the policy's probability for a token depends on which tokens were revealed before it (the unmasking prefix). Standard PPO batches all tokens together and ignores this dependency. Fragment rows isolate each unmasking round, so each training row captures the exact context under which those tokens were predicted.

### Why a Relativistic Critic?
Standard RLVR (RL with Verified Rewards) gives a binary 0/1 reward only at the end of a full response. This is sparse and noisy. The critic provides:
- **Dense** rewards: one per token, not per response
- **Comparative** scoring: expert-like trajectory = high reward, policy-like = low
- **Continuous** signal: `delta_sigmoid` gives gradient-friendly rewards in [0,1]

### Why Re-Initialize Critic Head Each Epoch?
The critic backbone is frozen (`freeze_backbone=True`), so only the 4096×1 linear head changes. Re-initializing it each epoch prevents the head from overfitting to old policy distributions. The replay buffer provides continuity — the critic always sees both current and past policy samples.

### Why ZeRO-3 + CPU Offload?
For a 4B parameter model in bf16:
- Weights: ~8 GB
- Gradients: ~8 GB
- Adam states (m, v): ~16 GB
- Total: ~32 GB — exceeds a single 24 GB GPU

ZeRO-3 shards all three across GPUs. CPU offload moves optimizer states to RAM (which is plentiful). On a single TITAN RTX (24 GB), ZeRO-3 + CPU offload makes 4B parameter training feasible.

### Why OmegaConf / YAML?
CLI overrides are essential for smoke tests and experiments (`smoke_test=true`, `model.pretrained_model=...`). OmegaConf's merge semantics (CLI wins over YAML) make it natural. The `write_runtime_config()` pattern ensures all child subprocesses see the same resolved config.

---

## 19. Extension Points

### Change Model Architecture
**File:** `models/sdar/modeling_sdar.py`  
Add layers, change attention type, or modify MLP. The `ValueModel` and `CriticModel` subclasses in `train/init_*_model.py` will inherit your changes automatically since they call `super().__init__()`.

### Add New Dataset
1. Add normalization script in `data/` that outputs `data/{name}_critic.jsonl` format
2. Build masking trajectories: `python data/build_entropy_masking_trajectories.py --input-jsonl ...`
3. Create config with `dataset.train_dataset: {name}` and `critic.normalized_data_path`

### Add New Model Family (e.g., LLaDA 2.0)
1. Add model class in `models/llada2/` (subclass `PretrainedModel`)
2. Register in `models/__init__.py`
3. Add `model_base: "llada2"` branch in `rl.py::train()` and `rl_v.py`
4. Add rollout script `sample/llada2_rl_rollout.py` following `trado_rl_rollout.py`
5. Add training scripts `train/train_llada2_policy.py` and `train_llada2_value.py`
6. Add configs in `configs/`

### Replace Loss Function
**Policy:** `train/train_trado_policy.py`, training loop — replace `min(surr1, surr2)` block.  
**Value:** `train/train_trado_value.py`, training loop — replace MSE term.  
**Critic:** `train/train_trado_critic.py` — replace `F.binary_cross_entropy_with_logits`.

### Change Reward Mode
**File:** `reward/rl_critic_reward.py`  
`critic.reward_mode: "delta_sigmoid"` is the current mode. Add new branches in the reward computation loop. The output must be a list of floats (one per response token) for `use_vector_reward=True` to trigger in value training.

### Add New Sampler Strategy
**File:** `sample/trado_rl_rollout.py`  
Add new `remasking_strategy` value to `SamplingParams`. This is passed through to JetEngine/vLLM which implements the actual sampling logic.

### Change Number of GPUs
No code changes needed. Set `experiment.deepspeed_file: auto` in the config and the launcher picks the right accelerate config based on `torch.cuda.device_count()`.

---

## 20. Known Risks and Technical Debt

### Hard-Coded Assumptions
- `SYSTEM_PROMPT_LEN = 28` in `train/init_trado_value_model.py` — assumes Trado's system prompt is exactly 28 tokens. If the system prompt changes, this constant breaks.
- Block attention construction assumes `L0 + 2*L1` format throughout. Any deviation in prompting format would require updating multiple places.
- Tokenizer special tokens (`mask_token_id`, `pad_token_id`) are assumed present. Fails silently with wrong model.

### Fragile Areas
- **`trado_steps_for_formatted_response`** in the expert cache builder: the GPT-2→Trado alignment depends on exact string matching of code candidates. Small formatting differences (trailing newlines, spacing) can cause silent misalignment.
- **`rollout_outputs_path`** filename construction: the path includes `model_ref = pretrained_model.replace("/", ".")`. On Windows, this can produce filenames that are too long.
- **Replay buffer**: Simple FIFO list stored as a `.pt` file. No deduplication, no priority sampling. May accumulate stale samples from early training when the policy was bad.

### Potential Bugs
- In `train_trado_value.py`, the `per_seq_reward` argument to `TrainDataset` is set to `torch.as_tensor(reward_scalar_list)`. When rewards are vectors (critic mode), this is a 2D tensor but the docstring says `(B,)` scalar. The code only uses `per_token_reward_all` in vector mode, so this doesn't crash, but it's confusing.
- The critic is re-initialized every epoch by calling `init_critic(config, epoch)` inside the epoch loop. If training is interrupted and resumed mid-epoch, the critic checkpoint from the aborted epoch may be partially trained.

### Performance Bottlenecks
- Expert cache building runs on CPU and can be slow for large datasets (CodeContests with 1000 examples takes minutes). Run once and cache.
- Critic training runs without DeepSpeed (single GPU, plain PyTorch). For very large backbones or long sequences, this is slower than policy/value training.
- The fragment row construction in `collect_training_data()` is CPU-bound Python with nested loops. For B=64 samples with 100 tokens each, this creates ~6400 rows.

---

## 21. Suggested Refactoring Opportunities

1. **Unify `_get_value_model` / `_get_critic_model`**: Both factories use the same pattern (add a Linear head). A single `_add_scalar_head(base_class, head_name, init_zero=False)` factory would eliminate duplication.

2. **Config validation dataclass**: Replace fragile `OmegaConf.select(..., default=None)` checks scattered across scripts with a single Pydantic or dataclass schema that validates the config at load time.

3. **Fragment row construction in pure PyTorch**: `collect_training_data()`'s inner loop over samples and rounds is Python-level. Rewriting with batched tensor operations would speed up the data preparation step.

4. **Centralize path construction**: `rollout_outputs_path`, checkpoint paths, and temp data paths are computed in multiple places. A single `ProjectPaths(config)` dataclass would make path changes a single-file edit.

5. **Decouple smoke_test flag**: The `_SMOKE_TEST` logic inside production training scripts mixes concerns. Separate smoke-test wrappers that call the production scripts would be cleaner.

---

## 22. Debugging Guide

### Shape Mismatch in Value Training
**Symptom:** `RuntimeError: The size of tensor a (X) must match...`  
**Cause 1:** `block_size` mismatch between rollout config and training config. The rollout's `step_map` has steps sized for `rollout.block_size` but training constructs attention with `training.block_size`. These must match.  
**Cause 2:** `max_gen_length` in config is smaller than the actual response in the JSON. Responses are silently truncated in `UniversalPrompting`.  
**Where to check:** `L1 = input_ids_lm.shape[1] - start_pos` in `train_trado_value.py` line ~337.

### Assertion Error: "some trainable tokens lack value predictions"
**Symptom:** `AssertionError: sequence 0: some trainable tokens lack value predictions`  
**Cause:** The fragment rows for a sequence don't collectively cover all non-padding response positions. Happens when the `step_map` has positions with values outside `[0, max_step]` (e.g., -1 set incorrectly).  
**Where to check:** `compute_returns_and_advantages_from_fragments()` line ~846. Inspect `dataset.step_map_all[0]` and `union_mask_resp` for sequence 0.

### NaN Loss in Value Training
**Symptom:** `loss = nan` after first few steps  
**Cause 1:** Value head initialized to zero → all predictions are 0 → GAE targets might have extreme values.  
**Cause 2:** Rewards contain NaN (from critic scoring). Check `{optimization_data}.json` for NaN entries.  
**Cause 3:** With the tiny smoke model, random weights + random rewards naturally produce NaN. Not a bug in smoke_test.  
**Where to check:** `loss_list` in `train_trado_value.py`, and the reward JSON for NaN.

### FileNotFoundError: `test_mbpp_rl_data.pt`
**Symptom:** Policy training crashes looking for `.pt` file  
**Cause:** Value training crashed before `save_dataset_tensors()` ran (line ~969).  
**Fix:** Always run value training before policy training. Check value training logs for the earlier error.

### HFValidationError: Invalid model ID
**Symptom:** `HFValidationError: ... is not a valid model identifier`  
**Cause:** Windows backslashes in model path. The HuggingFace validator requires forward slashes.  
**Fix:** Ensure model paths use forward slashes: `C:/Users/.../model` not `C:\Users\...\model`.

### OOM During Training
**Symptom:** `CUDA out of memory`  
**Cause 1:** `mixed_precision: "no"` uses fp32 → 2× memory. Change to `"bf16"`.  
**Cause 2:** `gradient_checkpointing_enable: false` → full activation memory.  
**Cause 3:** DeepSpeed config file doesn't match GPU count → ZeRO-3 sharding not applied.  
**Fix:** Set `mixed_precision: "bf16"`, `gradient_checkpointing_enable: true`, and ensure `deepspeed_file: auto`.

### Wrong GPU Count Detected
**Symptom:** Auto-selection picks wrong accelerate config  
**Cause:** `CUDA_VISIBLE_DEVICES` is set in the environment, masking some GPUs.  
**Fix:** `visible_gpu_count()` in `rl_trado_critic.py` checks `CUDA_VISIBLE_DEVICES` first. Unset it or set it explicitly.

---

## 23. Minimal Mental Model

- **What it is:** RL training for a block-attention discrete diffusion LLM (TraDo/SDAR) on code tasks.
- **Core loop:** Roll out responses → score with adversarial critic → compute dense rewards → update value model → update policy model.
- **Key data structure:** The "extended format" `[prompt | full_response | masked_tail]` is used everywhere — rollout, critic scoring, value/policy training.
- **Fragment rows:** Value training decomposes each response into N rounds of unmasking. Each round is one row in the training batch. Policy loss is computed only at the positions active in each round.
- **Critic = IRL:** A binary classifier (expert=1, policy=0) trained on partial unmasking prefixes. Its delta outputs become token-level rewards.
- **The model is Qwen3-architecture:** 32 layers, GQA (32Q/8KV heads), SwiGLU MLP, RMSNorm, RoPE. Code lives in `models/sdar/`.
- **Checkpoints are in `<project>/ckpt/`**, rollout data in `<project>/temp_data/`, results in `<project>/results/`.
- **To run:** Requires Linux + NVIDIA GPU + conda env + TraDo-4B weights from HuggingFace.

---

## 24. FAQ for Future Modifications

**Q: Where do I change the generation length?**  
`rollout.max_token` in config (controls vLLM generation) and `training.max_gen_length` (controls tokenizer padding). Both must be consistent.

**Q: How do I add a second dataset (e.g., HumanEval) alongside MBPP?**  
Create a new config file based on `rl_trado_critic_mbpp_linux.yaml`, change `dataset.train_dataset`, `dataset.optimization_data`, `critic.normalized_data_path`, and `critic.expert_trajectory_path`. Run the data preparation steps for HumanEval.

**Q: How do I swap to LLaDA 2.0?**  
Change `model.model_base: "llada"` and set `model.pretrained_model` to the LLaDA 2.0 HF path. The existing `models/llada/` code handles LLaDA. Note: TraceRL training scripts are currently Trado-specific; you'd need `train_llada_policy.py` analogues.

**Q: How do I increase the effective batch size without more GPUs?**  
Increase `training.gradient_accumulation_steps`. Effective batch = `batch_size_lm × num_gpus × gradient_accumulation_steps`.

**Q: How do I freeze the policy backbone and only train a LoRA adapter in production (not just smoke_test)?**  
Move the LoRA/4-bit loading block from the `if _SMOKE_TEST:` branch to the main branch in `train/train_trado_policy.py`. Adjust `r` and `lora_alpha` for production scale.

**Q: How do I resume training from epoch 15?**  
Set `experiment.start_from_scratch: false` and `experiment.current_epoch: 15` in the config (or as CLI override). The optimizer state is not saved separately — training resumes from the checkpoint weights with fresh optimizer state.

**Q: How do I make the critic stronger?**  
Increase `critic.num_steps_per_epoch` (more gradient steps per epoch), increase `critic.buffer_size` (more historical samples), or set `critic.freeze_backbone: false` (fine-tune the full backbone, much slower).

**Q: Where do I add a new reward signal (e.g., style rewards)?**  
Add a new reward script in `reward/`. The output must match the `{optimization_data}.json` format with a `"reward"` field that is either a list (vector, per-token) or a scalar. Then call it from the epoch loop in `rl_trado_critic.py`.

---

## 25. Code Navigation Cheat Sheet

| Task | Files to Inspect |
|---|---|
| Change model architecture | `models/sdar/modeling_sdar.py`, `models/sdar/configuration_sdar.py` |
| Change RL algorithm (PPO→GRPO etc.) | `train/train_trado_policy.py` (training loop ~line 900) |
| Change value estimation (GAE→TD(0)) | `train/train_trado_value.py::compute_returns_and_advantages_from_fragments` |
| Change critic reward mode | `reward/rl_critic_reward.py` (reward computation loop) |
| Change critic architecture | `train/init_trado_critic_model.py::_get_critic_model` |
| Add new dataset | `data/prepare_critic_code_data.py`, then new config in `configs/` |
| Change tokenization / prompt format | `train/prompting_utils.py::UniversalPrompting` |
| Adjust GPU sharding | `accelerate_configs/` + `experiment.deepspeed_file` in config |
| Change LR schedule | `configs/*.yaml` → `lr_scheduler` section |
| Debug rollout output format | `sample/trado_rl_rollout.py`, check `rollout_outputs_path()` |
| Debug reward values | `reward/rl_critic_reward.py`, print `raw_token_rewards` before scaling |
| Debug value training fragments | `train/train_trado_value.py::collect_training_data`, print `len(extended_input_ids_list)` |
| Debug checkpoint saving | `train/init_trado_value_model.py::save_checkpoint`, check `_repo_root` |
| Add evaluation metric | `reward/rl_code_reward.py`, extend the result aggregation |
| Change multi-GPU launch | `rl_trado_critic.py::run_accelerate`, `rl.py::_select_accel_config` |
| Run end-to-end on Windows (CPU) | `test_critic_windows.py` |

---

## 26. Suggested Questions to Ask an LLM

**Understanding the model:**
- "In `models/sdar/modeling_sdar.py`, explain exactly how the block-attention mask is constructed and why it differs from standard causal attention."
- "Walk me through one complete forward pass of `SDARForCausalLM` with `input_ids` of shape `[2, 384]`, tracing every tensor shape transformation."
- "Explain the purpose of per-head QK normalization in `SDARAttention.q_norm` / `k_norm`. What problem does it solve?"

**Understanding TraceRL:**
- "In `train/train_trado_value.py::one_round_vectorized`, explain what `pmask_blk` and `ge_mask_blk` represent geometrically in the unmasking process."
- "Explain why `compute_returns_and_advantages_from_fragments` groups rows by `seq_ids` before computing GAE, rather than processing each row independently."
- "The policy loss in `train_trado_policy.py` only applies to positions where `p_mask` is True. Why not train on all response positions?"

**Understanding the critic / IRL:**
- "In `reward/rl_critic_reward.py`, why is the reward `delta_sigmoid = score_k - score_{k-1}` rather than just `score_k`? What property does the delta preserve?"
- "Explain how the `visible_tail_mask` in `CriticModel.forward` ensures the critic scores a *partial* prefix rather than the full response."
- "Why is the critic re-initialized from the base model at the start of each epoch rather than fine-tuned from the previous epoch's weights?"

**For debugging:**
- "Given this traceback from `train_trado_value.py` [paste traceback], identify the root cause and the minimum fix."
- "The loss in value training is NaN from step 2. Given that rewards are zero-mean (z-score normalized), what would cause NaN in the GAE computation?"

**For extensions:**
- "I want to add a process reward model (PRM) that gives per-step rewards in addition to the critic rewards. Which files do I modify and what changes are needed?"
- "How would I implement GRPO (Group Relative Policy Optimization) as an alternative to PPO-clip in this codebase? What replaces the value model?"
- "I want to support multi-turn conversations instead of single-turn code generation. What changes are needed in `UniversalPrompting` and the data pipeline?"
