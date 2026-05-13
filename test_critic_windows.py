"""
Windows smoke test for the Trado critic pipeline.

Tests all new critic code (expert cache, critic init, critic train, reward
generation) on CPU with a tiny randomly-initialised SDAR model.  The vLLM
rollout and the DeepSpeed policy/value trainers are bypassed because they
are Linux-only; synthetic rollout data stands in for them.

Prerequisite (must exist before running):
    ai-project/data/mbpp_critic.jsonl
    ai-project/data/mbpp_masking_trajectories.jsonl

Usage:
    cd repos/dLLM-RL
    python test_critic_windows.py

Cleanup: the script removes its own artefacts (smoke_*) when it finishes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_KEEP_ARTIFACTS = "--keep" in sys.argv

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT.parent.parent / "data"

SMOKE_MODEL_DIR = REPO_ROOT / "smoke_model_windows"
SMOKE_DATA_DIR  = REPO_ROOT / "smoke_data_windows"
SMOKE_PROJECT   = "smoke_test_windows"
SMOKE_DIR       = REPO_ROOT / SMOKE_PROJECT


# -- pretty helpers ------------------------------------------------------------

def section(name: str) -> None:
    print(f"\n{'='*62}\n  {name}\n{'='*62}")

def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")

def info(msg: str) -> None:
    print(f"  [INFO] {msg}")

def die(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    _cleanup()
    sys.exit(1)

def check(cond: bool, msg: str) -> None:
    if cond:
        ok(msg)
    else:
        die(msg)


# -- cleanup -------------------------------------------------------------------

def _cleanup() -> None:
    for d in [SMOKE_MODEL_DIR, SMOKE_DATA_DIR, SMOKE_DIR]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


# -- stage helpers -------------------------------------------------------------

def run_script(subdir: str, script: str, config_path: Path, **overrides) -> None:
    """Run one of the critic pipeline scripts as a subprocess."""
    cwd = REPO_ROOT / subdir if subdir else REPO_ROOT
    # Config path must be relative to the script's cwd
    rel_config = os.path.relpath(config_path, cwd)
    cmd = [sys.executable, script, f"config={rel_config}"]
    for k, v in overrides.items():
        cmd.append(f"{k.replace('__', '.')}={v}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print("\n--- stdout ---")
        print(result.stdout[-3000:] if result.stdout else "(empty)")
        print("--- stderr ---")
        print(result.stderr[-3000:] if result.stderr else "(empty)")
        die(f"Script {subdir}/{script} failed with exit code {result.returncode}")


# =============================================================================
# 0. Prerequisites
# =============================================================================

section("0. Checking prerequisites")

for p in [DATA_ROOT / "mbpp_critic.jsonl",
          DATA_ROOT / "mbpp_masking_trajectories.jsonl"]:
    check(p.exists(), f"Found {p.name}")

from trado_critic_utils import load_jsonl, write_jsonl

all_records = load_jsonl(DATA_ROOT / "mbpp_critic.jsonl")
all_traj    = load_jsonl(DATA_ROOT / "mbpp_masking_trajectories.jsonl")

N = 3
traj_ids = {str(t["task_id"]) for t in all_traj}
records  = [r for r in all_records if str(r["task_id"]) in traj_ids][:N]
test_ids = {str(r["task_id"]) for r in records}
trajs    = [t for t in all_traj if str(t["task_id"]) in test_ids]

check(len(records) == N and len(trajs) == N,
      f"Loaded {N} MBPP examples (records + trajectories)")


# =============================================================================
# 1. Create tiny tokenizer + model
# =============================================================================

section("1. Creating tiny SDAR model (CPU, random weights)")

import torch
from transformers import GPT2TokenizerFast
from models.sdar import SDARConfig, SDARForCausalLM

SMOKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.add_special_tokens({"mask_token": "[MASK]", "pad_token": "[PAD]"})
tok.save_pretrained(SMOKE_MODEL_DIR)
ok(f"Tokenizer: vocab={len(tok)}, pad_id={tok.pad_token_id}, mask_id={tok.mask_token_id}")

cfg = SDARConfig(
    vocab_size=len(tok),
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=32,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    pad_token_id=tok.pad_token_id,
)
cfg.mask_token_id = tok.mask_token_id

tiny = SDARForCausalLM(cfg)
n_params = sum(p.numel() for p in tiny.parameters())
tiny.save_pretrained(SMOKE_MODEL_DIR)
del tiny
ok(f"SDAR model: {n_params:,} parameters -> {SMOKE_MODEL_DIR.name}/")


# =============================================================================
# 2. Write test data files
# =============================================================================

section("2. Writing test data files")

SMOKE_DATA_DIR.mkdir(parents=True, exist_ok=True)

norm_path = SMOKE_DATA_DIR / "test_mbpp_critic.jsonl"
traj_path = SMOKE_DATA_DIR / "test_mbpp_masking_trajectories.jsonl"
write_jsonl(norm_path, records)
write_jsonl(traj_path, trajs)
ok(f"Normalized JSONL: {norm_path.name} ({N} records)")
ok(f"Trajectory JSONL: {traj_path.name} ({N} records)")


# =============================================================================
# 3. Write config YAML
# =============================================================================

section("3. Writing config YAML")

from omegaconf import OmegaConf

# Use forward slashes so the rollout filename stays readable on Windows
model_path_str = str(SMOKE_MODEL_DIR).replace("\\", "/")

config_dict = {
    "experiment": {
        "project": SMOKE_PROJECT,
        "function": "train",
        "current_epoch": 1,
        "total_step": 1,
        "eval_every": 1,
        "save_every": 1,
        "num_node": 1,
        "node_index": 0,
        "deepspeed_file": "1_node_4_gpus_deepspeed_zero3",
    },
    "model": {
        "pretrained_model": model_path_str,
        "value_base_model":  model_path_str,
        "optimized_name":         "optimized",
        "optimized_value_name":   "optimized_value",
        "optimized_critic_name":  "optimized_critic",
        "model_base": "trado",
    },
    "dataset": {
        "train_dataset":    "test_mbpp",
        "optimization_data": "test_mbpp_rl_data",
        "data_type": "code",
    },
    "rollout": {
        "tensor_parallel_size": 1,
        "block_size": 4,
    },
    "training": {
        "block_size":                  4,
        "max_prompt_len":            256,
        "max_gen_length":            128,
        "seed":                       42,
        "gradient_accumulation_steps": 1,
    },
    "critic": {
        "enabled":               True,
        "normalized_data_path":  str(norm_path).replace("\\", "/"),
        "expert_trajectory_path": str(traj_path).replace("\\", "/"),
        "expert_cache_path":     f"{SMOKE_PROJECT}/temp_data/expert_trado_cache.pt",
        "buffer_path":           f"{SMOKE_PROJECT}/temp_data/critic_policy_buffer.pt",
        "buffer_size":           64,
        "learning_rate":         1e-4,
        "batch_size":            4,
        "num_steps_per_epoch":   2,
        "freeze_backbone":       True,
        "pooling":               "visible_response_mean",
        "reward_mode":           "delta_sigmoid",
    },
    "evaluation": {
        "eval_dataset": "test_mbpp",
    },
}

conf = OmegaConf.create(config_dict)
config_path = SMOKE_DATA_DIR / "smoke_config.yaml"
# Add back-reference so child scripts can find the file
config_dict_with_self = dict(config_dict, config=str(config_path).replace("\\", "/"))
conf = OmegaConf.create(config_dict_with_self)
OmegaConf.save(conf, config_path)
ok(f"Config written -> {config_path.name}")


# =============================================================================
# 4. Expert cache building
# =============================================================================

section("4. Expert cache building  (build_trado_expert_trajectory_cache.py)")

SMOKE_DIR.mkdir(parents=True, exist_ok=True)
(SMOKE_DIR / "temp_data").mkdir(parents=True, exist_ok=True)
(SMOKE_DIR / "ckpt").mkdir(parents=True, exist_ok=True)
(SMOKE_DIR / "results").mkdir(parents=True, exist_ok=True)

run_script("data", "build_trado_expert_trajectory_cache.py", config_path)

cache_path = REPO_ROOT / SMOKE_PROJECT / "temp_data" / "expert_trado_cache.pt"
check(cache_path.exists(), f"Expert cache file created at {cache_path.name}")

cache = torch.load(cache_path, map_location="cpu")
check(len(cache["records"]) == N, f"Cache has {N} expert records")
ok(f"First record keys: {list(cache['records'][0].keys())}")


# =============================================================================
# 5. Synthetic rollout data
# =============================================================================

section("5. Creating synthetic rollout data (replaces vLLM rollout)")

# Compute the exact filename the critic scripts will look for
from trado_critic_utils import rollout_outputs_path
rollout_path = rollout_outputs_path(conf, "train")
rollout_path.parent.mkdir(parents=True, exist_ok=True)
info(f"Rollout output path: {rollout_path.name}")

# Build one fake response per problem using the expert record responses
rollout_data = []
for rec in cache["records"]:
    response = rec["response"]
    trado_step_map = list(rec["trado_step_map"])
    rollout_data.append({
        "task_id":    rec["task_id"],
        "prompt":     rec["prompt"],
        "full_output": [response, response],      # 2 identical fake responses
        "step_map":    [trado_step_map, trado_step_map],
    })

with rollout_path.open("w", encoding="utf-8") as fh:
    json.dump(rollout_data, fh, ensure_ascii=False, indent=2)

check(rollout_path.exists(), f"Synthetic rollout JSON written ({len(rollout_data)} problems)")


# =============================================================================
# 6. Critic model init
# =============================================================================

section("6. Critic model init  (init_trado_critic_model.py)")

run_script("train", "init_trado_critic_model.py", config_path)

critic_ckpt = SMOKE_DIR / "ckpt" / "optimized_critic"
check(critic_ckpt.exists(), f"Critic checkpoint created at {critic_ckpt.name}/")


# =============================================================================
# 7. Critic training (2 steps)
# =============================================================================

section("7. Critic training  (train_trado_critic.py, 2 steps)")

run_script("train", "train_trado_critic.py", config_path)

buffer_path = REPO_ROOT / SMOKE_PROJECT / "temp_data" / "critic_policy_buffer.pt"
check(buffer_path.exists(), "Replay buffer saved")

buf = torch.load(buffer_path, map_location="cpu")
check(len(buf.get("samples", [])) > 0, f"Replay buffer has {len(buf.get('samples', []))} policy samples")

check((SMOKE_DIR / "ckpt" / "optimized_critic").exists(),
      "Updated critic checkpoint present after training")


# =============================================================================
# 8. Critic reward generation
# =============================================================================

section("8. Critic reward generation  (rl_critic_reward.py)")

run_script("reward", "rl_critic_reward.py", config_path)

reward_path = SMOKE_DIR / "temp_data" / "test_mbpp_rl_data.json"
check(reward_path.exists(), f"Reward output JSON written at {reward_path.name}")

with reward_path.open("r", encoding="utf-8") as fh:
    reward_data = json.load(fh)

check(len(reward_data) > 0, f"Reward data has {len(reward_data)} entries")

first = reward_data[0]
check("reward" in first and isinstance(first["reward"], list),
      f"Each entry has a 'reward' list (len={len(first['reward'])})")
check("step_map" in first, "Each entry has 'step_map'")
check("response" in first, "Each entry has 'response'")
ok(f"Sample rewards (first 5 tokens): {first['reward'][:5]}")


# =============================================================================
# Done
# =============================================================================

section("ALL TESTS PASSED")
print("""
  The full critic pipeline ran successfully on this Windows machine:
    1. Expert cache building     OK
    2. Synthetic rollout data    OK  (bypasses vLLM)
    3. Critic model init         OK
    4. Critic training (2 steps) OK
    5. Reward generation         OK

  The Linux training pipeline (rollout + policy/value train with DeepSpeed)
  must still be tested on the professor's machine, but all NEW critic code
  has been validated here.
""")

if _KEEP_ARTIFACTS:
    print("  Artifacts kept (--keep). Paths for training smoke test:")
    print(f"    model  : {SMOKE_MODEL_DIR}")
    print(f"    rewards: {SMOKE_DIR / 'temp_data' / 'test_mbpp_rl_data.json'}")
else:
    _cleanup()
