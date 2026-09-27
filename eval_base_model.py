"""
eval_base_model.py
==================
Standalone evaluation of the UNMODIFIED base TraDo-4B model — no critic,
no RL, no training. This gives us the anchor baseline number that every
trained checkpoint must be compared against.

It reuses the existing pipeline pieces (rollout → execute → score) exactly
as the training loop does at eval time, but points the model at the original
pretrained weights instead of any optimized_* checkpoint.

WHAT IT MEASURES:
    pass@1 (and pass@k) of base TraDo-4B on the eval split of a chosen dataset.

USAGE (single command):
    python eval_base_model.py \
        config=configs/rl_trado_critic_mbpp_linux.yaml \
        model.pretrained_model=<path_to_ORIGINAL_trado4B_weights> \
        experiment.project_name=BASELINE_trado4b_base \
        dataset.eval_dataset=humaneval \
        rollout.num_response_per_task=8 \
        rollout.max_token=2000 \
        training.max_gen_length=2000

IMPORTANT NOTES:
  - Point model.pretrained_model at the ORIGINAL TraDo-4B weights,
    NOT any ckpt/optimized_* directory.
  - Set max_token / max_gen_length back to 2000 (NOT 512) so we measure
    the model's true ceiling without truncation artifacts.
  - Run this on the SAME eval_dataset the training run reports "Eval Acc" on,
    so the numbers are directly comparable.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent


def run_python(script, config_path, extra_args=None):
    """Run a pipeline subprocess with cwd set to the script's subdirectory."""
    script_path = Path(script)
    cwd = (REPO_ROOT / script_path.parent).resolve()
    cmd = [sys.executable, script_path.name, f"config={config_path}"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*70}\nRunning: {' '.join(cmd)}  (cwd={cwd})\n{'='*70}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}")
        sys.exit(1)


def main():
    # ── Load config with CLI overrides (same pattern as rl_trado_critic.py) ──
    cli_conf = OmegaConf.from_cli()
    assert "config" in cli_conf, "Must pass config=<path> on the command line"
    base_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(base_conf, cli_conf)

    # ── Safety checks: make sure we are pointing at BASE weights ──
    pretrained = config.model.get("pretrained_model", "")
    if "optimized" in str(pretrained) or "ckpt" in str(pretrained):
        print("\n" + "!"*70)
        print("WARNING: model.pretrained_model points at what looks like a")
        print(f"         trained checkpoint:  {pretrained}")
        print("         For the BASE baseline, point it at the ORIGINAL")
        print("         pretrained TraDo-4B weights instead.")
        print("!"*70)
        resp = input("\nContinue anyway? [y/N]: ")
        if resp.lower() != "y":
            sys.exit(0)

    # ── Report what we are about to run ──
    print("\n" + "="*70)
    print("BASE MODEL BASELINE EVALUATION")
    print("="*70)
    print(f"  Model weights:      {config.model.pretrained_model}")
    print(f"  Eval dataset:       {config.dataset.get('eval_dataset', '<unset>')}")
    print(f"  Responses per task: {config.rollout.get('num_response_per_task', '<unset>')}")
    print(f"  Max tokens:         {config.rollout.get('max_token', '<unset>')}")
    print(f"  Max gen length:     {config.training.get('max_gen_length', '<unset>')}")
    print(f"  Project name:       {config.experiment.get('project_name', '<unset>')}")
    print("="*70)

    # ── Set up an isolated project directory for baseline outputs ──
    project_dir = Path(config.experiment.get("project_dir", "./baseline_eval"))
    (project_dir / "results").mkdir(parents=True, exist_ok=True)
    (project_dir / "temp_data").mkdir(parents=True, exist_ok=True)

    # Write the resolved config to a temp file the subprocesses will reuse
    runtime_config_path = project_dir / "baseline_resolved_config.yaml"
    OmegaConf.save(config, runtime_config_path)
    cfg = str(runtime_config_path)

    # ── STEP 1: Rollout on the EVAL split using base weights ──
    # trado_rl_rollout.py reads the eval dataset and generates responses.
    # We pass split=eval so it uses the eval dataset, not train.
    run_python(
        "sample/trado_rl_rollout.py",
        cfg,
        extra_args=["split=eval"],
    )

    # ── STEP 2: Execute generated code in the sandbox ──
    # rl_execute.py adds the "correctness" field to each rollout entry.
    run_python(
        "reward/rl_execute.py",
        cfg,
        extra_args=["split=eval"],
    )

    # ── STEP 3: Score pass@k from correctness ──
    # rl_code_reward.py computes pass@1 / pass@k and writes results file.
    run_python(
        "reward/rl_code_reward.py",
        cfg,
        extra_args=["split=eval"],
    )

    # ── STEP 4: Read and print the results ──
    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70)

    results_files = sorted((project_dir / "results").glob("results-eval-*.txt"))
    if not results_files:
        # Fall back: compute pass@k directly from the rollout JSON
        print("No results file found — computing pass@k from rollout JSON directly.\n")
        compute_pass_at_k_from_json(project_dir, config)
    else:
        latest = results_files[-1]
        print(f"Results file: {latest}\n")
        with open(latest) as f:
            print(f.read())

    print("\n" + "="*70)
    print("Record this number. It is the anchor every trained checkpoint")
    print("must be compared against. If TraceRL+critic does not beat this,")
    print("the method is not adding value.")
    print("="*70)


def compute_pass_at_k_from_json(project_dir, config):
    """
    Fallback pass@k computation directly from the rollout output JSON,
    in case the results-eval-*.txt file is not produced.

    Uses the standard unbiased estimator:
        pass@k = 1 - C(n-c, k) / C(n, k)
    """
    from math import comb
    import numpy as np

    # Find the rollout output JSON
    temp_files = list((project_dir / "temp_data").glob("outputs-*.json"))
    if not temp_files:
        print("Could not find rollout output JSON. Check the pipeline ran.")
        return

    with open(temp_files[-1]) as f:
        data = json.load(f)

    def pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)

    pass1_list, pass5_list = [], []
    for entry in data:
        correctness = entry.get("correctness", [])
        # A response is correct if ALL its test cases pass
        per_response_correct = [all(tests) for tests in correctness]
        n = len(per_response_correct)
        c = sum(per_response_correct)
        if n == 0:
            continue
        pass1_list.append(pass_at_k(n, c, 1))
        if n >= 5:
            pass5_list.append(pass_at_k(n, c, 5))

    print(f"Tasks evaluated:  {len(pass1_list)}")
    print(f"pass@1:  {100*np.mean(pass1_list):.2f}%")
    if pass5_list:
        print(f"pass@5:  {100*np.mean(pass5_list):.2f}%")


if __name__ == "__main__":
    main()
