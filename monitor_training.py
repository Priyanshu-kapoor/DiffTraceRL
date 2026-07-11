"""
monitor_training.py
===================
Drop this file into the root of the dLLM-RL project directory.

It does two things:
  1. LOGGING  — call log_critic_loss() and log_policy_reward() from
                inside the training scripts to append data to JSON logs.
  2. PLOTTING — run this script directly to generate plots from those logs.

Usage:
  # From inside train/train_trado_critic.py training loop:
  #   from monitor_training import log_critic_loss
  #   log_critic_loss(epoch, step, loss_value, project_dir)

  # From inside reward/rl_critic_reward.py after reward computation:
  #   from monitor_training import log_policy_reward
  #   log_policy_reward(epoch, reward_list_of_lists, project_dir)

  # To generate plots (run from project root):
  #   python monitor_training.py --project_dir <path_to_project>
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOGGING FUNCTIONS
# Add calls to these inside the training scripts
# ─────────────────────────────────────────────────────────────────────────────

def log_critic_loss(epoch: int, step: int, loss: float, project_dir: str):
    """
    Call this inside the training loop in train/train_trado_critic.py.

    WHERE TO ADD IT:
        # In train/train_trado_critic.py, inside the for-loop at step 8:
        for step in range(critic.num_steps_per_epoch):
            ...
            loss = F.binary_cross_entropy_with_logits(logits, labels_tensor)
            loss.backward(); optimizer.step(); optimizer.zero_grad()

            # ADD THIS:
            from monitor_training import log_critic_loss
            log_critic_loss(epoch=current_epoch, step=step,
                            loss=loss.item(), project_dir=config.experiment.project_dir)
    """
    log_path = Path(project_dir) / "logs" / "critic_loss_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {"epoch": epoch, "step": step, "loss": loss}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_policy_reward(epoch: int, reward_data: list, project_dir: str):
    """
    Call this in reward/rl_critic_reward.py after computing scaled rewards.

    WHERE TO ADD IT:
        # In reward/rl_critic_reward.py, after building output_data list:
        from monitor_training import log_policy_reward
        log_policy_reward(epoch=current_epoch,
                          reward_data=output_data,   # list of dicts with "reward" field
                          project_dir=config.experiment.project_dir)

    reward_data format:
        [{"reward": [0.02, 0.08, ...], ...}, ...]   # per-token rewards per response
    """
    log_path = Path(project_dir) / "logs" / "policy_reward_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate stats from the per-token reward vectors
    all_tokens = [r for entry in reward_data for r in entry["reward"]]
    returns    = [sum(entry["reward"]) for entry in reward_data]

    entry = {
        "epoch":             epoch,
        "mean_token_reward": float(np.mean(all_tokens)),
        "std_token_reward":  float(np.std(all_tokens)),
        "mean_return":       float(np.mean(returns)),
        "std_return":        float(np.std(returns)),
        "max_return":        float(np.max(returns)),
        "min_return":        float(np.min(returns)),
        "pct_positive":      float(np.mean([r > 0 for r in returns])),
        "n_responses":       len(returns),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PLOTTING FUNCTIONS
# Run python monitor_training.py --project_dir <path> to generate plots
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_critic_loss(project_dir: str, save: bool = True):
    """
    Plot 1: Critic loss per training step, coloured by epoch.
    Shows whether the critic is learning to distinguish expert from policy prefixes.

    Healthy pattern:
      - Within each epoch: loss starts ~0.693, drops to <0.4 by final steps
      - Across epochs: no systematic worsening (critic converges each time it is re-init'd)

    Red flags:
      - Loss stays at ~0.693 throughout  → critic not learning at all
      - Loss jumps to >1.0              → numerical instability
      - Near-zero loss from step 0      → labels are all same class (data bug)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    log_path = Path(project_dir) / "logs" / "critic_loss_log.jsonl"
    if not log_path.exists():
        print(f"No critic loss log found at {log_path}")
        return

    records = load_jsonl(log_path)
    epochs  = sorted(set(r["epoch"] for r in records))
    colors  = cm.viridis(np.linspace(0, 1, max(len(epochs), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss per step, coloured by epoch
    ax = axes[0]
    for i, epoch in enumerate(epochs):
        epoch_records = [r for r in records if r["epoch"] == epoch]
        steps  = [r["step"]  for r in epoch_records]
        losses = [r["loss"]  for r in epoch_records]
        ax.plot(steps, losses, color=colors[i], alpha=0.7, linewidth=1.2,
                label=f"Epoch {epoch}")

    ax.axhline(0.693, color="red", linestyle="--", linewidth=1.0,
               label="Random baseline (0.693)")
    ax.set_xlabel("Training Step (within epoch)")
    ax.set_ylabel("Critic BCE Loss")
    ax.set_title("Critic Loss Per Step\n(should drop from ~0.693 each epoch)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    # Right: final critic loss per epoch (last 10 steps averaged)
    ax = axes[1]
    final_losses = []
    for epoch in epochs:
        epoch_records = [r for r in records if r["epoch"] == epoch]
        last_10 = epoch_records[-10:] if len(epoch_records) >= 10 else epoch_records
        final_losses.append(np.mean([r["loss"] for r in last_10]))

    ax.plot(epochs, final_losses, marker="o", color="steelblue", linewidth=2)
    ax.axhline(0.693, color="red", linestyle="--", linewidth=1.0,
               label="Random baseline")
    ax.fill_between(epochs, 0.693, final_losses,
                    where=[l < 0.693 for l in final_losses],
                    alpha=0.15, color="green", label="Below random = critic learning")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Final Critic Loss (last 10 steps)")
    ax.set_title("Critic Convergence Per Epoch\n(should stay well below 0.693)")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = Path(project_dir) / "logs" / "critic_loss_plot.png"
    if save:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved critic loss plot → {out_path}")
    plt.show()


def plot_policy_reward(project_dir: str, save: bool = True):
    """
    Plot 2: Policy reward statistics across epochs.
    Shows whether the policy is generating increasingly expert-like code.

    Healthy pattern:
      - mean_return trending upward across epochs
      - std_return stable or slightly decreasing (policy becoming more consistent)
      - pct_positive staying near 0.5 (normalization working correctly)

    Red flags:
      - mean_return flat at ~0.0     → critic reward signal is dead
      - std_return near 0.0          → all responses get same reward (no discrimination)
      - pct_positive far from 0.5   → z-score normalization not working
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    log_path = Path(project_dir) / "logs" / "policy_reward_log.jsonl"
    if not log_path.exists():
        print(f"No policy reward log found at {log_path}")
        return

    records = load_jsonl(log_path)
    epochs       = [r["epoch"]        for r in records]
    mean_returns = [r["mean_return"]  for r in records]
    std_returns  = [r["std_return"]   for r in records]
    pct_positive = [r["pct_positive"] for r in records]
    max_returns  = [r["max_return"]   for r in records]
    min_returns  = [r["min_return"]   for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: mean return ± std across epochs
    ax = axes[0]
    ax.plot(epochs, mean_returns, marker="o", color="steelblue",
            linewidth=2, label="Mean return")
    ax.fill_between(epochs,
                    [m - s for m, s in zip(mean_returns, std_returns)],
                    [m + s for m, s in zip(mean_returns, std_returns)],
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.plot(epochs, max_returns, linestyle="--", color="green",
            alpha=0.6, linewidth=1, label="Max return")
    ax.plot(epochs, min_returns, linestyle="--", color="red",
            alpha=0.6, linewidth=1, label="Min return")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Return per Response (= critic final score)")
    ax.set_title("Policy Return Across Epochs\n(mean should trend upward)")
    ax.legend()

    # Middle: std of returns — measures discrimination quality
    ax = axes[1]
    ax.plot(epochs, std_returns, marker="s", color="darkorange", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Std of Returns")
    ax.set_title("Return Variance Across Epochs\n(near-zero = critic not discriminating)")
    ax.axhline(0, color="gray", linewidth=0.8)

    # Right: fraction of responses with positive return
    ax = axes[2]
    ax.plot(epochs, pct_positive, marker="^", color="purple", linewidth=2)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1,
               label="Expected (z-score → 50%)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of Responses with Positive Return")
    ax.set_title("% Positive Returns\n(should stay near 0.5 — z-score property)")
    ax.legend()

    plt.tight_layout()
    out_path = Path(project_dir) / "logs" / "policy_reward_plot.png"
    if save:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved policy reward plot → {out_path}")
    plt.show()


def plot_pass_at_k(results_dir: str, project_dir: str, save: bool = True):
    """
    Plot 3: pass@1 across eval checkpoints.
    Reads from <project>/results/results-eval-*.txt files.

    File format expected (one metric per line):
        pass@1: 0.234
        pass@5: 0.412
    """
    try:
        import matplotlib.pyplot as plt
        import glob
    except ImportError:
        print("matplotlib not installed.")
        return

    results_path = Path(results_dir)
    result_files = sorted(results_path.glob("results-eval-*.txt"))

    if not result_files:
        print(f"No eval result files found in {results_dir}")
        return

    epochs, pass1_scores = [], []

    for fpath in result_files:
        stem = fpath.stem
        parts = [p for p in stem.split("-") if p.startswith("epoch")]
        epoch = int(parts[0].replace("epoch", "")) if parts else len(epochs)

        with open(fpath) as f:
            for line in f:
                if "pass@1" in line.lower():
                    try:
                        val = float(line.split(":")[-1].strip())
                        epochs.append(epoch)
                        pass1_scores.append(val)
                        break
                    except ValueError:
                        pass

    if not epochs:
        print("Could not parse pass@1 values from result files.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, pass1_scores, marker="o", color="teal",
            linewidth=2, markersize=7, label="TraceRL + Critic (ours)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("pass@1")
    ax.set_title("pass@1 Over Training\n(primary metric — should trend upward)")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = Path(project_dir) / "logs" / "pass_at_1_plot.png"
    if save:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved pass@1 plot → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training diagnostics")
    parser.add_argument("--project_dir", type=str, required=True,
                        help="Path to the experiment project directory")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results/ folder for pass@1 plot (optional)")
    parser.add_argument("--no_save", action="store_true",
                        help="Show plots without saving")
    args = parser.parse_args()

    save = not args.no_save
    print(f"\nGenerating training diagnostic plots for: {args.project_dir}\n")

    plot_critic_loss(args.project_dir, save=save)
    plot_policy_reward(args.project_dir, save=save)

    if args.results_dir:
        plot_pass_at_k(args.results_dir, args.project_dir, save=save)
    else:
        print("Skipping pass@1 plot (no --results_dir provided)")

    print("\nDone. Plots saved to <project_dir>/logs/")
