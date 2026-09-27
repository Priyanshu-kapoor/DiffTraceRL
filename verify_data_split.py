"""
verify_data_split.py
====================
Checks for task-ID overlap between the training and evaluation datasets
and reports the MBPP HuggingFace split each dataset was built from.

Usage:
    python verify_data_split.py \
        --config configs/rl_trado_critic_mbpp_linux.yaml \
        --data-root ../../data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_data_file(name: str, data_root: Path) -> tuple[Path, str]:
    """Return (path, format) for a dataset name.  Tries .jsonl first, then .json."""
    for suffix, fmt in [(".jsonl", "jsonl"), (".json", "json")]:
        p = data_root / f"{name}{suffix}"
        if p.exists():
            return p, fmt
    # Also check repo-local data/
    for suffix, fmt in [(".json", "json"), (".jsonl", "jsonl")]:
        p = REPO_ROOT / "data" / f"{name}{suffix}"
        if p.exists():
            return p, fmt
    return None, None


def load_dataset(name: str, data_root: Path) -> tuple[list[dict], Path | None]:
    path, fmt = resolve_data_file(name, data_root)
    if path is None:
        return [], None
    if fmt == "jsonl":
        return load_jsonl(path), path
    return load_json(path), path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify train/eval dataset split overlap")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Workspace-level data root (default: ../../data relative to repo root)",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    data_root = Path(args.data_root).resolve() if args.data_root else (REPO_ROOT / ".." / ".." / "data").resolve()

    train_name = str(config.dataset.train_dataset)
    eval_name  = str(config.evaluation.eval_dataset)

    print(f"Config:           {args.config}")
    print(f"Data root:        {data_root}")
    print(f"Train dataset:    {train_name}")
    print(f"Eval dataset:     {eval_name}")
    print()

    train_items, train_path = load_dataset(train_name, data_root)
    eval_items,  eval_path  = load_dataset(eval_name, data_root)

    if not train_items:
        print(f"ERROR: Could not load train dataset '{train_name}' from {data_root}")
        print("       Run: python data/prepare_critic_code_data.py --dataset mbpp")
        sys.exit(1)

    if not eval_items:
        print(f"ERROR: Could not load eval dataset '{eval_name}' from {data_root}")
        sys.exit(1)

    print(f"Train file:  {train_path}  ({len(train_items)} records)")
    print(f"Eval file:   {eval_path}   ({len(eval_items)} records)")
    print()

    train_ids = {str(item["task_id"]) for item in train_items}
    eval_ids  = {str(item["task_id"]) for item in eval_items}
    overlap   = train_ids & eval_ids

    print(f"Unique train task IDs:  {len(train_ids)}")
    print(f"Unique eval task IDs:   {len(eval_ids)}")
    print(f"Overlapping task IDs:   {len(overlap)}")
    print()

    same_file = (train_path is not None and eval_path is not None
                 and train_path.resolve() == eval_path.resolve())
    same_name = train_name == eval_name

    if same_file or same_name:
        print("WARNING: Train and eval datasets are the SAME file.")
        print("         Evaluation accuracy reflects in-distribution performance only.")
        print("         For a proper held-out benchmark, use separate splits:")
        print("           - build mbpp_critic_train.jsonl from dataset['train']  (374 tasks)")
        print("           - build mbpp_critic_eval.jsonl  from dataset['test']   (500 tasks)")
        print("         Then update the config:")
        print("           dataset.train_dataset: 'mbpp_critic_train'")
        print("           evaluation.eval_dataset: 'mbpp_critic_eval'")
        print()
    elif overlap:
        print(f"WARNING: {len(overlap)} task IDs appear in BOTH train and eval splits.")
        print("         Sample overlapping IDs:", sorted(overlap)[:10])
        print()
    else:
        print("OK: No overlap between train and eval task IDs.")
        print()

    # Show HuggingFace source split for context
    sources = {item.get("source_dataset") for item in train_items}
    print(f"Source dataset(s) in train file: {sources}")
    print()
    print("MBPP split note:")
    print("  dataset['train']      — 374 tasks (used by prepare_critic_code_data.py)")
    print("  dataset['validation'] — 90 tasks")
    print("  dataset['test']       — 500 tasks (standard held-out benchmark)")
    print()
    print("  Currently prepare_critic_code_data.py only uses dataset['train'].")
    print("  To create a proper eval split from dataset['test'], add a --split")
    print("  argument to prepare_critic_code_data.py --dataset mbpp.")
    print()

    if overlap and not same_file:
        sys.exit(2)


if __name__ == "__main__":
    main()
