"""
Build GPT-2 entropy-ranked masking trajectories for any normalized critic-source JSONL.

The output schema mirrors the earlier MBPP-only trajectory file, but now the
source text comes from a generic field such as `trajectory_code`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trado_critic_utils import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPT-2 entropy masking trajectories")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--text-field",
        default="trajectory_code",
        help="Which field in the normalized JSONL contains the code to score.",
    )
    parser.add_argument(
        "--step-ratio",
        type=float,
        default=0.2,
        help="Fraction of tokens grouped into each reveal step.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help="Autoregressive model used to compute next-token entropy.",
    )
    return parser.parse_args()


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum(dim=-1)


def get_entropy(tokenizer, model, text: str) -> tuple[list[int] | None, list[float] | None]:
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < 2:
        return None, None

    with torch.no_grad():
        outputs = model(**inputs)

    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    entropy = compute_entropy(shift_logits)

    return shift_labels[0].tolist(), entropy[0].tolist()


def build_steps(tokens: list[int], entropy: list[float], step_ratio: float) -> list[list[int]]:
    token_indices = list(range(len(tokens)))
    sorted_indices = sorted(token_indices, key=lambda idx: entropy[idx], reverse=True)
    step_size = max(1, int(len(tokens) * step_ratio))

    steps = []
    for start in range(0, len(tokens), step_size):
        steps.append(sorted_indices[start : start + step_size])

    # Reverse so lower-entropy tokens are revealed first, matching the earlier
    # MBPP experiments the user already prepared.
    steps.reverse()
    return steps


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    source_items = load_jsonl(Path(args.input_jsonl).resolve())
    output_items = []

    for item in source_items:
        code_text = item[args.text_field]
        tokens, entropy = get_entropy(tokenizer, model, code_text)
        if tokens is None:
            continue

        steps = build_steps(tokens, entropy, args.step_ratio)
        trajectory = []
        for step_id, reveal_positions in enumerate(steps):
            trajectory.append(
                {
                    "step": step_id,
                    "reveal_positions": reveal_positions,
                }
            )

        output_items.append(
            {
                "task_id": item["task_id"],
                "question": item["question"],
                "trajectory_code": code_text,
                "tokens": tokens,
                "text_tokens": tokenizer.convert_ids_to_tokens(tokens),
                "trajectory": trajectory,
            }
        )

    output_path = Path(args.output_jsonl).resolve()
    write_jsonl(output_path, output_items)

    print(f"Built {len(output_items)} masking trajectories")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
