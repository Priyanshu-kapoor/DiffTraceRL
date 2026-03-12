"""
Build a Trado-ready expert trajectory cache from normalized critic-source data.

Why this script exists:
- masking trajectories are built in GPT-2 token space
- the Trado critic must score prefixes in Trado token space
- datasets now come from multiple sources (MBPP, HumanEval, CodeContests), so
  the cache builder must no longer assume MBPP-specific prompt/response shapes

The normalized JSONL file provides:
- question
- prefix
- trajectory_code
- response_prefix / response_suffix
- test_method

This is enough to reconstruct the exact prompt/response surface that rollout
uses, while still transferring expert reveal order from GPT-2 tokens onto Trado
tokens.
"""

import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trado_critic_utils import (
    format_code_response,
    load_jsonl,
    render_rollout_prompt,
    resolve_repo_path,
)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def dense_step_map_from_trajectory(trajectory: list[dict], token_count: int) -> list[int]:
    """
    Convert sparse reveal positions into one dense step id per GPT-2 token.
    """
    step_map = [-1] * token_count
    for step_entry in trajectory:
        step_id = int(step_entry["step"])
        for position in step_entry["reveal_positions"]:
            if position < 0 or position >= token_count:
                raise ValueError(f"Reveal position {position} is out of bounds for token_count={token_count}")
            if step_map[position] != -1:
                raise ValueError(f"Token position {position} is revealed more than once")
            step_map[position] = step_id

    missing = [idx for idx, value in enumerate(step_map) if value == -1]
    if missing:
        raise ValueError(f"Trajectory does not cover all GPT-2 tokens. Missing positions: {missing[:10]}")

    return step_map


def build_char_step_map(text: str, offsets: list[tuple[int, int]], token_step_map: list[int]) -> list[int]:
    """
    Expand token-level reveal steps to a character-level array.

    Later, Trado token spans can read from this character array to inherit the
    expert reveal order even though the token boundaries differ from GPT-2.
    """
    char_steps = [-1] * len(text)
    for token_index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        step_id = token_step_map[token_index]
        for char_index in range(start, end):
            char_steps[char_index] = max(char_steps[char_index], step_id)

    missing = [idx for idx, value in enumerate(char_steps) if value == -1]
    if missing:
        raise ValueError(f"Character-level step map has gaps. Missing character positions: {missing[:10]}")

    return char_steps


def trado_steps_for_formatted_response(
    trajectory_code: str,
    response_prefix: str,
    response_suffix: str,
    trajectory_entry: dict,
    gpt2_tokenizer,
    trado_tokenizer,
) -> tuple[str, list[int], list[int]]:
    """
    Build the final Trado-aligned response and step map.

    Important detail:
    - The expert trajectory is always built on `trajectory_code`, i.e. the part
      of the assistant response that the model actually generates.
    - The final response text may still include formatting wrappers such as code
      fences. Those wrapper tokens get synthetic reveal steps:
        * response prefix -> step 0
        * response suffix -> final code step
    """
    trajectory_code = trajectory_code.rstrip("\n")
    response = format_code_response(trajectory_code, response_prefix, response_suffix)

    gpt2_encoded = gpt2_tokenizer(trajectory_code, add_special_tokens=False, return_offsets_mapping=True)
    gpt2_ids = list(gpt2_encoded["input_ids"])
    gpt2_offsets = list(gpt2_encoded["offset_mapping"])

    stored_ids = list(trajectory_entry["tokens"])
    if gpt2_ids != stored_ids:
        raise ValueError(
            f"GPT-2 token ids do not match the stored trajectory tokens for task_id={trajectory_entry['task_id']}"
        )

    gpt2_step_map = dense_step_map_from_trajectory(trajectory_entry["trajectory"], len(gpt2_ids))
    code_char_steps = build_char_step_map(trajectory_code, gpt2_offsets, gpt2_step_map)
    last_step = max(gpt2_step_map)

    formatted_char_steps = [0] * len(response_prefix)
    formatted_char_steps.extend(code_char_steps)
    formatted_char_steps.extend([last_step] * len(response_suffix))

    trado_encoded = trado_tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    if not getattr(trado_tokenizer, "is_fast", False):
        raise ValueError("The Trado tokenizer must be a fast tokenizer to expose offset mappings")

    trado_ids = list(trado_encoded["input_ids"])
    trado_offsets = list(trado_encoded["offset_mapping"])
    trado_step_map: list[int] = []

    for start, end in trado_offsets:
        if end <= start:
            raise ValueError("Encountered a zero-length Trado token span while building expert cache")
        char_slice = formatted_char_steps[start:end]
        if not char_slice:
            raise ValueError("Encountered an empty Trado character slice while building expert cache")
        trado_step_map.append(max(char_slice))

    if len(trado_ids) != len(trado_step_map):
        raise ValueError("Trado token ids and Trado step map have different lengths")

    return response, trado_ids, trado_step_map


def main():
    config = get_config()

    trajectory_path = resolve_repo_path(config.critic.expert_trajectory_path)
    normalized_path = resolve_repo_path(
        config.critic.get("normalized_data_path", config.critic.reference_data_path)
    )
    cache_path = resolve_repo_path(config.critic.expert_cache_path)

    trajectories = load_jsonl(trajectory_path)
    normalized_records = load_jsonl(normalized_path)

    trajectories_by_id = {str(item["task_id"]): item for item in trajectories}
    records_by_id = {str(item["task_id"]): item for item in normalized_records}

    trajectory_ids = set(trajectories_by_id)
    record_ids = set(records_by_id)
    if trajectory_ids != record_ids:
        missing_in_ref = sorted(trajectory_ids - record_ids)
        missing_in_traj = sorted(record_ids - trajectory_ids)
        raise ValueError(
            "The expert trajectory file and reference file must match exactly on task_id. "
            f"Missing in reference: {missing_in_ref[:10]}, missing in trajectory: {missing_in_traj[:10]}"
        )

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    trado_tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if not getattr(trado_tokenizer, "is_fast", False):
        raise ValueError("The configured Trado tokenizer is not a fast tokenizer")

    records = []
    for task_id in sorted(record_ids):
        ref_item = records_by_id[task_id]
        traj_item = trajectories_by_id[task_id]

        if ref_item["question"] != traj_item["question"]:
            raise ValueError(f"Question mismatch for task_id={task_id}")

        prompt = render_rollout_prompt(
            question=ref_item["question"],
            test_method=ref_item["test_method"],
            prefix=ref_item.get("prefix", ""),
        )
        response, trado_ids, trado_step_map = trado_steps_for_formatted_response(
            trajectory_code=ref_item["trajectory_code"],
            response_prefix=ref_item.get("response_prefix", "```python\n"),
            response_suffix=ref_item.get("response_suffix", "\n```"),
            trajectory_entry=traj_item,
            gpt2_tokenizer=gpt2_tokenizer,
            trado_tokenizer=trado_tokenizer,
        )

        records.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "reference": ref_item["reference_code"],
                "response": response,
                "trado_input_ids": trado_ids,
                "trado_step_map": trado_step_map,
                "test_method": ref_item["test_method"],
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "records": records,
            "meta": {
                "trajectory_path": str(trajectory_path),
                "reference_path": str(normalized_path),
                "record_count": len(records),
                "tokenizer_path": config.model.pretrained_model,
            },
        },
        cache_path,
    )
    print(f"Saved expert cache to {cache_path}")


if __name__ == "__main__":
    main()
