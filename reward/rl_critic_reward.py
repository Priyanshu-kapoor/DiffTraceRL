"""
Generate dense critic rewards for the standalone Trado critic pipeline.

This script reads the rollout outputs produced by the existing Trado sampler,
reconstructs the partial prefixes implied by `step_map`, scores those prefixes
with the standalone critic, and writes the standard optimization JSON consumed
by `train_trado_value.py`.

The output schema intentionally matches the existing reward scripts:
- prompt
- response
- step_map
- reward        (vector instead of scalar)
"""

import json
import os
import sys
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from models import SDARForCausalLM
from trado_critic_utils import (
    align_step_map,
    build_prefix_examples,
    make_basic_block_attention,
    process_pad,
    resolve_repo_path,
    rollout_outputs_path,
    z_score_normalize,
)
from train.init_trado_critic_model import _get_critic_model
from train.prompting_utils import UniversalPrompting


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def flatten_rollout_outputs(data: list[dict]) -> tuple[list[dict], list[list[int]]]:
    """
    Flatten responses across prompts while keeping track of prompt-level groups.
    """
    flat_samples = []
    grouped_indices = []
    for item in data:
        group = []
        for response, step_map in zip(item.get("full_output", []), item.get("step_map", [])):
            if response is None or response == "":
                continue
            flat_samples.append(
                {
                    "prompt": item["prompt"],
                    "response": response,
                    "step_map": step_map,
                }
            )
            group.append(len(flat_samples) - 1)
        grouped_indices.append(group)
    return flat_samples, grouped_indices


def score_examples(model, examples: list[dict], start_pos: int, pad_id: int, block_size: int, batch_size: int, device):
    """
    Score critic prefix examples in small batches.
    """
    if not examples:
        return []

    total_len = examples[0]["extended_input_ids"].numel()
    base_attention = make_basic_block_attention(total_len, start_pos, block_size)
    scores = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start : batch_start + batch_size]
            extended_input_ids = torch.stack([item["extended_input_ids"] for item in batch_examples], dim=0)
            visible_tail_masks = torch.stack([item["visible_tail_mask"] for item in batch_examples], dim=0)
            tok_idx_ext = torch.stack([item["tok_idx_ext"] for item in batch_examples], dim=0)

            attention_mask = base_attention.repeat_interleave(extended_input_ids.shape[0], dim=0)
            attention_mask = process_pad(attention_mask, extended_input_ids, start_pos, pad_id)

            logits = model(
                input_ids=extended_input_ids.to(device),
                attention_mask=attention_mask.to(device),
                position_ids=tok_idx_ext.to(device),
                visible_tail_mask=visible_tail_masks.to(device),
            )
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())

    return scores


def main():
    config = get_config()
    outputs_path = rollout_outputs_path(config, "train")

    with outputs_path.open("r", encoding="utf-8") as handle:
        rollout_data = json.load(handle)

    flat_samples, grouped_indices = flatten_rollout_outputs(rollout_data)
    if not flat_samples:
        raise ValueError("Rollout outputs contain no policy samples to score")

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    critic_checkpoint = os.path.join(_repo_root, config.experiment.project, "ckpt", config.model.optimized_critic_name)
    tokenizer = AutoTokenizer.from_pretrained(critic_checkpoint, trust_remote_code=True)
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_prompt_len=config.training.max_prompt_len,
        max_gen_length=config.training.max_gen_length,
        ignore_id=-100,
    )

    critic_model_class = _get_critic_model(SDARForCausalLM, "critic_head")
    model = critic_model_class.from_pretrained(critic_checkpoint, trust_remote_code=True, torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    prompt_list = [item["prompt"] for item in flat_samples]
    response_list = [item["response"] for item in flat_samples]
    step_maps = [item["step_map"] for item in flat_samples]

    input_ids, _, start_pos, _ = uni_prompting((prompt_list, response_list))
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    resp_len = input_ids.shape[1] - start_pos

    aligned_step_maps = [align_step_map(step_map, resp_len) for step_map in step_maps]
    prefix_examples = build_prefix_examples(
        input_ids=input_ids,
        start_pos=start_pos,
        pad_id=pad_id,
        mask_id=mask_id,
        step_maps=aligned_step_maps,
    )

    example_scores = score_examples(
        model=model,
        examples=prefix_examples,
        start_pos=start_pos,
        pad_id=pad_id,
        block_size=config.training.block_size,
        batch_size=int(config.critic.batch_size),
        device=device,
    )

    # Collect prefix scores back by response index.
    scores_by_sample = defaultdict(list)
    for example, score in zip(prefix_examples, example_scores):
        scores_by_sample[int(example["sample_index"])].append((int(example["step"]), float(score)))

    raw_token_rewards = []
    trajectory_returns = []

    for sample_index, sample in enumerate(flat_samples):
        valid_resp = input_ids[sample_index, start_pos:].ne(pad_id)
        step_map = aligned_step_maps[sample_index]

        step_scores = sorted(scores_by_sample[sample_index], key=lambda item: item[0])
        previous_score = 0.0
        reward_by_step = {}
        for step_id, current_score in step_scores:
            reward_by_step[step_id] = current_score - previous_score
            previous_score = current_score

        token_rewards = []
        for token_index, step_id in enumerate(step_map):
            if bool(valid_resp[token_index]):
                token_rewards.append(float(reward_by_step.get(step_id, 0.0)))
            else:
                token_rewards.append(0.0)

        raw_token_rewards.append(token_rewards)
        trajectory_returns.append(sum(reward_by_step.values()))

    normalized_returns = [0.0 for _ in flat_samples]
    for group in grouped_indices:
        if not group:
            continue
        group_returns = [trajectory_returns[index] for index in group]
        group_norm = z_score_normalize(group_returns)
        for offset, sample_index in enumerate(group):
            normalized_returns[sample_index] = group_norm[offset]

    final_data = []
    for sample_index, sample in enumerate(flat_samples):
        token_rewards = raw_token_rewards[sample_index]
        token_sum = sum(token_rewards)
        target_return = normalized_returns[sample_index]

        if token_sum == 0 or target_return == 0:
            scaled_rewards = [0.0 for _ in token_rewards]
        else:
            scale = target_return / token_sum
            scaled_rewards = [float(value * scale) for value in token_rewards]

        final_data.append(
            {
                "prompt": sample["prompt"],
                "reward": scaled_rewards,
                "response": sample["response"],
                "step_map": aligned_step_maps[sample_index],
            }
        )

    optimization_path = resolve_repo_path(
        os.path.join(config.experiment.project, "temp_data", f"{config.dataset.optimization_data}.json")
    )
    optimization_path.parent.mkdir(parents=True, exist_ok=True)
    with optimization_path.open("w", encoding="utf-8") as handle:
        json.dump(final_data, handle, indent=2, ensure_ascii=False)

    avg_return = sum(trajectory_returns) / len(trajectory_returns)
    print(f"Saved critic rewards to {optimization_path}")
    print(f"Scored {len(final_data)} responses, average raw critic return={avg_return:.4f}")


if __name__ == "__main__":
    main()
