"""
Train the standalone Trado critic on expert prefixes vs policy prefixes.

This script is intentionally simple:
- plain PyTorch instead of a second accelerate/deepspeed stack
- explicit replay-buffer update at the top
- explicit prefix batch construction inside the training loop

The goal is readability. Every major step is commented so the flow can be
understood without chasing helper layers.
"""

import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from models import SDARForCausalLM
from trado_critic_utils import (
    align_step_map,
    build_tok_idx_ext,
    make_basic_block_attention,
    process_pad,
    resolve_repo_path,
    rollout_outputs_path,
)
from train.init_trado_critic_model import _get_critic_model
from train.init_trado_value_model import save_checkpoint
from train.prompting_utils import UniversalPrompting


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def load_rollout_policy_samples(outputs_path: Path) -> list[dict]:
    """
    Flatten rollout outputs into simple `{prompt, response, step_map}` samples.
    """
    with outputs_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    samples = []
    for item in data:
        prompt = item["prompt"]
        full_outputs = item.get("full_output", [])
        step_maps = item.get("step_map", [])
        for response, step_map in zip(full_outputs, step_maps):
            if response is None or response == "":
                continue
            samples.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "step_map": step_map,
                }
            )
    return samples


def update_replay_buffer(config, new_samples: list[dict]) -> list[dict]:
    """
    Keep the replay buffer as a plain FIFO list saved under project/temp_data.
    """
    buffer_path = resolve_repo_path(config.critic.buffer_path)
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    if buffer_path.exists():
        payload = torch.load(buffer_path, map_location="cpu")
        buffer = list(payload.get("samples", []))
    else:
        buffer = []

    buffer.extend(new_samples)
    max_size = int(config.critic.buffer_size)
    if len(buffer) > max_size:
        buffer = buffer[-max_size:]

    torch.save({"samples": buffer}, buffer_path)
    return buffer


def load_expert_samples(config) -> list[dict]:
    cache_path = resolve_repo_path(config.critic.expert_cache_path)
    payload = torch.load(cache_path, map_location="cpu")
    records = list(payload["records"])
    return [
        {
            "prompt": item["prompt"],
            "response": item["response"],
            "step_map": item["trado_step_map"],
        }
        for item in records
    ]


def build_prefix_descriptors(samples: list[dict]) -> list[dict]:
    """
    Expand each response into one training item per reveal step.
    """
    descriptors = []
    for sample in samples:
        raw_step_map = sample["step_map"]
        if raw_step_map is None or raw_step_map == "":
            continue
        steps = sorted(set(int(x) for x in raw_step_map))
        for step in steps:
            descriptors.append(
                {
                    "prompt": sample["prompt"],
                    "response": sample["response"],
                    "step_map": raw_step_map,
                    "prefix_step": step,
                }
            )
    return descriptors


def choose_batch(items: list[dict], count: int) -> list[dict]:
    if not items:
        return []
    if len(items) >= count:
        return random.sample(items, count)
    return random.choices(items, k=count)


def build_training_batch(descriptors: list[dict], tokenizer, uni_prompting, block_size: int):
    """
    Tokenize prompt/response pairs and then build the partial-prefix state for
    the requested reveal step.

    The resulting tensors mirror the existing Trado prefix representation:
    - first half: prompt + full response
    - second half: duplicated response with unrevealed tokens masked out
    """
    prompt_list = [item["prompt"] for item in descriptors]
    response_list = [item["response"] for item in descriptors]
    prefix_steps = [int(item["prefix_step"]) for item in descriptors]
    step_maps = [item["step_map"] for item in descriptors]

    input_ids, _, start_pos, _ = uni_prompting((prompt_list, response_list))
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    resp_len = input_ids.shape[1] - start_pos

    tok_idx_ext = build_tok_idx_ext(input_ids, start_pos, pad_id)
    extended_input_ids = []
    visible_tail_masks = []

    for row_index in range(input_ids.shape[0]):
        sample_ids = input_ids[row_index]
        resp_tokens = sample_ids[start_pos:].clone()
        valid_resp = resp_tokens.ne(pad_id)
        step_map = torch.tensor(align_step_map(step_maps[row_index], resp_len), dtype=torch.long)

        # Visible prefix after reveal step `prefix_steps[row_index]`.
        visible_tail = valid_resp & step_map.le(prefix_steps[row_index])

        tail = resp_tokens.clone()
        tail[~visible_tail] = mask_id

        extended = torch.empty(sample_ids.shape[0] + resp_len, dtype=sample_ids.dtype)
        extended[: sample_ids.shape[0]] = sample_ids
        extended[sample_ids.shape[0] :] = tail

        extended_input_ids.append(extended)
        visible_tail_masks.append(visible_tail)

    extended_input_ids = torch.stack(extended_input_ids, dim=0)
    visible_tail_masks = torch.stack(visible_tail_masks, dim=0)

    # Reuse the same block-attention pattern as the Trado training scripts.
    total_len = extended_input_ids.shape[1]
    base_attention = make_basic_block_attention(total_len, start_pos, block_size)
    attention_mask = base_attention.repeat_interleave(extended_input_ids.shape[0], dim=0)
    attention_mask = process_pad(attention_mask, extended_input_ids, start_pos, pad_id)

    return {
        "extended_input_ids": extended_input_ids,
        "attention_mask": attention_mask,
        "tok_idx_ext": tok_idx_ext,
        "visible_tail_masks": visible_tail_masks,
    }


def main():
    config = get_config()
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    outputs_path = rollout_outputs_path(config, "train")
    policy_samples = load_rollout_policy_samples(outputs_path)
    replay_buffer = update_replay_buffer(config, policy_samples)
    expert_samples = load_expert_samples(config)

    expert_prefixes = build_prefix_descriptors(expert_samples)
    policy_prefixes = build_prefix_descriptors(replay_buffer)
    if not expert_prefixes:
        raise ValueError("No expert prefixes were built from the expert cache")
    if not policy_prefixes:
        raise ValueError("No policy prefixes were available in the replay buffer")

    project_name = config.experiment.project
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    critic_checkpoint = os.path.join(_repo_root, project_name, "ckpt", config.model.optimized_critic_name)
    tokenizer = AutoTokenizer.from_pretrained(critic_checkpoint, trust_remote_code=True)
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_prompt_len=config.training.max_prompt_len,
        max_gen_length=config.training.max_gen_length,
        ignore_id=-100,
    )

    critic_model_class = _get_critic_model(SDARForCausalLM, "critic_head")
    model = critic_model_class.from_pretrained(critic_checkpoint, trust_remote_code=True, torch_dtype="auto")

    # V1 deliberately freezes the shared backbone so the critic path is easy to
    # reason about and does not interfere with policy optimization internals.
    if config.critic.freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("critic_head")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config.critic.learning_rate),
    )

    half_batch = max(1, int(config.critic.batch_size) // 2)
    num_steps = int(config.critic.num_steps_per_epoch)
    running_loss = 0.0

    for step_index in range(num_steps):
        batch_descriptors = choose_batch(expert_prefixes, half_batch) + choose_batch(policy_prefixes, half_batch)
        labels = torch.tensor(
            [1.0] * half_batch + [0.0] * half_batch,
            dtype=torch.float32,
            device=device,
        )

        batch = build_training_batch(batch_descriptors, tokenizer, uni_prompting, config.training.block_size)
        extended_input_ids = batch["extended_input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tok_idx_ext = batch["tok_idx_ext"].to(device)
        visible_tail_masks = batch["visible_tail_masks"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=extended_input_ids,
            attention_mask=attention_mask,
            position_ids=tok_idx_ext,
            visible_tail_mask=visible_tail_masks,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        if (step_index + 1) % 10 == 0 or step_index == num_steps - 1:
            avg_loss = running_loss / (step_index + 1)
            print(
                f"[critic] step {step_index + 1}/{num_steps} "
                f"loss={loss.item():.4f} avg_loss={avg_loss:.4f}"
            )

    save_checkpoint(model, tokenizer, config, config.model.optimized_critic_name)


if __name__ == "__main__":
    main()
