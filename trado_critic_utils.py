"""
Utilities shared by the standalone Trado critic pipeline.

This file intentionally keeps the shared logic very small:

1. Resolve repo paths and workspace-level `ai-project/data` paths.
2. Read and write JSONL files used by the new dataset-prep scripts.
3. Reproduce the exact code-task prompt shapes used by the existing rollout code.
4. Recreate the same Trado block-attention / position-id logic used by the
   existing value/policy training scripts.
5. Turn a response-level `step_map` into critic prefix states.

The original pipeline files are left untouched. These helpers only support the
new critic path and the new dataset-prep scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent.parent
WORKSPACE_DATA_ROOT = WORKSPACE_ROOT / "data"


def resolve_repo_path(path_str: str) -> Path:
    """Resolve config paths relative to the dLLM-RL repo root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_workspace_data_path(path_str: str) -> Path:
    """
    Resolve paths relative to the workspace-level `ai-project/data` layout.

    The new professor-facing workflow stores raw datasets, cleaned JSONL files,
    entropy scores, and masking trajectories under the workspace root rather
    than inside the git repo itself.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of Python dicts."""
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    """Write a list of dicts to JSONL using UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def rollout_model_ref(config) -> str:
    """
    Reproduce the naming logic used by the existing rollout/eval scripts.

    The rollout output filename embeds the model path string, not the actual
    absolute path on disk. We must mirror that behavior exactly so the critic
    scripts open the same temp files the existing sampler created.
    """
    if config.experiment.current_epoch == 1:
        return config.model.pretrained_model
    return "../" + config.experiment.project + "/ckpt/" + config.model.optimized_name


def rollout_outputs_name(config, function_name: str) -> str:
    model_ref = rollout_model_ref(config)
    if function_name == "train":
        dataset = config.dataset.train_dataset
        prefix = "rl"
    elif function_name == "evaluation":
        dataset = config.evaluation.eval_dataset
        prefix = "eval"
    else:
        raise ValueError(f"Unsupported function_name: {function_name}")
    return f"{prefix}-{model_ref.replace('/', '.')}-{dataset}"


def rollout_outputs_path(config, function_name: str) -> Path:
    outputs_name = rollout_outputs_name(config, function_name)
    return REPO_ROOT / config.experiment.project / "temp_data" / f"outputs-{outputs_name}.json"


def format_function_prompt(question: str, prefix: str = "") -> str:
    """
    Match the function-style code prompt used by `sample/trado_rl_rollout.py`.

    For HumanEval-style tasks, `prefix` already contains the opening code block
    plus the fixed scaffold that the model must continue from.
    """
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "Place your code within a single Python code block ```python ```. "
        "Do not include more than one code block. <|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{prefix}"
    )


def format_stdio_prompt(question: str) -> str:
    """
    Match the stdio-style code prompt used by `sample/trado_rl_rollout.py`.
    """
    return (
        "<|im_start|>user\n"
        "This is the problem:\n"
        f"{question}\n"
        "You should put your code in ```python ```. "
        "Use input() to read input and print() to produce output in your script. "
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_rollout_prompt(question: str, test_method: str, prefix: str = "") -> str:
    """Render the exact prompt string the rollout script uses for code tasks."""
    if test_method == "function":
        return format_function_prompt(question, prefix)
    if test_method == "stdio":
        return format_stdio_prompt(question)
    raise ValueError(f"Unsupported test_method: {test_method}")


def format_code_response(trajectory_code: str, response_prefix: str, response_suffix: str) -> str:
    """
    Build the assistant response text that the critic should score.

    `trajectory_code` is the model-generated portion whose reveal order comes
    from GPT-2 entropy ranking. The optional prefix/suffix let us represent both:
    - MBPP / CodeContests: a full fenced code block
    - HumanEval: only the generated completion plus the closing fence, because
      the opening fence and scaffold already live inside `prefix`
    """
    stripped = trajectory_code.rstrip("\n")
    return f"{response_prefix}{stripped}{response_suffix}"


def format_mbpp_prompt(question: str) -> str:
    """Backwards-compatible wrapper for the original MBPP-only helper."""
    return format_function_prompt(question, "")


def format_mbpp_response(reference: str) -> str:
    """Backwards-compatible wrapper for the original MBPP-only helper."""
    return format_code_response(reference, "```python\n", "\n```")


def align_step_map(step_map: Iterable[int] | str | None, target_len: int) -> list[int]:
    """
    Trim or extend a step map to the target token length.

    Existing Trado training code already tolerates mild mismatches by trimming
    or padding step ids. The critic path follows the same idea, but does it
    explicitly so the reward vectors we emit already match the tokenized length.
    """
    if step_map is None or step_map == "":
        values = list(range(target_len))
    else:
        values = [int(x) for x in step_map]

    if not values:
        values = list(range(target_len))

    if len(values) > target_len:
        values = values[:target_len]
    elif len(values) < target_len:
        fill_value = max(values) + 1 if values else 0
        values = values + [fill_value] * (target_len - len(values))

    return values


def z_score_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    if std == 0:
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]


def build_tok_idx_ext(input_ids: torch.Tensor, start_pos: int, pad_id: int) -> torch.Tensor:
    """
    Recreate the same token-position ids used by the existing Trado scripts.

    `input_ids` has shape (B, L0 + L1).
    The returned tensor has shape (B, L0 + 2*L1) because the response segment is
    duplicated in the critic prefix representation.
    """
    _, seq_len = input_ids.shape
    idx = torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)
    valid = (idx >= start_pos) | input_ids.ne(pad_id)
    tok_idx = valid.long().cumsum(dim=-1) - 1
    tok_idx = tok_idx.masked_fill(~valid, 1)
    tok_idx_resp = tok_idx[:, start_pos:]
    return torch.cat([tok_idx, tok_idx_resp], dim=1)


def make_basic_block_attention(total_len: int, start_pos: int, block_size: int) -> torch.Tensor:
    """
    Copy the block-attention construction from the existing Trado training code.

    `total_len` must be `L0 + 2*L1`. The first half is prompt + full response;
    the second half is the masked response copy that represents a partial prefix.
    """
    batch = 1
    prompt_len = start_pos
    resp_len = (total_len - prompt_len) // 2
    assert prompt_len + 2 * resp_len == total_len, "Expected total_len = L0 + 2*L1"

    bias = torch.full((batch, 1, total_len, total_len), 0)

    rows_tail = torch.arange(prompt_len + resp_len, prompt_len + 2 * resp_len)
    rows_resp = torch.arange(prompt_len, prompt_len + resp_len)

    for block_idx in range((resp_len + block_size - 1) // block_size):
        left_end = prompt_len + min(block_idx * block_size, resp_len)
        right_start = prompt_len + resp_len + (left_end - prompt_len)

        i_start = block_idx * block_size
        i_end = min((block_idx + 1) * block_size, resp_len)

        block_rows_tail = rows_tail[i_start:i_end]
        bias[:, :, block_rows_tail.unsqueeze(-1), 0:left_end] = 1
        bias[:, :, block_rows_tail.unsqueeze(-1), right_start:(right_start + block_size)] = 1

        block_rows_resp = rows_resp[i_start:i_end]
        left_end = prompt_len + min((block_idx + 1) * block_size, resp_len)
        bias[:, :, block_rows_resp.unsqueeze(-1), 0:left_end] = 1

    if prompt_len > 0:
        prompt_blocks = (prompt_len + block_size - 1) // block_size
        for block_idx in range(prompt_blocks):
            row_end = max(prompt_len - block_idx * block_size, 0)
            row_start = max(prompt_len - (block_idx + 1) * block_size, 0)
            if row_end > row_start:
                block_rows = torch.arange(row_start, row_end)
                bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1

    return bias


def process_pad(attn: torch.Tensor, input_ids: torch.Tensor, start_pos: int, pad_id: int) -> torch.Tensor:
    """
    Copy the padding guard from the existing Trado scripts.

    The original code zeroes attention into prompt-padding columns and fixes rows
    that would otherwise become all-zero.
    """
    total_len = input_ids.shape[1]
    device = input_ids.device

    cols = torch.arange(total_len, device=device)
    key_mask = (cols < start_pos).unsqueeze(0) & input_ids.eq(pad_id)
    attn.masked_fill_(key_mask[:, None, None, :], 0)

    base = attn[:, 0]
    bad = (base.sum(dim=-1) == 0) & (torch.arange(base.size(1), device=device).unsqueeze(0) < start_pos)
    bad_batch, bad_rows = bad.nonzero(as_tuple=True)
    base[bad_batch, bad_rows, :] = 0
    base[bad_batch, bad_rows, bad_rows] = 1
    return attn


def build_prefix_examples(
    input_ids: torch.Tensor,
    start_pos: int,
    pad_id: int,
    mask_id: int,
    step_maps: list[list[int]],
) -> list[dict]:
    """
    Build critic prefix states in the same extended format as existing Trado code.

    Each returned example represents the state *after* revealing all tokens with
    `step_id <= current_step`.

    Output fields:
    - extended_input_ids: (L0 + 2*L1,)
    - visible_tail_mask: (L1,) marks visible tokens in the duplicated tail copy
    - tok_idx_ext: (L0 + 2*L1,)
    - sample_index: which response this prefix came from
    - step: reveal step id for this prefix
    """
    prompt_plus_resp = input_ids
    tok_idx_ext = build_tok_idx_ext(prompt_plus_resp, start_pos, pad_id)
    resp_len = prompt_plus_resp.shape[1] - start_pos
    examples: list[dict] = []

    for sample_index in range(prompt_plus_resp.shape[0]):
        sample_ids = prompt_plus_resp[sample_index]
        resp_tokens = sample_ids[start_pos:].clone()
        valid_resp = resp_tokens.ne(pad_id)
        step_map = torch.tensor(align_step_map(step_maps[sample_index], resp_len), dtype=torch.long)

        if valid_resp.any():
            unique_steps = sorted(set(step_map[valid_resp].tolist()))
        else:
            unique_steps = []

        for step in unique_steps:
            visible_tail_mask = valid_resp & step_map.le(step)
            tail = resp_tokens.clone()
            tail[~visible_tail_mask] = mask_id

            extended_input_ids = torch.empty(sample_ids.shape[0] + resp_len, dtype=sample_ids.dtype)
            extended_input_ids[: sample_ids.shape[0]] = sample_ids
            extended_input_ids[sample_ids.shape[0] :] = tail

            examples.append(
                {
                    "extended_input_ids": extended_input_ids,
                    "visible_tail_mask": visible_tail_mask,
                    "tok_idx_ext": tok_idx_ext[sample_index],
                    "sample_index": sample_index,
                    "step": step,
                }
            )

    return examples
