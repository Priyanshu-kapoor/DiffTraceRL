"""
Initialize or refresh the standalone Trado critic checkpoint.

This mirrors the existing value-model bootstrap script on purpose so the new
critic path is easy to understand. The only extra behavior is that we preserve
an existing `critic_head` if it already exists while refreshing the shared
backbone weights from the latest policy checkpoint.
"""

import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from models import SDARForCausalLM
from train.init_trado_value_model import save_checkpoint


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def _get_critic_model(base_llm_class, critic_head_prefix="critic_head"):
    class CriticModel(base_llm_class):
        def __init__(self, config: AutoConfig):
            try:
                import deepspeed

                zero_off = deepspeed.zero.Init(enabled=False)
            except Exception:
                from contextlib import nullcontext

                zero_off = nullcontext()

            with zero_off:
                super().__init__(config)

            self.critic_head_prefix = critic_head_prefix
            critic_head = nn.Linear(config.hidden_size, 1, bias=False)
            setattr(self, critic_head_prefix, critic_head)

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, visible_tail_mask=None, **kwargs):
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            hidden = outputs["last_hidden_state"]

            if visible_tail_mask is None:
                return hidden

            tail_hidden = hidden[:, -visible_tail_mask.shape[1] :, :]
            mask = visible_tail_mask.to(tail_hidden.device).unsqueeze(-1).to(tail_hidden.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (tail_hidden * mask).sum(dim=1) / denom
            return getattr(self, self.critic_head_prefix)(pooled).squeeze(-1)

    return CriticModel


def policy_source_path(config) -> str:
    if config.experiment.current_epoch == 1:
        return config.model.pretrained_model
    return "../" + config.experiment.project + "/ckpt/" + config.model.optimized_name


def main():
    config = get_config()
    critic_model_class = _get_critic_model(SDARForCausalLM, "critic_head")

    source_path = policy_source_path(config)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    critic_save_dir = os.path.join(_repo_root, config.experiment.project, "ckpt", config.model.optimized_critic_name)

    tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)

    # Start from the existing critic checkpoint if it exists so we keep the
    # learned critic head. Otherwise bootstrap from the current policy model.
    if os.path.isdir(critic_save_dir):
        critic_model = critic_model_class.from_pretrained(
            critic_save_dir,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    else:
        critic_model = critic_model_class.from_pretrained(
            source_path,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        with torch.no_grad():
            getattr(critic_model, "critic_head").weight.zero_()

    # Load the latest policy checkpoint separately, then copy only the shared
    # backbone weights into the critic model. The critic head is left untouched.
    source_model = SDARForCausalLM.from_pretrained(
        source_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    critic_state = critic_model.state_dict()
    source_state = source_model.state_dict()
    for name, tensor in source_state.items():
        if name in critic_state:
            critic_state[name] = tensor

    critic_model.load_state_dict(critic_state, strict=False)
    save_checkpoint(critic_model, tokenizer, config, config.model.optimized_critic_name)


if __name__ == "__main__":
    main()
