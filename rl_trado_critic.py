"""
Standalone Trado critic pipeline.

This file intentionally does not modify the existing `rl.py` / `rl_v.py`
drivers. It adds a new, isolated path that:

1. Builds a repo-local rollout/eval dataset JSON from a normalized root-level
   JSONL file under `ai-project/data`.
2. Builds a Trado-aligned expert trajectory cache from GPT-2 masking trajectories.
3. Runs the existing Trado rollout and execution scripts unchanged.
4. Trains a standalone critic on expert prefixes vs policy prefixes.
5. Generates dense vector rewards for the existing Trado value/policy trainers.
6. Reuses the existing evaluation path with execution-based code rewards.

The code stays explicit on purpose. The goal is that each subprocess step is
easy to read and easy to map back to the original plan.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def resolve_from_repo(path_str: str) -> Path:
    """Resolve config paths relative to the dLLM-RL repo root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def config_path_for_child(config, child_cwd: Path) -> str:
    """
    Reuse the same config file for every subprocess.

    Child scripts run from subdirectories like `train/` or `reward/`, so the
    config path must be rewritten relative to that child working directory.
    """
    config_path = resolve_from_repo(str(config.config))
    relative = os.path.relpath(config_path, child_cwd)
    return f"config={relative}"


def cli_value(value) -> str:
    """Serialize simple override values for OmegaConf CLI parsing."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def begin_with(path: Path) -> None:
    """Create or truncate a results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                import json

                items.append(json.loads(line))
    return items


def normalized_data_path(config) -> Path:
    """Prefer the new normalized-data key while keeping the legacy fallback."""
    path_str = config.critic.get("normalized_data_path")
    if path_str is None:
        path_str = config.critic.get("reference_data_path")
    if path_str is None:
        raise ValueError("critic.normalized_data_path or critic.reference_data_path must be configured")
    return resolve_from_repo(path_str)


def accelerate_config_path(config) -> Path:
    return REPO_ROOT / "accelerate_configs" / f"{config.experiment.deepspeed_file}.yaml"


def is_placeholder_model_ref(value: str | None) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text == "/abs/path/of/model" or "abs/path/of/model" in text


def visible_gpu_count() -> int:
    import torch

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible = [item.strip() for item in cvd.split(",") if item.strip() != ""]
        return len(visible)
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def warn_or_raise(config, message: str) -> None:
    if bool(config.experiment.get("preflight_only", False)):
        print(f"[preflight warning] {message}")
        return
    raise ValueError(message)


def validate_model_references(config) -> None:
    pretrained_model = str(config.model.pretrained_model)
    value_base_model = str(config.model.value_base_model)

    if is_placeholder_model_ref(pretrained_model):
        raise ValueError("model.pretrained_model is still a placeholder. Set it to a real Trado model path or HF id.")
    if is_placeholder_model_ref(value_base_model):
        raise ValueError("model.value_base_model is still a placeholder. Set it to a real value-base model path or HF id.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True, use_fast=True)
    except Exception as exc:
        raise ValueError(f"Failed to load tokenizer from model.pretrained_model={pretrained_model!r}: {exc}") from exc

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("The tokenizer loaded from model.pretrained_model is not a fast tokenizer.")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer loaded from model.pretrained_model has no pad_token_id.")
    if tokenizer.mask_token_id is None:
        raise ValueError("The tokenizer loaded from model.pretrained_model has no mask_token_id.")

    try:
        AutoConfig.from_pretrained(value_base_model, trust_remote_code=True)
    except Exception as exc:
        raise ValueError(f"Failed to load config from model.value_base_model={value_base_model!r}: {exc}") from exc


def run_preflight_checks(config) -> None:
    validate_model_references(config)

    normalized_path = normalized_data_path(config)
    if not normalized_path.exists():
        raise ValueError(f"critic.normalized_data_path does not exist: {normalized_path}")

    trajectory_path = resolve_from_repo(str(config.critic.expert_trajectory_path))
    if Path(trajectory_path).name == "masking_trajectories.jsonl":
        raise ValueError(
            "critic.expert_trajectory_path points to the old generic masking file. "
            "Use the dataset-specific file such as mbpp_masking_trajectories.jsonl."
        )
    if not trajectory_path.exists():
        raise ValueError(f"critic.expert_trajectory_path does not exist: {trajectory_path}")

    accel_path = accelerate_config_path(config)
    if not accel_path.exists():
        raise ValueError(f"Accelerate config does not exist: {accel_path}")

    accel_cfg = OmegaConf.load(accel_path)
    required_processes = int(accel_cfg.get("num_processes", 0))
    gpu_count = visible_gpu_count()
    tp = int(config.rollout.tensor_parallel_size)

    if gpu_count < tp:
        warn_or_raise(
            config,
            f"Visible GPUs ({gpu_count}) are fewer than rollout.tensor_parallel_size ({tp}).",
        )
    if required_processes > 0 and gpu_count < required_processes:
        warn_or_raise(
            config,
            f"Visible GPUs ({gpu_count}) are fewer than accelerate num_processes ({required_processes}).",
        )

    print(f"[preflight] normalized_data_path={normalized_path}")
    print(f"[preflight] expert_trajectory_path={trajectory_path}")
    print(f"[preflight] accelerate_config={accel_path}")
    print(f"[preflight] visible_gpus={gpu_count} tensor_parallel_size={tp} accelerate_num_processes={required_processes}")


def write_runtime_config(config) -> Path:
    """
    Save the merged runtime config once and reuse it for all child subprocesses.

    This avoids the old bug where top-level CLI overrides were not seen by
    child scripts because they only reread the original YAML file.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="rl_trado_critic_"))
    runtime_config_path = temp_dir / "resolved_config.yaml"
    OmegaConf.save(config, runtime_config_path, resolve=True)
    return runtime_config_path


def build_repo_dataset_entry(item: dict) -> dict:
    """
    Convert one normalized critic-source record into the JSON format expected by the
    existing Trado rollout/execution scripts.

    Important fields:
    - `question`    : used to render the user prompt
    - `test_method` : tells `rl_execute.py` whether this is function or stdio
    - `prefix`      : function tasks may pre-fill the assistant with code prefix
    """
    repo_item = {
        "task_id": item["task_id"],
        "question": item["question"],
        "reference": item["reference_code"],
        "test_method": item["test_method"],
        "test_time_limit": int(item.get("test_time_limit", 1)),
        "prefix": item.get("prefix", ""),
    }
    if item["test_method"] == "function":
        repo_item["test_list"] = list(item["test_list"])
    elif item["test_method"] == "stdio":
        repo_item["test_input"] = list(item["test_input"])
        repo_item["test_output"] = list(item["test_output"])
    else:
        raise ValueError(f"Unsupported test_method: {item['test_method']}")
    return repo_item


def ensure_repo_dataset(config) -> None:
    """
    Materialize a repo-local dataset JSON from the normalized root-level JSONL.

    The existing sampler only knows how to read `../data/<dataset>.json` from
    inside `sample/`. Rather than changing that code, we create the expected
    JSON file(s) here for the new critic pipeline only.
    """
    source_path = normalized_data_path(config)
    items = load_jsonl(source_path)

    dataset = [build_repo_dataset_entry(item) for item in items]
    dataset.sort(key=lambda item: item["task_id"])

    dataset_names = {
        str(config.dataset.train_dataset),
        str(config.evaluation.eval_dataset),
    }
    for name in dataset_names:
        target_path = REPO_ROOT / "data" / f"{name}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            import json

            json.dump(dataset, handle, indent=2, ensure_ascii=False)


def init_value_model(config, epoch: int) -> None:
    run_python(
        config,
        "train",
        "init_trado_value_model.py",
        experiment__current_epoch=epoch,
    )


def build_expert_cache(config, epoch: int) -> None:
    run_python(
        config,
        "data",
        "build_trado_expert_trajectory_cache.py",
        experiment__current_epoch=epoch,
    )


def sample_train(config, epoch: int) -> None:
    run_python(
        config,
        "sample",
        "trado_rl_rollout.py",
        experiment__function="train",
        experiment__current_epoch=epoch,
    )


def execute_train(config, epoch: int) -> None:
    # The critic reward replaces train-time scalar reward, but we still run the
    # existing execution step so the rollout file keeps the same code-task fields
    # and can still be inspected with the normal executor outputs.
    run_python(
        config,
        "reward",
        "rl_execute.py",
        experiment__function="train",
        experiment__current_epoch=epoch,
    )


def init_critic(config, epoch: int) -> None:
    run_python(
        config,
        "train",
        "init_trado_critic_model.py",
        experiment__current_epoch=epoch,
    )


def train_critic(config, epoch: int) -> None:
    run_python(
        config,
        "train",
        "train_trado_critic.py",
        experiment__current_epoch=epoch,
    )


def generate_critic_reward(config, epoch: int) -> None:
    run_python(
        config,
        "reward",
        "rl_critic_reward.py",
        experiment__current_epoch=epoch,
    )


def train_value(config, epoch: int) -> None:
    run_accelerate(
        config,
        "train_trado_value.py",
        experiment__current_epoch=epoch,
    )


def train_policy(config, epoch: int) -> None:
    run_accelerate(
        config,
        "train_trado_policy.py",
        experiment__current_epoch=epoch,
    )


def evaluation_pairs(config) -> list[tuple[str, int]]:
    strategies = list(config.evaluation.remasking_strategy)
    top_k_values = list(config.evaluation.top_k)
    if len(strategies) != len(top_k_values):
        raise ValueError("evaluation.remasking_strategy and evaluation.top_k must have the same length")
    return list(zip(strategies, top_k_values))


def run_evaluation(config, epoch: int) -> None:
    for remasking_strategy, top_k in evaluation_pairs(config):
        overrides = {
            "experiment__function": "evaluation",
            "experiment__current_epoch": epoch,
            "evaluation__block_size": config.evaluation.block_size,
            "evaluation__top_k": top_k,
            "evaluation__remasking_strategy": remasking_strategy,
        }

        run_python(config, "sample", "trado_rl_rollout.py", **overrides)
        run_python(config, "reward", "rl_execute.py", experiment__function="evaluation", experiment__current_epoch=epoch)
        run_python(config, "reward", "rl_code_reward.py", **overrides)


def format_override_keys(overrides: dict) -> dict:
    """
    Convert Python-friendly helper keyword names into OmegaConf dot keys.

    `run_python(..., experiment__current_epoch=1)` is easier to read in Python
    code than repeatedly spelling raw `"experiment.current_epoch"` strings.
    """
    return {key.replace("__", "."): value for key, value in overrides.items()}


def run_python(config, subdir: str, script_name: str, **overrides) -> None:
    child_cwd = REPO_ROOT / subdir
    command = [sys.executable, script_name, config_path_for_child(config, child_cwd)]
    for key, value in format_override_keys(overrides).items():
        command.append(f"{key}={cli_value(value)}")
    subprocess.run(command, cwd=child_cwd, check=True)


def run_accelerate(config, script_name: str, **overrides) -> None:
    command = [
        "accelerate",
        "launch",
        "--num_machines",
        str(config.experiment.num_node),
        "--machine_rank",
        str(config.experiment.node_index),
        "--main_process_ip",
        "127.0.0.1",
        "--main_process_port",
        "8888",
        "--config_file",
        str(accelerate_config_path(config)),
        f"train/{script_name}",
        config_path_for_child(config, REPO_ROOT),
    ]
    for key, value in format_override_keys(overrides).items():
        command.append(f"{key}={cli_value(value)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def initialize_run_artifacts(config) -> None:
    """
    Create the project directories and truncate the most important result files.

    This matches the behavior of the original launchers when starting from
    scratch, but only for the new isolated critic project.
    """
    project_dir = REPO_ROOT / config.experiment.project
    (project_dir / "results").mkdir(parents=True, exist_ok=True)
    (project_dir / "temp_data").mkdir(parents=True, exist_ok=True)

    policy_ref = "../" + config.experiment.project + "/ckpt/" + config.model.optimized_name
    value_ref = "../" + config.experiment.project + "/ckpt/" + config.model.optimized_value_name

    begin_with(project_dir / "results" / f"results-rl-{policy_ref.replace('/', '.')}-{config.dataset.train_dataset}.txt")
    begin_with(project_dir / "results" / f"results-eval-{policy_ref.replace('/', '.')}-{config.evaluation.eval_dataset}.txt")
    begin_with(project_dir / "results" / f"results-rl-{value_ref.replace('/', '.')}-{config.dataset.train_dataset}.txt")


def main():
    config = get_config()

    if config.model.model_base != "trado":
        raise ValueError("This standalone critic pipeline only supports model_base='trado'")
    if config.dataset.data_type != "code":
        raise ValueError("This standalone critic pipeline only supports code datasets")

    run_preflight_checks(config)
    runtime_config_path = write_runtime_config(config)
    config.config = str(runtime_config_path)

    ensure_repo_dataset(config)
    build_expert_cache(config, int(config.experiment.current_epoch))

    if bool(config.experiment.get("preflight_only", False)):
        print(f"[critic-pipeline] preflight-only mode completed successfully with {runtime_config_path}")
        return

    if config.experiment.start_from_scratch:
        initialize_run_artifacts(config)
        init_value_model(config, 1)

    current_epoch = int(config.experiment.current_epoch)
    total_step = int(config.experiment.total_step)
    eval_every = int(config.experiment.eval_every)

    while current_epoch <= total_step:
        print(f"[critic-pipeline] epoch {current_epoch}/{total_step}")

        sample_train(config, current_epoch)
        execute_train(config, current_epoch)

        # The replay buffer update happens at the top of `train_trado_critic.py`.
        init_critic(config, current_epoch)
        train_critic(config, current_epoch)
        generate_critic_reward(config, current_epoch)

        train_value(config, current_epoch)
        train_policy(config, current_epoch)

        if current_epoch % eval_every == 0:
            run_evaluation(config, current_epoch)

        current_epoch += 1


if __name__ == "__main__":
    main()
