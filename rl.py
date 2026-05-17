import os
import sys
import subprocess
from pathlib import Path
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def _select_accel_config(config) -> str:
    """Return the accelerate config path, auto-detecting GPU count if needed."""
    try:
        import torch
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        n_gpus = 1

    explicit = str(OmegaConf.select(config, "experiment.deepspeed_file", default="auto"))
    if explicit and explicit != "auto":
        candidate = Path("accelerate_configs") / f"{explicit}.yaml"
        if candidate.exists():
            accel_cfg = OmegaConf.load(candidate)
            cfg_procs = int(accel_cfg.get("num_processes", 0))
            if cfg_procs == 0 or cfg_procs == n_gpus:
                return str(candidate)
            print(
                f"[config] deepspeed_file={explicit!r} expects {cfg_procs} process(es) "
                f"but {n_gpus} GPU(s) visible — auto-selecting instead"
            )

    for stage in [3, 2]:
        candidate = Path("accelerate_configs") / f"1_node_{n_gpus}_gpus_deepspeed_zero{stage}.yaml"
        if candidate.exists():
            print(f"[config] Auto-selected accelerate config: {candidate.name}")
            return str(candidate)

    available = sorted(p.name for p in Path("accelerate_configs").glob("*.yaml"))
    raise FileNotFoundError(
        f"No accelerate config for {n_gpus} GPU(s). "
        f"Create accelerate_configs/1_node_{n_gpus}_gpus_deepspeed_zero3.yaml\n"
        f"Available: {available}"
    )

if __name__ == "__main__":
    config = get_config()

    start_from_scratch = config.experiment.start_from_scratch
    project_name = config.experiment.project
    model_base = config.model.model_base

    from omegaconf import MISSING
    if OmegaConf.select(config, "model.value_base_model", default=MISSING) is not MISSING:
        have_value_model = True
    else:
        have_value_model = False

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
    
    def init_value_model(i, cfg):
        project_name = cfg.experiment.project
        subprocess.run(
            f'python init_sdar_value_model.py '
            f'config=../configs/{project_name}.yaml '
            f'experiment.current_epoch={i} ',
            shell=True,
            cwd='train',
            check=True,
        )
    
    if start_from_scratch:
        os.makedirs(f"{project_name}/results", exist_ok=True)
        optimized_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
        begin_with(f"{project_name}/results/results-rl-" + optimized_model.replace("/", ".") + "-" + config.dataset.train_dataset + ".txt")
        begin_with(f"{project_name}/results/results-eval-" + optimized_model.replace("/", ".") + "-" + config.dataset.train_dataset + ".txt")
        if have_value_model:
            init_value_model(1, config)
            optimized_value_model = "../" + project_name + "/ckpt/" + config.model.optimized_value_name
            begin_with(f"{project_name}/results/results-rl-" + optimized_value_model.replace("/", ".") + "-" + config.dataset.train_dataset + ".txt")
    
    def sample(i, type, block_size = None, top_k = None, remasking_strategy = None):
        if model_base == "dream":
            script_name = "dream_rl_rollout.py"
        elif model_base == "llada" or model_base == "mmada":
            script_name = "llada_rl_rollout.py"
        elif model_base == "sdar":
            script_name = "sdar_rl_rollout.py"
        elif model_base == "trado":
            script_name = "trado_rl_rollout.py"
        subprocess.run(
            f'python {script_name} '
            f'config=../configs/{project_name}.yaml '
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
            f'experiment.current_epoch={i} ',
            shell=True,
            cwd='sample',
            check=True,
        )
    
    def reward(i, type, is_code_task, block_size = None, top_k = None, remasking_strategy = None):
        if is_code_task:
            script_name = "rl_code_reward.py"
        else:
            script_name = "rl_reward.py"
        subprocess.run(
            f'python {script_name} '
            f'config=../configs/{project_name}.yaml '
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
            f'experiment.current_epoch={i} ',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    
    def process_reward(i):
        cfg_i = f"config=../configs/{project_name}.yaml"
        ep    = f"experiment.current_epoch={i}"

        base = ["conda", "run", "-n", "CURE2", "--no-capture-output", "python", "-u"]

        subprocess.run(base + ["rl_process_divide_data.py", cfg_i, ep], cwd="reward", check=True)
        subprocess.run(base + ["llm_process_reward.py",    cfg_i, ep], cwd="sample", check=True)
        subprocess.run(base + ["rl_process_reward.py",     cfg_i, ep], cwd="reward", check=True)
    
    def execute(i, type):
        subprocess.run(
            f"python rl_execute.py "
            f"config=../configs/{project_name}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={i} ",
            shell=True,
            cwd='reward',
            check=True,
        )
            
    
    def train(i, target = None):
        if target is None:
            if model_base == "dream":
                script_name = "rl_dream.py"
            elif model_base == "llada":
                script_name = "rl_llada.py"
            elif model_base == "mmada":
                script_name = "rl_mmada.py"
            elif model_base == "sdar":
                script_name = "rl_sdar.py"
            elif model_base == "trado":
                script_name = "rl_trado.py"
        elif target == "policy":
            if model_base == "sdar":
                script_name = "train_sdar_policy.py"
            elif model_base == "trado":
                script_name = "train_trado_policy.py"
        elif target == "value":
            if model_base == "sdar":
                script_name = "train_sdar_value.py"
            elif model_base == "trado":
                script_name = "train_trado_value.py"
        accel_cfg = _select_accel_config(config)
        subprocess.run(
            f'accelerate launch '
            f'--num_machines 1 '
            f'--machine_rank 0 '
            f'--main_process_ip 127.0.0.1 '
            f'--main_process_port 8888 '
            f'--config_file {accel_cfg} '
            f'train/{script_name} '
            f'config=configs/{project_name}.yaml '
            f'experiment.current_epoch={i} ',
            shell=True,
            check=True,
        )
    
    if config.dataset.data_type == "code":
        is_code_task = True
    else:
        is_code_task = False
    
    if OmegaConf.select(config, "model.process_reward_model", default=MISSING) is not MISSING and config.model.process_reward_model is not None:
        is_process_reward = True
    else:
        is_process_reward = False

    i = config.experiment.current_epoch

    while i <= config.experiment.total_step:
        
        
        sample(i, "train")
        if is_code_task:
            execute(i, "train")
        
        if is_process_reward:
            process_reward(i)
        else:
            reward(i, "train", is_code_task)

        if have_value_model:
            train(i, target = "value")
            train(i, target = "policy")
        else:
            train(i, target = None)

        if i % config.experiment.eval_every == 0:
            if model_base == "sdar":
                remasking_strategy_list = config.evaluation.remasking_strategy
                top_k_list = config.evaluation.top_k
                block_size = config.evaluation.block_size
                for j in range(len(remasking_strategy_list)):
                    remasking_strategy = remasking_strategy_list[j]
                    top_k = top_k_list[j]
                    sample(i, "evaluation", block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
                    if is_code_task:
                        execute(i, "evaluation")
                    reward(i, "evaluation", is_code_task, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
            else:
                block_size_list = config.evaluation.block_size
                remasking_strategy_list = config.evaluation.remasking_strategy
                if OmegaConf.select(config, "evaluation.top_k", default=MISSING) is not MISSING:
                    top_k = config.evaluation.top_k
                else:
                    top_k = None
                for j in range(len(remasking_strategy_list)):
                    remasking_strategy = remasking_strategy_list[j]
                    if model_base == "dream":
                        block_size = block_size_list[j]
                    elif model_base == "llada" or model_base == "mmada":
                        block_size = config.evaluation.block_size
                    sample(i, "evaluation", block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
                    if is_code_task:
                        execute(i, "evaluation")
                    reward(i, "evaluation", is_code_task, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)

        i += 1



