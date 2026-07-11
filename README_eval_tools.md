# Evaluation & Monitoring Tools

Two standalone diagnostic scripts for tracking training health and establishing
baseline performance. Neither script modifies training behaviour — they only
observe and report.

- `monitor_training.py` — logs critic loss and policy reward during training, and plots them.
- `eval_base_model.py` — evaluates the unmodified base TraDo-4B model to establish the anchor baseline.

---

## 1. `monitor_training.py`

Tracks whether learning is actually happening by logging two quantities across
epochs and plotting them: critic BCE loss and policy reward statistics.

### Setup — add two logging calls

Drop `monitor_training.py` into the project root, then add one logging call to
each of two training scripts.

**In `train/train_trado_critic.py`**, inside the critic training loop, after `loss.backward()`:

```python
from monitor_training import log_critic_loss
log_critic_loss(
    epoch=current_epoch,
    step=step,
    loss=loss.item(),
    project_dir=config.experiment.project_dir,
)
```

**In `reward/rl_critic_reward.py`**, after the `output_data` list is built:

```python
from monitor_training import log_policy_reward
log_policy_reward(
    epoch=current_epoch,
    reward_data=output_data,       # list of dicts, each with a "reward" field
    project_dir=config.experiment.project_dir,
)
```

These append to JSON-lines logs under `<project_dir>/logs/`.

### Generate plots

After a few epochs have run:

```bash
python monitor_training.py --project_dir <path_to_experiment>
```

Optionally include the pass@1 plot from eval result files:

```bash
python monitor_training.py \
    --project_dir <path_to_experiment> \
    --results_dir <path_to_experiment>/results
```

Plots are written to `<project_dir>/logs/` as PNG files.

### How to read the output

**Critic loss** should start near `0.693` (a random binary classifier) and drop
below `0.4` within each epoch. If it stays at `0.693`, the critic is not
distinguishing expert from policy prefixes. If it is near zero from the first
step, the labels are likely all one class — a data-pipeline bug.

**Policy reward** — the mean return should trend upward across epochs. The
fraction of responses with positive return should stay near `0.5` (a property of
the z-score normalization). If the standard deviation of returns is near zero,
the critic is not discriminating between responses and the reward signal is dead.

---

## 2. `eval_base_model.py`

Evaluates the **original, untrained** TraDo-4B weights on the eval split, giving
the anchor number that every trained checkpoint must beat. Without this baseline,
a trained checkpoint's accuracy cannot be interpreted.

It reuses the existing pipeline (rollout → execute → score) exactly as the
training loop does at eval time, but points at the base weights with no critic
and no training.

### Run

```bash
python eval_base_model.py \
    config=configs/rl_trado_critic_mbpp_linux.yaml \
    model.pretrained_model=<PATH_TO_ORIGINAL_TraDo-4B_WEIGHTS> \
    experiment.project_name=BASELINE_trado4b_base \
    experiment.project_dir=./baseline_eval \
    dataset.eval_dataset=<SAME_DATASET_AS_TRAINING> \
    rollout.num_response_per_task=8 \
    rollout.max_token=2000 \
    training.max_gen_length=2000
```

### Two critical settings

- Point `model.pretrained_model` at the **original** TraDo-4B weights — **not**
  any `ckpt/optimized_*` folder.
- Set `max_token` and `max_gen_length` to **2000, not 512**, so the baseline
  reflects the model's true ceiling without truncating longer solutions.

Run the baseline on the **same** `eval_dataset` that training reports accuracy on,
so the numbers are directly comparable.

### Note on CLI flags

The three pipeline stages are invoked as subprocesses. If your repo uses a
different flag than `split=eval` to select the eval split, adjust the
`run_python(...)` calls near the bottom of `main()`. The script prints each
subprocess command before running it, so any mismatch is easy to spot.

### Output

Prints pass@1 (and pass@k where available) and writes a results file under
`<project_dir>/results/`. If the results file is not produced, the script falls
back to computing pass@k directly from the rollout JSON using the standard
unbiased estimator `pass@k = 1 - C(n-c, k) / C(n, k)`.

---

## Recommended workflow

1. Run `eval_base_model.py` once on the base weights — record the anchor pass@1.
2. Add the two logging calls and start/continue training.
3. Run `monitor_training.py` periodically to confirm the critic is learning and
   the policy reward is trending up.
4. Compare each trained checkpoint's pass@1 against the base anchor from step 1.
