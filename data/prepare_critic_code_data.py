"""
Prepare normalized critic-source JSONL files for MBPP, HumanEval, and CodeContests.
"""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path
from datasets import load_from_disk

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trado_critic_utils import write_jsonl


# ---- PATH CONFIGURATION ----

REPO_DATA_DIR = Path(__file__).resolve().parent

# Use large storage for all datasets
WORKSPACE_DATA_ROOT = Path("/dcs/large/u5671205/data")


LANG_UNKNOWN = 0
LANG_PYTHON = 1
LANG_CPP = 2
LANG_PYTHON3 = 3
LANG_JAVA = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare normalized critic-source JSONL for code datasets"
    )

    parser.add_argument(
        "--dataset",
        choices=["mbpp", "humaneval", "codecontests"],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help=(
            "HuggingFace split to read (MBPP only). 'train' (374 tasks) is the "
            "RL training set; 'test' (500 tasks) is the standard held-out "
            "benchmark and should be used to build a separate eval file so "
            "evaluation is not measured on in-distribution training tasks."
        ),
    )

    parser.add_argument(
        "--data-root",
        default=str(WORKSPACE_DATA_ROOT),
        help="Dataset root directory (default: /dcs/large/u5671205/data)",
    )

    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional explicit output JSONL path",
    )

    parser.add_argument(
        "--pilot-size",
        type=int,
        default=1000,
        help="Pilot size for CodeContests after filtering",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10086,
        help="Random seed for CodeContests sampling",
    )

    return parser.parse_args()


def default_output_path(dataset: str, data_root: Path, pilot_size: int, split: str = "train") -> Path:
    if dataset == "mbpp":
        # "test" is MBPP's held-out benchmark split; name it "_eval" so config
        # files can point train_dataset and evaluation.eval_dataset at two
        # genuinely disjoint files instead of both loading mbpp_critic.jsonl.
        if split == "train":
            return data_root / "mbpp_critic_train.jsonl"
        if split == "test":
            return data_root / "mbpp_critic_eval.jsonl"
        return data_root / f"mbpp_critic_{split}.jsonl"
    if dataset == "humaneval":
        return data_root / "humaneval_critic.jsonl"
    return data_root / f"codecontests_pilot_{pilot_size}_critic.jsonl"


def prepare_mbpp(data_root: Path, split: str = "train") -> list[dict]:
    dataset = load_from_disk(str(data_root / "mbpp_dataset"))
    records = []

    for row in dataset[split]:
        records.append(
            {
                "source_dataset": "mbpp",
                "task_id": str(row["task_id"]),
                "question": (
                    "You are an expert Python programmer, and here is your task:\n"
                    + row["text"]
                    + "\nYour code should pass these tests:\n\n"
                    + "\n".join(row["test_list"])
                ),
                "reference_code": row["code"],
                "trajectory_code": row["code"],
                "test_method": "function",
                "prefix": "",
                "response_prefix": "```python\n",
                "response_suffix": "\n```",
                "test_time_limit": 1,
                "test_list": list(row["test_list"]),
            }
        )

    return records


def humaneval_question(prompt: str) -> str:
    return "You need to complete this code:\n" + prompt


def humaneval_prefix(prompt: str) -> str:
    return "Here is the completed code:\n```python\n" + prompt


def humaneval_test_entry(test: str, entry_point: str) -> str:
    return test.rstrip() + f"\n\ncheck({entry_point})"


def prepare_humaneval(data_root: Path) -> list[dict]:
    dataset = load_from_disk(str(data_root / "humaneval_dataset"))
    records = []

    for row in dataset["test"]:
        prompt = row["prompt"]
        completion = row["canonical_solution"]

        records.append(
            {
                "source_dataset": "humaneval",
                "task_id": str(row["task_id"]),
                "question": humaneval_question(prompt),
                "reference_code": prompt + completion,
                "trajectory_code": completion,
                "test_method": "function",
                "prefix": humaneval_prefix(prompt),
                "response_prefix": "",
                "response_suffix": "\n```",
                "test_time_limit": 1,
                "test_list": [
                    humaneval_test_entry(row["test"], row["entry_point"])
                ],
            }
        )

    return records


def looks_like_python3(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def pick_python_solution(language_ids: list[int], solutions: list[str]) -> str | None:

    for language_id, solution in zip(language_ids, solutions):
        if language_id == LANG_PYTHON3 and looks_like_python3(solution):
            return solution

    for language_id, solution in zip(language_ids, solutions):
        if language_id == LANG_PYTHON and looks_like_python3(solution):
            return solution

    return None


def merge_stdio_tests(row: dict) -> tuple[list[str], list[str]]:

    seen = set()
    inputs: list[str] = []
    outputs: list[str] = []

    for key in ["public_tests", "private_tests", "generated_tests"]:

        test_block = row.get(key)
        if not isinstance(test_block, dict):
            continue

        block_inputs = list(test_block.get("input", []))
        block_outputs = list(test_block.get("output", []))

        for test_input, test_output in zip(block_inputs, block_outputs):

            pair = (test_input, test_output)

            if pair in seen:
                continue

            seen.add(pair)

            inputs.append(test_input)
            outputs.append(test_output)

    return inputs, outputs


def prepare_codecontests(
    data_root: Path,
    pilot_size: int,
    seed: int,
) -> list[dict]:

    dataset = load_from_disk(str(data_root / "codecontests_dataset"))

    filtered = []

    for row_index, row in enumerate(dataset["train"]):

        description = row.get("description")

        if not isinstance(description, str) or not description.strip():
            continue

        if row.get("input_file") or row.get("output_file"):
            continue

        solution = pick_python_solution(
            list(row["solutions"]["language"]),
            list(row["solutions"]["solution"]),
        )

        if solution is None:
            continue

        test_input, test_output = merge_stdio_tests(row)

        if not test_input or not test_output:
            continue

        time_limit = row.get("time_limit")

        if isinstance(time_limit, dict):
            test_time_limit = max(1, int(time_limit.get("seconds", 1)))
        else:
            test_time_limit = 1

        filtered.append(
            {
                "source_dataset": "codecontests",
                "task_id": f"CodeContests/{row_index}",
                "question": description.strip(),
                "reference_code": solution,
                "trajectory_code": solution,
                "test_method": "stdio",
                "prefix": "",
                "response_prefix": "```python\n",
                "response_suffix": "\n```",
                "test_time_limit": test_time_limit,
                "test_input": test_input,
                "test_output": test_output,
                "contest_name": row.get("name", ""),
                "difficulty": int(row.get("difficulty", -1)),
            }
        )

    if pilot_size > 0 and len(filtered) > pilot_size:
        rng = random.Random(seed)
        filtered = rng.sample(filtered, pilot_size)
        filtered.sort(key=lambda item: item["task_id"])

    return filtered


def main() -> None:

    args = parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mbpp":
        records = prepare_mbpp(data_root, args.split)

    elif args.dataset == "humaneval":
        records = prepare_humaneval(data_root)

    else:
        records = prepare_codecontests(
            data_root,
            args.pilot_size,
            args.seed,
        )

    output_path = (
        Path(args.output_jsonl).resolve()
        if args.output_jsonl
        else default_output_path(
            args.dataset,
            data_root,
            args.pilot_size,
            args.split,
        )
    )

    write_jsonl(output_path, records)

    print(f"Prepared {len(records)} normalized records")
    print(f"Saved critic-source JSONL to {output_path}")


if __name__ == "__main__":
    main()