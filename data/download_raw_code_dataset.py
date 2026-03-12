"""
Download raw code datasets into the workspace-level `ai-project/data` folder.

Why this script exists:
- The professor-facing workflow should not depend on ad-hoc local files.
- MBPP, HumanEval, and CodeContests should all live under the same root data
  directory beside the repo.
- The critic prep scripts can then assume a stable raw-dataset location.

This script downloads one raw Hugging Face dataset and saves it to disk with
`datasets.save_to_disk(...)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


REPO_DATA_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_DATA_DIR.parent.parent.parent
WORKSPACE_DATA_ROOT = WORKSPACE_ROOT / "data"


DATASET_MAP = {
    "mbpp": {
        "hf_name": "mbpp",
        "target_dir": "mbpp_dataset",
    },
    "humaneval": {
        "hf_name": "openai/openai_humaneval",
        "target_dir": "humaneval_dataset",
    },
    "codecontests": {
        "hf_name": "deepmind/code_contests",
        "target_dir": "codecontests_dataset",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one raw code dataset into ai-project/data")
    parser.add_argument("--dataset", choices=sorted(DATASET_MAP), required=True)
    parser.add_argument(
        "--output-root",
        default=str(WORKSPACE_DATA_ROOT),
        help="Workspace-level data root. Defaults to ai-project/data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DATASET_MAP[args.dataset]

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / spec["target_dir"]

    print(f"Downloading {spec['hf_name']} ...")
    dataset = load_dataset(spec["hf_name"])

    if output_dir.exists():
        print(f"Removing previous dataset directory at {output_dir}")
        import shutil

        shutil.rmtree(output_dir)

    dataset.save_to_disk(str(output_dir))
    print(f"Saved raw dataset to {output_dir}")


if __name__ == "__main__":
    main()
