"""
Sanity-check the Linux training environment before running Trado critic RL.

This script is intentionally conservative:
- it treats the current native Windows environment as unsupported for full RL
- it checks the CUDA-visible PyTorch build, not just whether a GPU exists
- it verifies the Linux-side packages that the Trado critic path depends on

It is meant to be run on the professor's Linux machine after environment setup
and before `rl_trado_critic.py ... experiment.preflight_only=true`.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_command(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:
        return False, str(exc)
    text = (completed.stdout or completed.stderr or "").strip()
    return True, text


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_os() -> CheckResult:
    system = platform.system()
    if system != "Linux":
        return CheckResult(
            "platform",
            False,
            f"Expected Linux for full training, found {system}. Native Windows is only supported for CPU/preflight checks.",
        )
    return CheckResult("platform", True, f"Linux detected: {platform.platform()}")


def check_python() -> CheckResult:
    version = sys.version_info
    if (version.major, version.minor) != (3, 10):
        return CheckResult(
            "python",
            False,
            f"Expected Python 3.10 for the upstream training stack, found {version.major}.{version.minor}.{version.micro}.",
        )
    return CheckResult("python", True, f"Python {version.major}.{version.minor}.{version.micro}")


def check_nvidia_smi() -> CheckResult:
    if shutil.which("nvidia-smi") is None:
        return CheckResult("nvidia-smi", False, "nvidia-smi was not found in PATH.")
    ok, text = run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not ok:
        return CheckResult("nvidia-smi", False, text)
    return CheckResult("nvidia-smi", True, text)


def check_torch_cuda() -> list[CheckResult]:
    try:
        import torch
    except Exception as exc:
        return [CheckResult("torch", False, f"Failed to import torch: {exc}")]

    results = [
        CheckResult("torch", True, f"torch={torch.__version__} torch_cuda={torch.version.cuda}"),
    ]

    if not torch.cuda.is_available():
        results.append(
            CheckResult(
                "torch.cuda",
                False,
                "torch.cuda.is_available() is False. This usually means a CPU-only torch wheel or a broken CUDA runtime.",
            )
        )
        return results

    count = int(torch.cuda.device_count())
    if count < 1:
        results.append(CheckResult("torch.cuda", False, "torch.cuda.device_count() returned 0."))
        return results

    gpu_name = torch.cuda.get_device_name(0)
    results.append(CheckResult("torch.cuda", True, f"CUDA visible with {count} GPU(s); gpu0={gpu_name}"))
    return results


def check_modules() -> list[CheckResult]:
    checks = []
    required_modules = [
        ("accelerate", "accelerate"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("deepspeed", "deepspeed"),
        ("flash_attn", "flash_attn"),
        ("xformers", "xformers"),
        ("triton", "triton"),
        ("flashinfer", "flashinfer"),
    ]
    for label, module_name in required_modules:
        ok = module_exists(module_name)
        detail = "import ok" if ok else f"missing module: {module_name}"
        checks.append(CheckResult(label, ok, detail))
    return checks


def check_dev_shm() -> CheckResult:
    if not shutil.which("df"):
        return CheckResult("dev_shm", True, "skipped /dev/shm check because `df` is unavailable")
    ok, text = run_command(["df", "-h", "/dev/shm"])
    if not ok:
        return CheckResult("dev_shm", False, text)
    return CheckResult("dev_shm", True, text)


def main() -> int:
    results: list[CheckResult] = []
    results.append(check_os())
    results.append(check_python())
    results.append(check_nvidia_smi())
    results.extend(check_torch_cuda())
    results.extend(check_modules())
    results.append(check_dev_shm())

    failed = [item for item in results if not item.ok]
    for item in results:
        status = "PASS" if item.ok else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}")

    if failed:
        print("\nEnvironment is not ready for Linux Trado critic training.")
        print("Fix the failed checks above, then rerun this script and the launcher preflight.")
        return 1

    print("\nEnvironment looks ready for the launcher preflight.")
    print("Next step: python rl_trado_critic.py config=configs/rl_trado_critic_mbpp_with_value.yaml experiment.preflight_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
