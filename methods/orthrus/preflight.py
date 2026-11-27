"""Inspect the local experiment inputs and write a machine-readable run record."""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def optional_command(argv):
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-mode", required=True)
    parser.add_argument("--target-npu", required=True)
    parser.add_argument("--train-npus", required=True)
    args = parser.parse_args()
    from training.data import load_records
    # Check the exact imports used by the trainer before scanning a full epoch.
    from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2  # noqa: F401
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    model = args.model.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 36 or config.get("hidden_size") != 2560:
        raise SystemExit(f"Expected the survey's 36-layer Qwen3-4B at {model}")
    index = model / "model.safetensors.index.json"
    weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in sorted(weights):
        file = model / name
        if not file.is_file() or file.stat().st_size == 0:
            raise SystemExit(f"Missing model weights: {file}")
    data = load_records(dataset)
    if not len(data):
        raise SystemExit("The training dataset is empty")
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "datasets", "numpy", "speculators",
                 "vllm", "vllm-ascend", "triton-ascend", "tensorboard"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parent
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": sys.executable,
        "runtime": packages, "platform": platform.platform(),
        "model": str(model), "model_type": config["model_type"], "weight_shards": sorted(weights),
        "dataset": str(dataset), "rows_before_training_filter": len(data), "columns": data.column_names,
        "dataset_fingerprint": getattr(data, "_fingerprint", None),
        "git_commit": optional_command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_branch": optional_command(["git", "-C", str(root), "branch", "--show-current"]),
        "git_status": optional_command(["git", "-C", str(root), "status", "--porcelain", "--", "."]),
        "target_npu": args.target_npu, "trainer_npus": args.train_npus,
        "teacher_mode": args.teacher_mode, "cann_toolkit": os.getenv("ASCEND_HOME_PATH"),
        "npu_inventory": optional_command(["npu-smi", "info", "-l"]),
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Model: {model.name} ({len(weights)} weight files). Dataset: {len(data):,} rows, columns {data.column_names}")
    print(f"Provenance: {args.output}")


if __name__ == "__main__":
    main()
