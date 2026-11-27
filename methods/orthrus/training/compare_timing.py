"""Compare the full-size timed window with the same updates of a previous epoch."""
import argparse
import json
from pathlib import Path


def compare(baseline, benchmark):
    old = json.loads((baseline / "run.json").read_text())
    new = benchmark["training"]
    controls = ("model", "config_sha256", "dataset_fingerprint", "world_size", "seed",
                "optimizer_steps", "document_max_length", "pack_tokens", "max_anchors",
                "block_size", "mask_token_id", "blocks_per_chunk", "head_chunk_size", "loss")
    differing = [key for key in controls if old[key] != new[key]]
    if differing:
        raise ValueError("Different training controls: " + ", ".join(differing))
    start, end = benchmark["first_measured_step"] - 1, benchmark["last_measured_step"]
    markers = {0: {"elapsed_seconds": 0.0, "context_tokens_seen": 0, "documents_seen": 0}}
    with (baseline / "metrics.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            if item["step"] in (start, end):
                markers[item["step"]] = item
            if item["step"] >= end:
                break
    if start not in markers or end not in markers:
        raise ValueError(f"The baseline metrics need logged updates {start} and {end} for this comparison.")
    first, last = markers[start], markers[end]
    for name, key in (("context_tokens", "context_tokens_seen"), ("documents", "documents_seen")):
        if last[key] - first[key] != benchmark[name]:
            raise ValueError(f"Different {name} counts in updates {start + 1}..{end}.")
    seconds = (last["elapsed_seconds"] - first["elapsed_seconds"]) / (end - start)
    current = benchmark["seconds_per_update"]
    return {"updates": [start + 1, end], "baseline_seconds_per_update": seconds,
            "benchmark_seconds_per_update": current, "speedup": seconds / current}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("benchmark_json", type=Path)
    args = parser.parse_args()
    summary = compare(args.baseline_run, json.loads(args.benchmark_json.read_text()))
    print(f"Updates {summary['updates'][0]}–{summary['updates'][1]}: "
          f"baseline {summary['baseline_seconds_per_update']:.2f} s/update, "
          f"current {summary['benchmark_seconds_per_update']:.2f} s/update, "
          f"speedup {summary['speedup']:.3f}x.")
    args.benchmark_json.with_name("comparison.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
