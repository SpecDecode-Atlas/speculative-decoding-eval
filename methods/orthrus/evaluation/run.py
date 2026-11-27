"""Evaluate Orthrus and a matched autoregressive control. Run from the repository root."""

import argparse
import fcntl
import json
import os
import platform
import re
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path

import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.prompts import SOURCES, digest, local_rows, public_rows, render
from evaluation.reference import ar_decode, orthrus_decode
from orthrus_atlas import OrthrusView


def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def json_text(record):
    return json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def materialize(args, tokenizer, name, vocab, context):
    path = args.output / "prompts" / f"{name}.jsonl"
    meta = path.with_suffix(".source.json")
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if digest(path) != info["prompt_sha256"]:
            raise ValueError(f"Frozen prompts were modified: {path}")
        return [json.loads(line) for line in path.read_text().splitlines()]
    if args.prompts_dir:
        rows, source = local_rows(args.prompts_dir, name, args.limit)
    else:
        rows, source = public_rows(name, args.limit)
    prompts = []
    for i, row in enumerate(rows):
        ids = render(row, tokenizer, args.thinking)
        if max(ids) >= vocab or len(ids) + args.max_new_tokens > context:
            raise ValueError(f"{name} row {i}: vocabulary or context limit exceeded")
        record_id = row.get("id", row.get("question_id", i)) if isinstance(row, dict) else i
        prompts.append({"id": record_id, "input_ids": ids})
    if not prompts:
        raise ValueError(f"No prompts for {name}")
    atomic(path, "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts))
    atomic(meta, json_text({**source, "prompt_sha256": digest(path), "count": len(prompts)}))
    return prompts


def summarize(rows):
    if any(not row["parity"] for row in rows):
        return {"status": "parity_failure", "checked_prompts": len(rows)}
    total = Counter()
    for row in rows:
        for name in ("output_tokens", "rounds", "accepted_draft", "proposed", "committed_after_prefill",
                     "draft_passes", "verify_passes", "ar_seconds", "orthrus_seconds"):
            total[name] += row[name]
    def ratio(a, b):
        return total[a] / total[b] if total[b] else None

    passes = total["draft_passes"] + total["verify_passes"]
    return {"status": "completed", "prompts": len(rows), **dict(total),
            "committed_per_round": ratio("committed_after_prefill", "rounds"),
            "accepted_draft_per_round": ratio("accepted_draft", "rounds"),
            "fraction_proposed_accepted": ratio("accepted_draft", "proposed"),
            "postprefill_tokens_per_both_passes": total["committed_after_prefill"] / passes if passes else None,
            "ar_tokens_per_second": ratio("output_tokens", "ar_seconds"),
            "orthrus_tokens_per_second": ratio("output_tokens", "orthrus_seconds"),
            "wall_clock_speedup": ratio("ar_seconds", "orthrus_seconds")}


def write_report(directory, summary):
    atomic(directory / "summary.json", json_text(summary))
    lines = ["# Orthrus greedy evaluation", "",
             "Single-request Hugging Face SDPA, including prefill and decode. Output includes the first generated "
             "token and any generated EOS. Acceptance metrics exclude the token emitted by prefill.", "",
             "| Workload | Status | Prompts | Committed/round | Accepted drafts/round | Orthrus tok/s | AR tok/s | Speedup |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.items():
        values = [row.get(k) for k in ("committed_per_round", "accepted_draft_per_round",
                                       "orthrus_tokens_per_second", "ar_tokens_per_second", "wall_clock_speedup")]
        metrics = [f"{x:.3f}" if x is not None else "—" for x in values]
        lines.append(" | ".join(["| " + name, row["status"], str(row.get("prompts", row.get("checked_prompts", 0))),
                                 *metrics]) + " |")
    atomic(directory / "summary.md", "\n".join(lines) + "\n")


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def pair(target, view, tensor, cap, eos, order):
    results = {}
    for which in order:
        synchronize(tensor.device)
        started = time.perf_counter()
        result = (ar_decode(target, tensor, cap, eos) if which == "ar" else
                  orthrus_decode(target, view, tensor, cap, eos))
        synchronize(tensor.device)
        results[which] = (result, time.perf_counter() - started)
    return results


def resolve_adapter(args):
    path = args.checkpoint.resolve()
    if path.is_file() and path.name == "latest.json":
        path = path.parent / json.loads(path.read_text())["checkpoint"]
    adapter = path / "adapter.pt" if path.is_dir() else path
    manifest = adapter.parent / "manifest.json"
    metadata = {}
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        trained_target = Path(saved["model"]).resolve()
        target = args.model.resolve()
        same_snapshot = re.fullmatch("[0-9a-f]{40}", trained_target.name) and trained_target.name == target.name
        if target != trained_target and not same_snapshot:
            raise ValueError(f"Checkpoint expects the target {trained_target}; received {target}")
        for key in ("block_size", "mask_token_id"):
            if getattr(args, key) != saved[key]:
                raise ValueError(f"Checkpoint requires --{key.replace('_', '-')} {saved[key]}")
        metadata = {"manifest_sha256": digest(manifest), "training_step": saved["step"],
                    "smoke_training": saved.get("training", {}).get("smoke")}
    return adapter, metadata


def evaluate(args):
    if args.device.split(":")[0] == "npu":
        import torch_npu
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(42)
    adapter, metadata = resolve_adapter(args)
    config = {"protocol": "atlas-orthrus-1", "target": str(args.model.resolve()),
              "adapter": str(adapter.resolve()), "adapter_sha256": digest(adapter),
              "benchmarks": args.benchmarks, "prompt_directory": str(args.prompts_dir.resolve()) if args.prompts_dir else None,
              "limit": args.limit, "max_new_tokens": args.max_new_tokens, "thinking": args.thinking,
              "mask_token_id": args.mask_token_id, "block_size": args.block_size, "device": str(device),
              "torch": str(torch.__version__), "transformers": transformers.__version__,
              "dtype": args.dtype, "sdpa": True, "temperature": 0, "concurrency": 1,
              "host": platform.node(), "timing": "synchronized wall time, prefill+decode, excludes tokenization",
              **metadata}
    if device.type == "npu":
        config.update(torch_npu=str(torch_npu.__version__), device_name=torch.npu.get_device_name(device))
    elif device.type == "cuda":
        config["device_name"] = torch.cuda.get_device_name(device)
    config_path = args.output / "protocol.json"
    if config_path.exists() and config != json.loads(config_path.read_text()):
        raise ValueError("A different protocol or checkpoint already owns this output. Use a fresh --output directory.")
    atomic(config_path, json_text(config))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = getattr(torch, args.dtype)
    target = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=dtype,
                                                 attn_implementation="sdpa").to(device).eval()
    view = OrthrusView(target, mask_token_id=args.mask_token_id, block_size=args.block_size)
    state = torch.load(adapter, map_location="cpu", weights_only=True)
    view.load_state_dict(state, strict=True)
    del state
    view.to(device=device, dtype=dtype).eval()
    eos = target.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos or [])
    summary = {name: {"status": "not_started"} for name in args.benchmarks}
    write_report(args.output, summary)
    for name in args.benchmarks:
        rows = []
        try:
            prompts = materialize(args, tokenizer, name, target.config.vocab_size, target.config.max_position_embeddings)
            warm = torch.tensor([prompts[0]["input_ids"]], device=device)
            pair(target, view, warm, min(32, args.max_new_tokens), set(), ["ar", "orthrus"])
            for index, sample in enumerate(tqdm(prompts, desc=name)):
                path = args.output / "requests" / name / f"{index:05d}.json"
                signature = sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()
                if path.exists():
                    row = json.loads(path.read_text())
                    if row["prompt_sha256"] != signature:
                        raise ValueError(f"Request fingerprint mismatch in {path}")
                else:
                    ids = torch.tensor([sample["input_ids"]], device=device)
                    order = ["ar", "orthrus"] if index % 2 == 0 else ["orthrus", "ar"]
                    result = pair(target, view, ids, args.max_new_tokens, stops, order)
                    ar, ar_s = result["ar"]
                    spec, spec_s = result["orthrus"]
                    row = {"id": sample["id"], "prompt_sha256": signature, "prompt_tokens": ids.shape[1],
                           "output_tokens": len(spec.tokens), "output_ids": spec.tokens,
                           "text": tokenizer.decode(spec.tokens, skip_special_tokens=True),
                           "parity": spec.tokens == ar.tokens, "order": order, "ar_seconds": ar_s,
                           "orthrus_seconds": spec_s, "rounds": spec.rounds, "accepted_draft": spec.accepted_draft,
                           "proposed": spec.proposed, "draft_passes": spec.draft_passes, "verify_passes": spec.verify_passes,
                           "committed_after_prefill": spec.committed_after_prefill}
                    if not row["parity"]:
                        row["ar_output_ids"] = ar.tokens
                    atomic(path, json_text(row))
                rows.append(row)
                if not row["parity"]:
                    raise RuntimeError(f"{name} failed greedy token parity on request {sample['id']}; inspect {path}")
            summary[name] = summarize(rows)
            write_report(args.output, summary)
        except BaseException:
            summary[name] = {"status": "partial", "checked_prompts": len(rows)}
            if rows and not rows[-1]["parity"]:
                summary[name]["status"] = "parity_failure"
            write_report(args.output, summary)
            raise
    print(f"Results: {args.output / 'summary.md'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prompts-dir", type=Path)
    p.add_argument("--benchmarks", choices=list(SOURCES), nargs="+", default=list(SOURCES))
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--mask-token-id", type=int, default=151669)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--thinking", action="store_true", help="Enable the Qwen3 chat template's thinking mode")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = p.parse_args()
    if args.limit < 1 or args.max_new_tokens < 1 or len(set(args.benchmarks)) != len(args.benchmarks):
        p.error("Use positive sample/token limits and unique benchmark names")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(args)


if __name__ == "__main__":
    main()
