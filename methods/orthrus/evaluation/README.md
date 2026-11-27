# Evaluate a trained Orthrus checkpoint

Run from the method checkout in the same Conda environment used for training. Choose an available card. With `ASCEND_RT_VISIBLE_DEVICES=4`, `npu:0` refers to physical card 4.

```bash
MODEL=/home/n84449292/a84384644/models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
# Set CHECKPOINT to a saved directory that contains adapter.pt.
ASCEND_RT_VISIBLE_DEVICES=4 python -m evaluation.run \
    --model "$MODEL" --checkpoint "$CHECKPOINT" --output "$CHECKPOINT/eval" --device npu:0
```

For a quick test use `--benchmarks gsm8k --limit 2 --max-new-tokens 32` and a separate output directory. The default benchmark runs one request at a time, greedily, with BF16, Hugging Face SDPA, a 16-position block and a cap of 512 new tokens per prompt. The Qwen chat template runs with thinking disabled by default. Use `--thinking` to enable it for both methods. Prompt templates and sample order affect acceptance and speed; use the same frozen manifests for every method reported together.

You can also pass the trainer's `checkpoints/latest.json` to `--checkpoint`. The evaluator validates the training manifest's target, mask ID and block size, and records the resolved adapter digest, training step and smoke-run flag.

## Prompt sets

The first run can build manifests from Hugging Face. The SHA of each dataset revision and the SHA-256 of the tokenized manifest are stored beside the prompts. The public protocol takes the first 128 records in source order, capped by split size.

| Key | Public source | Workload |
|---|---|---|
| gsm8k | openai/gsm8k, main | Test questions |
| math500 | HuggingFaceH4/MATH-500 | Test problems |
| humaneval | openai/openai_humaneval | Test function-completion prompts |
| mbpp | google-research-datasets/mbpp, sanitized | Test prompts |
| lcb | livecodebench/code_generation_lite | Cumulative release v5; problem and available starter code |
| aime25 | yentinglin/aime_2025, default | All 30 problems |
| mtbench | HuggingFaceH4/mt_bench_prompts | First turn of each of 80 conversations |
| alpaca | tatsu-lab/alpaca | Stanford Alpaca training-split instruction prompts, with optional input |

For an offline run or comparisons to earlier results, pass `--prompts-dir /path/to/manifests`. Place each selected workload in a file named for its key, such as `gsm8k.jsonl`. Each line can contain an ID and the final token sequence:

```json
{"id":"case-001","input_ids":[100,200,300]}
```

These integers illustrate the file format. A record can instead contain `formatted_prompt` (already templated), `messages` (chat history), or `prompt` (one user message). The loader also accepts `turns` and uses its first turn. The emitted `prompts/` directory is directly reusable with `--prompts-dir`; this preserves the exact model inputs across methods.

## What is measured

Orthrus prefills the prompt once to obtain the first target token. Each cycle runs the diffusion view on the pending token and mask positions, then verifies proposals in a single causal forward pass. The longest agreeing prefix survives, with one target correction at a mismatch or one bonus after complete agreement when there is room. Verification stops at EOS or the output cap. The cache is cropped to the exact verified context. The AR control uses the same model, attention backend, prompt IDs, EOS set and token cap. Both paths are warmed up before each workload, and their execution order alternates by prompt.

| Reported metric | Calculation |
|---|---|
| Committed per round | Sum of new tokens emitted after prefill divided by total verification rounds. Includes corrections and bonuses actually emitted. |
| Accepted drafts per round | Total matching proposal tokens retained divided by total verification rounds. |
| Tokens per both passes | Tokens emitted after prefill divided by diffusion passes plus verification passes. |
| Tokens/s | Total output tokens, including the first generated token and any generated EOS, divided by summed synchronized device wall time from prefill through completed decoding. |
| Wall-clock speedup | Summed AR seconds divided by summed Orthrus seconds on identical token outputs. |

The throughput measurement includes Python control flow and device synchronization for this Hugging Face implementation. It excludes tokenization, model loading, prompt downloading, and queueing. Report the engine when comparing measurements from other implementations.

## Result files

`summary.md` and `summary.json` contain benchmark-level results. `protocol.json` records the checkpoint digest, target, device, framework versions, token cap and decoding settings. Each `requests/<benchmark>/<index>.json` contains both latencies, acceptance counters, generated token IDs and decoded text for downstream scoring. This runner measures decoding parity and performance. Benchmark accuracy, pass@1 and judge scores require the corresponding task scorer and complete answers.

The runner compares all greedy output IDs for every prompt. A mismatch saves both traces, marks the benchmark failed, and stops before producing a speedup for that workload. Different batched shapes can expose numerical differences in low precision. Completed per-prompt records are written atomically. Rerun the same command to resume, or choose a new output directory to change settings.
