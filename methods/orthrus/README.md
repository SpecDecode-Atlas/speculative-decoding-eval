# Orthrus on Qwen3-4B

This branch trains a parallel drafting view of Qwen3-4B for one epoch on deduplicated PerfectBlend rollouts. It compares greedy Orthrus decoding with greedy autoregressive (AR) decoding on the same target, prompts, hardware and inference backend. The research question is how many tokens the target commits per verification round and whether those extra tokens reduce wall time.

**Branch:** `orthrus` in `SpecDecode-Atlas/speculative-decoding-eval`. Shared vLLM, vLLM-Ascend and Speculators installations can stay on their existing release branches. This implementation uses the Hugging Face Qwen3 decoder, PyTorch SDPA and the Speculators multipack sampler. The released Qwen3 inference implementation and the paper are the architectural references. [Paper](https://arxiv.org/abs/2605.12825) · [Authors’ Qwen3 code, pinned revision](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/model.py)

## Start on an Ascend A2

Run these commands from `methods/orthrus` in the method worktree. The launcher activates the existing `spec_decode_atlas` Conda environment and sources CANN and ATB 9.1.0. The default uses seven trainers on NPUs 1–7. Each trainer computes causal teacher features on its own card using the frozen Qwen3 weights already present there.

```bash
bash run.sh --prepare-only     # Check snapshot, Arrow data, environment and provenance
bash run.sh --check-only       # One short optimizer update through the actual teacher and training ranks
```

For the full epoch, keep the run directory in a shell variable. The shell prints a live log as soon as the job starts.

```bash
export RUN_ROOT=/home/n84449292/a84384644/SpecDecode-Atlas/runs/orthrus
mkdir -p "$RUN_ROOT"
RUN=$(mktemp -d "$RUN_ROOT/epoch1-$(date +%Y%m%d-%H%M%S)-XXXX")
OUTPUT_DIR="$RUN" nohup bash run.sh --run > "$RUN/console.log" 2>&1 < /dev/null &
echo "PID $!  |  RUN $RUN"
tail -f "$RUN/console.log"
```

Stopping `tail` leaves the training process running. The launcher writes the training log and a separate teacher log when using a dedicated target. To open TensorBoard on the server, use `tensorboard --logdir "$RUN"`.

### Inputs

| Setting | Default |
|---|---|
| `BASE` | `/home/n84449292/a84384644` |
| `MODEL` | `$BASE/models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c` |
| `DATASET` | `$BASE/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072.dedup` |
| `CONDA_ENV` | `/home/n84449292/miniforge3/envs/spec_decode_atlas` |
| `CANN_ROOT` | Detects CANN 9.1.0 under the configured data root or the shared tools volume |
| `TRAIN_NPUS` | `1,2,3,4,5,6,7`, physical IDs |
| `TEACHER_MODE` / `FSDP_STRATEGY` | `local` / `shard-grad-op`. Set `TEACHER_MODE=dedicated TARGET_NPUS=0` to run the separate target service. |
| `MASTER_PORT` | `29647` on loopback |
| `RUN_ROOT` | `$BASE/SpecDecode-Atlas/runs/orthrus` |

The dataset loader expects pretokenized `input_ids` per document. It recognizes `loss_mask` and label masking with `-100`. An optional `attention_mask` removes padding. With neither supervision field, all eligible next-token positions are supervised. The supplied rollouts are already deduplicated. Eligible documents contribute once per epoch. Starts are sampled without replacement across each rank's pack where the first successor is supervised. Later slots beyond the same document or outside supervision contribute zero loss. This allows a partial block near a document's end. The run records the eligible and excluded row counts and the source fingerprint.

Training uses one Python environment. The serving stack can retain NumPy 1.26.4. If importing `datasets` fails specifically because a newly installed PyArrow demands NumPy 2, `python -m pip install --no-deps 'pyarrow==21.0.0'` is the pairing verified in the CPU tests with NumPy 1.26.4 and datasets 5.0.0. Use it in the activated environment experiencing that error. Source data, model weights and run artifacts live outside the method checkout.

## What this branch changes

1. The frozen target computes clean causal K/V at every decoder layer. Each trainable diffusion layer starts from that target layer’s attention projections and Q/K normalization weights. The draft view reuses the target’s frozen embedding, residual path, MLPs, layer norms and vocabulary head.
2. A sampled anchor at document position `a` supplies the known token `x[a]`. The other 15 inputs are mask token `151669`. Each query can attend to true target K/V at positions strictly less than `a` plus its own 16-position bidirectional block. Other sampled blocks, future clean tokens and other documents are masked out. Diffusion row `j` predicts `x[a+j+1]` and is compared with the target distribution at row `a+j`.
3. At evaluation, the target emits the first token during prefill. The draft proposes up to 15 continuation tokens. One target pass checks all proposals. The decoder commits the matching prefix plus the target correction at the first mismatch, or the target bonus when all proposals match, subject to EOS and the output cap. It crops the shared target cache after each verification.

This follows the public Qwen3 implementation’s separate **Q/K/V/O projections and Q/K RMSNorm** at each of the 36 layers. The paper describes Q/K/V in its parameter accounting. For the 4B configuration, the release-style adapter has 943,727,616 trainable parameters. The five DFlash capture layers, four conditioning anchors, DFlash checkpoint, redraft pass and boundary-exploration loss in the example launcher are specific to the four-anchor experiment. Orthrus needs the complete layerwise target K/V and one visible token per sampled block. [Model source](orthrus_atlas/model.py) · [Authors’ architecture](https://github.com/chiennv2000/orthrus/blob/4dceab65156b3dfb5dadbb11181a0e65d0ad314d/src/model.py#L87-L105)

## Survey controls and paper settings

| Setting | This survey run | Paper, Appendix A |
|---|---|---|
| Training data | Deduplicated PerfectBlend Qwen3-4B rollouts | Nemotron v2 chat, math and code |
| Epochs | 1 | 2 |
| Document length | Up to 3,072 | 2,048 |
| Per-rank token and anchor budgets | 9,216 tokens and up to 768 sampled starts per optimizer update, with independent documents | 256 anchor blocks per packed sequence |
| Block | 16 positions, 1 visible token and 15 proposals | 32 positions |
| Objective | 0.1 hard CE to teacher argmax + 0.9 total variation over the full vocabulary; slot `j` weighted by `exp(-j/7)` starting at zero | Forward KL from the frozen teacher |
| Optimizer | AdamW, LR 0.0006, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, gradient norm clip 1 | Peak LR 0.0002, gradient norm clip 1 |
| Schedule | 50 warmup updates, then cosine; accumulation 1 | 5% warmup, then cosine; accumulation 16 |
| Execution | Seed 42, FP32 master parameters, BF16 compute, SDPA, FSDP on multiple trainers, activation checkpointing | BF16, FSDP-2 on GPUs |
| Working chunks | Up to 64 blocks per forward, 256 prediction rows per vocabulary-head chunk | — |
| Monitoring | Four loader workers per trainer, log every 10 updates; TensorBoard; separate post-training evaluation | — |

The adapter initializes from this target snapshot. It can be reconstructed from the snapshot plus the saved adapter weights. The survey results should be labeled **Orthrus, Atlas one-epoch CE/TV recipe**. [Paper training details](https://arxiv.org/html/2605.12825v2#A1) · [Atlas objective](training/objective.py)

### Target placement and training cost

The default computes each clean document's teacher states once on its trainer. This eliminates the dedicated service's transfer of 432 MiB of K/V and 15 MiB of final hidden states for a 3,072-token BF16 document. The teacher still runs a real forward pass on each trainer. Historical experiments can select `TEACHER_MODE=dedicated`. [Teacher protocol](teacher/README.md)

The adapter stays gathered between the microchunks of an optimizer update using FSDP `SHARD_GRAD_OP` and `no_sync()`. Gradients synchronize on the final chunk and Adam state stays sharded. This trades device-memory residency between chunks for fewer all-gathers. A PyTorch 2.10 scheduling test using a simulated seven-rank transport counted one adapter all-gather per four-chunk update, compared with four under `FSDP_STRATEGY=full-shard`. The attention implementation also slices off clean K/V past the latest anchor in each chunk. Those positions are already inaccessible to every query. [PyTorch sharding strategies](https://docs.pytorch.org/docs/2.10/fsdp.html#torch.distributed.fsdp.ShardingStrategy)

When sampled blocks overlap, several student predictions may share a teacher position. Within a vocabulary-head chunk, the trainer computes the frozen projection and softmax once per distinct teacher position. Each student prediction keeps its own loss and weight.

#### Time full-size updates before committing to an epoch

Run on the same seven available cards as the baseline. The benchmark uses the epoch's actual packed data, optimizer and learning-rate schedule. It warms up on ten updates and times updates 11–20. It writes `benchmark.json`, including seconds per update, context tokens/s, teacher payload, batch-plan digests and peak device memory where available. Benchmark outputs use a separate run directory and omit training checkpoints.

```bash
bash run.sh --benchmark
# To compare directly with a previous epoch's run.json and metrics.jsonl:
BASELINE_RUN=/path/to/previous/epoch-run bash run.sh --benchmark
```

`BASELINE_RUN` checks the training controls, document count and token count for the measured window, and writes `comparison.json` with the observed speedup. This compares training wall time. To isolate teacher and FSDP placement on the current implementation, run another benchmark sequentially with `TEACHER_MODE=dedicated FSDP_STRATEGY=full-shard`. The timing window can be adjusted with `BENCHMARK_STEPS` and `BENCHMARK_WARMUP`. The full-size test exercises the memory footprint of retaining gathered parameters across chunks. Use `FSDP_STRATEGY=full-shard` if that buffer exceeds your available memory.

### Checkpoints, timing and resume

At the ceiling of 25%, 50%, 75% and 100% of optimizer updates, the trainer saves the diffusion adapter, optimizer state, run metadata and each rank’s RNG state. Packed documents have different lengths, so those boundaries measure updates. Four full FP32 adapter and Adam checkpoint sets for this 4B target need roughly **42 GiB in total**, before staging, logs and filesystem overhead. The target weights are shared across checkpoints.

The short check also writes an adapter and optimizer snapshot, roughly 10.5 GiB for Qwen3-4B. It tests the distributed path on two blocks per rank. Peak memory at the full 64-block chunk size must be measured during training. If a fresh run needs smaller working chunks, set `BLOCKS_PER_CHUNK` and `HEAD_CHUNK_SIZE`; the 9,216-token and 768-anchor update budgets stay the same.

| File under the run directory | Contents |
|---|---|
| `run.json`, `progress.json`, `metrics.jsonl` | Run settings, update progress and training measurements |
| `tensorboard/` | Training scalars |
| `session-*/launcher.log`, `session-*/teacher.log` | Process logs for each launch; teacher log in dedicated mode |
| `session-*/provenance.json`, `session-*/status.tsv` | Source revision, dataset fingerprint, machine and package versions; stage timestamps, elapsed seconds and exit code |
| `checkpoints/latest.json` | Last complete checkpoint directory name and update |
| `checkpoints/checkpoint-*/` | Adapter and trainer state, manifest and per-rank RNG |

Recover the checkpoint after an interrupted run. Set `RUN` to the original run directory if this is a new shell.

```bash
CKPT="$RUN/checkpoints/$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN/checkpoints/latest.json")"
OUTPUT_DIR="$RUN" RESUME="$CKPT" bash run.sh --run
```

Resume from a completed checkpoint using the same target, data, number of trainers and update recipe. Teacher placement and FSDP buffer strategy can change on resume. The trainer records the transition and restores the full adapter, optimizer state and per-rank RNG. In-flight updates after the latest saved checkpoint must be rerun after an interruption. At this model size, checkpoint I/O is part of elapsed training time.

## Evaluate the one-epoch checkpoint

After the training process reports completion, select its last checkpoint. These commands run the full survey suite on physical NPU 0.

```bash
CHECKPOINT="$RUN/checkpoints/latest.json" OUTPUT_DIR="$RUN" EVAL_NPU=0 bash run.sh --evaluate
```

The eight workloads are GSM8K, MATH-500, HumanEval, MBPP (sanitized), LiveCodeBench v5, AIME 2025, MT-Bench (first turn), and Alpaca. The default uses the first 128 available prompts per workload, including 30 AIME questions and 80 MT-Bench prompts, with up to 512 new tokens. Source revisions, split names and the final prompt IDs are persisted. Public dataset access is needed the first time those prompts are materialized. To reuse the exact prompts from a previous study or run offline, supply one file per benchmark, for example `$PROMPTS/gsm8k.jsonl` with lines such as `{"id":"g0","input_ids":[10,20,30]}`. Existing token IDs bypass prompt formatting. Text and chat message inputs are also supported.

```bash
# Small offline comparison before running all eight datasets.
CHECKPOINT="$RUN/checkpoints/latest.json" OUTPUT_DIR="$RUN" bash run.sh --evaluate \
    --prompts-dir "$PROMPTS" --benchmarks gsm8k --limit 4 --max-new-tokens 64
```

The report path appears in the log under `session-evaluate-*/evaluation/summary.md`; `summary.json` contains the numerical totals. The runner stores prompt IDs, per-request output tokens, per-method timings, counts and parity checks. Reuse the frozen prompt directory and identical decoding settings across method branches. For a retry of an interrupted evaluation, pass `--output /path/to/the/existing/evaluation` after `--evaluate`.

| Metric | Definition |
|---|---|
| Committed tokens per verification round | Pooled post-prefill emitted tokens divided by verification rounds. Includes emitted corrections and bonuses. EOS and the length cap apply. |
| Accepted drafts per round | Pooled matching draft tokens divided by verification rounds |
| Output tokens/s | Total generated tokens, including the first and any generated EOS, divided by summed synchronized wall time, including prefill and decode |
| Speedup | Matched AR wall time divided by Orthrus wall time on the same prompts |
| Greedy parity | Exact output token IDs must agree for every request. On mismatch both outputs are saved and the affected benchmark has no speedup result. |

This evaluates **single-request greedy PyTorch/Hugging Face inference**. vLLM continuous-batching throughput, stochastic decoding and task scores such as pass@1 require separate experiments. Both paths run against the same 4B target and SDPA implementation. The runner warms both paths, alternates their measurement order, and records the timing boundaries and prompt-selection protocol. Compare across methods using an equivalent engine, batch size and timing definition. [Benchmark runner](evaluation/run.py)

## Implementation guide

| Path | Responsibility |
|---|---|
| [run.sh](run.sh), [preflight.py](preflight.py) | Activate the environment, validate inputs, place processes, capture logs and elapsed time |
| [orthrus_atlas/model.py](orthrus_atlas/model.py) | Per-layer diffusion parameters, prefix/block attention mask, frozen backbone reuse |
| [training/data.py](training/data.py), [training/objective.py](training/objective.py) | Document isolation, packing, anchor sampling, CE/TV and vocabulary chunking |
| [training/train.py](training/train.py), [training/checkpoint.py](training/checkpoint.py) | Distributed updates, optimizer schedule, logs, checkpoints and resume |
| [teacher/](teacher/) | Local dedicated target service and client |
| [evaluation/](evaluation/) | Prompt preparation, cached greedy verifier, AR control, parity and results |
| [tests/](tests/) | Small CPU numerical and behavioral checks |

For the next method, create its branch from the shared evaluation repository’s `main`. Keep its implementation under `methods/<name>/`, give it the same target/dataset and matched controls, and document any method-specific differences. Include the branch commit, dataset fingerprint, prompt hashes, decoding settings, hardware, elapsed training time and output summary with published comparisons.

### Validation before release

The CPU test used PyTorch 2.10.0, Transformers 5.5.4, NumPy 1.26.4, PyArrow 21.0.0 and datasets 5.0.0. A small Qwen3 with the production vocabulary trained for four updates on eight synthetic Arrow documents, wrote four checkpoints and produced TensorBoard events. Resuming at 50% reproduced the final adapter, AdamW state and rank RNG bit for bit. Reloading the trained adapter produced identical greedy AR and Orthrus tokens in all eight workload paths using synthetic prompt manifests.

With the revised trainer, four BF16 CPU updates on the same eight documents and full production vocabulary reproduced the original adapter weights to within a maximum absolute difference of 3.7e-7. The local teacher reported zero transferred bytes. These measurements cover the small CPU fixture. Use the matched benchmark to measure full-model throughput on an A2.

Run `python -m unittest discover -s tests` from the method directory. The numerical tests cover native-Qwen3 attention agreement, prefix/block isolation, full-gradient CE/TV behavior, sampler coverage and rejection/crop cases. The Unix socket-pair test checks the teacher/client tensor payload. The server's `--check-only` command tests the selected placement, real 4B target and a short optimizer step. Use `--benchmark` for A2 training timings and `--evaluate` for decoding metrics.

### Attribution

Chien Van Nguyen, Chaitra Hegde, Van Cuong Pham, Ryan A. Rossi, Franck Dernoncourt and Thien Huu Nguyen introduced Orthrus in *Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion* (2026). The authors distribute their reference implementation under the MIT license. This branch adds the Atlas training, Ascend placement and evaluation protocol. [Paper](https://arxiv.org/abs/2605.12825) · [Upstream license notice](UPSTREAM-LICENSE.txt)
