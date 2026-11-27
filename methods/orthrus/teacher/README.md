# Dedicated target prefill

The target process loads the frozen Qwen3 decoder on one Ascend device. For each unpadded document it computes causal attention once, then returns every layer’s post-RoPE keys and values and the decoder’s final normalized hidden states. Positions start at zero for each document. The trainer reuses the same document cache for all its block anchors. Its frozen local vocabulary head projects only the positions it needs for the teacher distribution. This keeps the full-vocabulary logits off the socket.

The trainer and the target must run on the same host. Use a short, run-specific Unix-domain socket path. The socket is readable and writable only by its owner. The training launcher manages its lifetime. To run the service manually for diagnostics from the repository root:

```bash
socket_dir=$(mktemp -d)
socket="$socket_dir/teacher.sock"
ASCEND_RT_VISIBLE_DEVICES=0 python -u -m teacher.service \
    --model "$MODEL" --socket "$socket" --device npu:0 \
    --max-length 3072 > "$socket_dir/teacher.log" 2>&1 &
teacher_pid=$!
python -m teacher.client --socket "$socket" --wait 1800 --model "$MODEL"
```

After the trainers exit, run `kill "$teacher_pid"` in that shell to release the device.

The cost of this topology is host transfer bandwidth. With Qwen3-4B, a full 3,072-token document transfers 432 MiB of BF16 KV and 15 MiB of normalized hidden states. Each completed transfer appears in cumulative counters in the teacher log. The forward time includes request decoding and device synchronization. Transfer time includes device-to-host staging and socket sends. The receiver moves one contiguous tensor at a time to its own NPU. Seven ranks with three full documents each consume about 9.17 GiB of teacher payload per training update. Benchmark inference bypasses this service.
