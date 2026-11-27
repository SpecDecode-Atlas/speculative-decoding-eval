"""Single-host Qwen3 causal KV/hidden-state server. See teacher/README.md."""

import argparse
import contextlib
import os
from pathlib import Path
import signal
import socket
import time

from .wire import ProtocolError, read_header, write_header


def synchronize(torch, device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class PrefillServer:
    def __init__(self, args):
        import torch
        if args.device.startswith("npu"):
            import torch_npu  # noqa: F401 - registers the Ascend device
        from transformers import AutoModel
        from orthrus_atlas import cache_layers

        self.torch = torch
        self.cache_layers = cache_layers
        self.device = torch.device(args.device)
        if self.device.type == "npu":
            torch.npu.set_device(self.device)
        self.path = str(Path(args.model).resolve(strict=True))
        self.max_length = args.max_length
        self.dtype = getattr(torch, args.dtype)
        self.dtype_name = args.dtype
        # Load only the decoder. The trainers already have a frozen LM head and
        # project selected positions when computing the total-variation loss.
        self.model = AutoModel.from_pretrained(
            self.path, dtype=self.dtype, attn_implementation="sdpa", local_files_only=True,
        ).eval().requires_grad_(False).to(self.device)
        if self.model.config.model_type != "qwen3":
            raise ValueError("This service expects a text Qwen3 checkpoint")
        self.stats = {"requests": 0, "tokens": 0, "bytes": 0, "forward_s": 0.0, "send_s": 0.0}
        self.log_every = args.log_every

    def _prefill(self, ids):
        torch = self.torch
        if not isinstance(ids, list) or not 0 < len(ids) <= self.max_length:
            raise ValueError(f"One document of 1 through {self.max_length} tokens is required")
        vocab = self.model.config.vocab_size
        if any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError("Invalid input token ID for the target vocabulary")

        with torch.inference_mode():
            tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
            output = self.model(input_ids=tokens, use_cache=True, return_dict=True)
            kv = self.cache_layers(output.past_key_values)
            hidden = output.last_hidden_state
        cfg = self.model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        kv_shape = (1, cfg.num_key_value_heads, len(ids), head_dim)
        if len(kv) != cfg.num_hidden_layers or hidden.shape != (1, len(ids), cfg.hidden_size):
            raise RuntimeError("Unexpected Qwen3 decoder output dimensions")
        if hidden.dtype != self.dtype:
            raise RuntimeError(f"Unexpected hidden dtype {hidden.dtype}")
        for key, value in kv:
            if key.shape != kv_shape or value.shape != kv_shape or key.dtype != self.dtype or value.dtype != self.dtype:
                raise RuntimeError("Unexpected attention cache shape or dtype")
        return kv, hidden, kv_shape

    def handle(self, sock):
        started = time.monotonic()
        try:
            request = read_header(sock)
            common = {"status": "ok", "model": self.path, "max_length": self.max_length}
            if request.get("op") == "ping":
                write_header(sock, common)
                return
            if request.get("op") != "prefill":
                raise ProtocolError("Expected a ping or prefill request")
            kv, hidden, (_, heads, length, dim) = self._prefill(request.get("ids"))
            synchronize(self.torch, self.device)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            print(f"Rejected teacher request: {error}", flush=True)
            with contextlib.suppress(OSError):
                write_header(sock, {"status": "error", "message": str(error)[:4096]})
            return

        ready = time.monotonic()
        try:
            write_header(sock, dict(common, length=length, layers=len(kv), kv_heads=heads,
                                    head_dim=dim, hidden_size=hidden.shape[-1], dtype=self.dtype_name))
            # Stage at most one tensor on the host at a time. Torch's NumPy BF16
            # conversion is unsupported; the byte view keeps the exact bits.
            def send(tensor):
                host = tensor.detach().contiguous().to(device="cpu")
                raw = host.view(self.torch.uint8).numpy()
                sock.sendall(memoryview(raw).cast("B"))
            for key, value in kv:
                send(key)
                send(value)
            send(hidden)
        except OSError as error:
            print(f"Interrupted teacher transfer: {error}", flush=True)
            return
        sent = time.monotonic()
        s = self.stats
        s["requests"] += 1
        s["tokens"] += length
        s["bytes"] += (len(kv) * 2 * heads * dim + hidden.shape[-1]) * length * hidden.element_size()
        s["forward_s"] += ready - started
        s["send_s"] += sent - ready
        if s["requests"] == 1 or s["requests"] % self.log_every == 0:
            print(f"teacher documents={s['requests']} tokens={s['tokens']} payload_GiB={s['bytes'] / 2**30:.3f} "
                  f"forward_s={s['forward_s']:.2f} transfer_s={s['send_s']:.2f}", flush=True)


def serve(args):
    if args.log_every < 1 or args.max_length < 1 or args.timeout <= 0:
        raise ValueError("log-every, max-length, and timeout must be positive")
    path = Path(args.socket).absolute()
    if len(os.fsencode(path)) > 107:
        raise ValueError("The Unix socket name is too long. Use a short folder under /tmp.")
    if os.path.lexists(path):
        raise FileExistsError(f"Socket path already exists: {path}. Use a fresh run-specific socket path.")
    # Decode one document at a time, independent across clients and documents.
    server = PrefillServer(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        previous_umask = os.umask(0o177)
        try:
            listener.bind(os.fspath(path))
        finally:
            os.umask(previous_umask)
        identity = path.stat()
        try:
            listener.listen(128)
            print(f"Teacher ready: {path}, model={server.path}, device={server.device}, dtype={args.dtype}", flush=True)
            while True:
                sock, _ = listener.accept()
                with sock:
                    sock.settimeout(args.timeout)
                    server.handle(sock)
        finally:
            with contextlib.suppress(FileNotFoundError):
                if os.path.samestat(identity, path.stat()):
                    path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3-4B target snapshot")
    parser.add_argument("--socket", required=True, help="Per-run Unix-domain socket on this node")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--timeout", type=float, default=180, help="Socket I/O timeout in seconds")
    parser.add_argument("--log-every", type=int, default=100, help="Emit aggregate timings every N documents")
    args = parser.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        serve(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
