"""Client for document-local causal caches from a dedicated target process."""

import argparse
import os
import socket
import time

from .wire import ProtocolError, read_header, read_into, write_header


class TeacherClient:
    def __init__(self, socket_path, timeout=180, retries=8, model_path=None):
        self.socket_path = os.fspath(socket_path)
        self.timeout = timeout
        self.retries = retries
        self.model_path = os.path.realpath(model_path) if model_path else None
        self.received_bytes = 0
        self.requests = 0
        self.wait_seconds = 0.0

    def _request(self, sock, op, **arguments):
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        write_header(sock, {"op": op, **arguments})
        info = read_header(sock)
        if info.get("status") != "ok":
            raise RuntimeError(f"Target rejected {op}: {info.get('message', 'unknown error')}")
        if self.model_path and info.get("model") != self.model_path:
            raise ProtocolError("Teacher and trainer are using different target checkpoint paths")
        return info

    def ping(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            return self._request(sock, "ping")

    def infer(self, input_ids_1d, device):
        """Return ((K,V), ...) in HF layout and the post-final-norm [1,L,H] teacher states.

        A call always contains one document. Token positions start at zero. The
        tensors are detached and live on the requested trainer device.
        """
        import torch

        ids = torch.as_tensor(input_ids_1d)
        if ids.ndim != 1 or ids.numel() == 0 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Teacher input must be one nonempty integer token-ID vector")
        tokens = ids.detach().cpu().tolist()
        started = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    info = self._request(sock, "prefill", ids=tokens)
                    if info.get("length") != len(tokens):
                        raise ProtocolError("Teacher returned a different sequence length")
                    layers, heads, dim, hidden_size = (
                        info[field] for field in ("layers", "kv_heads", "head_dim", "hidden_size")
                    )
                    if not (1 <= layers <= 128 and 1 <= heads <= 128 and
                            1 <= dim <= 512 and 1 <= hidden_size <= 32768):
                        raise ProtocolError("Invalid teacher tensor dimensions")
                    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[info["dtype"]]
                    kv_shape = (1, heads, len(tokens), dim)

                    def receive(shape):
                        host = torch.empty(shape, dtype=dtype, device="cpu")
                        read_into(sock, host.view(torch.uint8).numpy())
                        return host.to(device=device)

                    kv = tuple((receive(kv_shape), receive(kv_shape)) for _ in range(layers))
                    final_hidden = receive((1, len(tokens), hidden_size))
                self.requests += 1
                self.received_bytes += (2 * layers * heads * dim + hidden_size) * len(tokens) * final_hidden.element_size()
                self.wait_seconds += time.monotonic() - started
                return kv, final_hidden
            except OSError:
                if attempt == self.retries:
                    raise
                time.sleep(min(2.0, 0.25 * 2 ** attempt))
        raise AssertionError("Teacher retries must be nonnegative")


def main():
    parser = argparse.ArgumentParser(description="Wait for the local Orthrus target service")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--wait", type=float, default=0, help="Maximum startup wait in seconds")
    parser.add_argument("--model", help="Expected local target snapshot path")
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait
    client = TeacherClient(args.socket, timeout=2, retries=0, model_path=args.model)
    while True:
        try:
            info = client.ping()
            print(f"Teacher ready on {args.socket} with {info['model']} (max document {info['max_length']} tokens)")
            return
        except OSError as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Target did not become ready on {args.socket}: {error}") from error
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
