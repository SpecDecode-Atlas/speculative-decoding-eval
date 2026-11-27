"""The real localhost client and teacher must preserve every document's KV and hidden."""
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

import torch

from orthrus_atlas import cache_layers
from transformers import AutoModelForCausalLM

from teacher.client import TeacherClient
from teacher.service import PrefillServer
from test_orthrus_view import tiny_target


class TeacherTests(unittest.TestCase):
    def test_unix_socket_bf16_payload_matches_independent_target(self):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
                pass
        except PermissionError:
            self.skipTest("The host sandbox blocks named Unix sockets; socketpair transport is tested separately")
        torch.set_num_threads(1)
        torch.manual_seed(991)
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            checkpoint, sock = work / "qwen", work / "target.sock"
            reference = tiny_target().to(torch.bfloat16)
            reference.save_pretrained(checkpoint)
            reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
            with (work / "teacher.log").open("w+") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "teacher.service", "--model", str(checkpoint),
                     "--device", "cpu", "--dtype", "bfloat16", "--socket", str(sock), "--max-length", "16"],
                    cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    client = TeacherClient(sock, timeout=10, retries=0, model_path=checkpoint)
                    deadline = time.monotonic() + 45
                    while True:
                        if proc.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            self.fail(f"Teacher startup failed: {log.read()}")
                        try:
                            self.assertEqual(client.ping()["max_length"], 16)
                            break
                        except OSError:
                            time.sleep(0.1)
                    for doc in ([7, 10, 11, 12, 99], [78, 3]):
                        ids = torch.tensor(doc)
                        kv, hidden = client.infer(ids, "cpu")
                        with torch.inference_mode():
                            output = reference.model(input_ids=ids[None], use_cache=True)
                        torch.testing.assert_close(hidden, output.last_hidden_state, rtol=0.0, atol=0.0)
                        for actual, expected in zip(kv, cache_layers(output.past_key_values), strict=True):
                            for tensor, gold in zip(actual, expected, strict=True):
                                torch.testing.assert_close(tensor, gold, rtol=0.0, atol=0.0)
                    self.assertEqual(client.requests, 2)
                    with self.assertRaisesRegex(RuntimeError, "length|16 tokens"):
                        client.infer(torch.zeros(17, dtype=torch.long), "cpu")
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

    def test_client_and_teacher_exchange_actual_bfloat16_bytes_on_socketpair(self):
        # socketpair is available even on hosts that prohibit binding a pathname.
        # Only connect() is a no-op. The OS transports every header and tensor.
        torch.set_num_threads(1)
        torch.manual_seed(442)
        with tempfile.TemporaryDirectory() as work:
            reference = tiny_target().to(torch.bfloat16)
            reference.save_pretrained(work)
            reference = AutoModelForCausalLM.from_pretrained(work, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
            server = PrefillServer(SimpleNamespace(
                model=work, max_length=16, log_every=10, device="cpu", dtype="bfloat16",
            ))
            left, right = socket.socketpair()
            class AlreadyConnected:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    left.close()
                def connect(self, address):
                    pass
                def __getattr__(self, attr):
                    return getattr(left, attr)
            thread = Thread(target=server.handle, args=(right,), daemon=True)
            with right:
                thread.start()
                client = TeacherClient("socketpair", timeout=10, retries=0, model_path=work)
                ids = torch.tensor([21, 16, 24, 78, 3])
                facade = SimpleNamespace(socket=lambda *args: AlreadyConnected(), AF_UNIX=1, SOCK_STREAM=1)
                with patch("teacher.client.socket", facade):
                    kv, hidden = client.infer(ids, "cpu")
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
            with torch.inference_mode():
                wanted = reference.model(input_ids=ids[None], use_cache=True)
            torch.testing.assert_close(hidden, wanted.last_hidden_state, rtol=0.0, atol=0.0)
            for actual, expected in zip(kv, cache_layers(wanted.past_key_values), strict=True):
                for tensor, gold in zip(actual, expected, strict=True):
                    torch.testing.assert_close(tensor, gold, rtol=0.0, atol=0.0)
            cfg = reference.config
            items = ids.numel() * (2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim + cfg.hidden_size)
            self.assertEqual(client.received_bytes, items * 2)
