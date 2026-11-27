"""A length-prefixed JSON control frame followed by contiguous tensor bytes."""

import json
import struct


VERSION = 1
MAX_CONTROL_BYTES = 256 * 1024


class ProtocolError(RuntimeError):
    """A request or response used an incompatible schema."""


def read_into(sock, buffer):
    view = memoryview(buffer).cast("B")
    while view.nbytes:
        n = sock.recv_into(view)
        if n == 0:
            raise ConnectionError("Teacher connection closed before the frame was complete")
        view = view[n:]


def write_header(sock, header):
    encoded = json.dumps({"version": VERSION, **header}, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_CONTROL_BYTES:
        raise ProtocolError("Teacher control frame is too large")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)


def read_header(sock):
    size = bytearray(4)
    read_into(sock, size)
    (length,) = struct.unpack("!I", size)
    if not 0 < length <= MAX_CONTROL_BYTES:
        raise ProtocolError("Invalid teacher control frame size")
    content = bytearray(length)
    read_into(sock, content)
    result = json.loads(content)
    if not isinstance(result, dict) or result.get("version") != VERSION:
        raise ProtocolError("Unsupported teacher protocol version")
    return result
