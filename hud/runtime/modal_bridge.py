"""The sandbox side of ``ModalRuntime``'s transport, run as the sandbox's main process.

Modal reaches a sandbox port with an HTTP connect token, while an environment
serves its control channel over raw TCP. This script accepts WebSocket
connections on ``--listen`` and splices each to ``--upstream`` on loopback,
where the environment command it supervises serves. With ``--start-file`` it
holds the environment back until the runtime has started what it depends on,
and it exits with the environment's exit code.

It runs inside the environment image with ``python3 -c``, so it uses only the
standard library.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import os
import signal
import struct
import sys

WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CHUNK = 64 * 1024


def _frame(payload: bytes, opcode: int = 0x2) -> bytes:
    head = bytes([0x80 | opcode])
    length = len(payload)
    if length < 126:
        return head + bytes([length]) + payload
    if length < 1 << 16:
        return head + bytes([126]) + struct.pack(">H", length) + payload
    return head + bytes([127]) + struct.pack(">Q", length) + payload


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    first, second = await reader.readexactly(2)
    length = second & 0x7F
    if length == 126:
        (length,) = struct.unpack(">H", await reader.readexactly(2))
    elif length == 127:
        (length,) = struct.unpack(">Q", await reader.readexactly(8))
    mask = await reader.readexactly(4) if second & 0x80 else None
    payload = await reader.readexactly(length)
    if mask is not None:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return first & 0x0F, payload


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while chunk := await reader.read(CHUNK):
        writer.write(_frame(chunk))
        await writer.drain()


async def _serve_connection(
    client: asyncio.StreamReader, client_writer: asyncio.StreamWriter, upstream_port: int
) -> None:
    try:
        request = await client.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        client_writer.close()
        return
    headers = {
        name.strip().lower(): value.strip()
        for line in request.decode("latin-1").split("\r\n")[1:]
        if line
        for name, _, value in (line.partition(":"),)
    }
    key = headers.get("sec-websocket-key")
    if headers.get("upgrade", "").lower() != "websocket" or key is None:
        client_writer.write(b"HTTP/1.1 426 Upgrade Required\r\nContent-Length: 0\r\n\r\n")
        await client_writer.drain()
        client_writer.close()
        return
    try:
        upstream, upstream_writer = await asyncio.open_connection("127.0.0.1", upstream_port)
    except OSError:
        client_writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
        await client_writer.drain()
        client_writer.close()
        return
    digest = hashlib.sha1((key + WEBSOCKET_GUID).encode()).digest()  # noqa: S324 - the handshake
    accept = base64.b64encode(digest).decode()
    client_writer.write(
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        + f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode()
    )
    await client_writer.drain()
    pump = asyncio.ensure_future(_pump(upstream, client_writer))
    try:
        while True:
            opcode, payload = await _read_frame(client)
            if opcode in (0x0, 0x1, 0x2):
                upstream_writer.write(payload)
                await upstream_writer.drain()
            elif opcode == 0x8:
                client_writer.write(_frame(payload[:2], 0x8))
                await client_writer.drain()
                return
            elif opcode == 0x9:
                client_writer.write(_frame(payload, 0xA))
                await client_writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError):
        return
    finally:
        pump.cancel()
        with contextlib.suppress(BaseException):
            await pump
        upstream_writer.close()
        client_writer.close()


async def _run(args: argparse.Namespace) -> int:
    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _serve_connection(reader, writer, args.upstream)

    server = await asyncio.start_server(accept, "0.0.0.0", args.listen)  # noqa: S104
    if args.start_file is not None:
        # The runtime creates the file once the services the environment
        # depends on are up; a file is the one signal every image can receive.
        while not await asyncio.to_thread(os.path.exists, args.start_file):  # noqa: ASYNC110
            await asyncio.sleep(0.2)
    child = await asyncio.create_subprocess_exec(*args.command)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, child.send_signal, signum)
    try:
        return await child.wait()
    finally:
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, required=True)
    parser.add_argument("--upstream", type=int, required=True)
    parser.add_argument("--start-file")
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()
    code = asyncio.run(_run(args))
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
