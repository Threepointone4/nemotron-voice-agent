# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Genesys Audio Connector simulator for smoke-testing the AudioHook endpoint.

It plays the Genesys Cloud side of the protocol: it connects with the Genesys
headers (signed when a client secret is given), runs the ``open`` transaction,
streams caller audio as 8 kHz PCMU, answers ``disconnect`` with a ``close``
transaction, and saves the bot audio to a WAV file.

Example, from the repository root, with ``GENESYS_AUDIOHOOK_API_KEY`` set::

    uv run python src/examples/genesys_audiohook/simulator.py --url ws://localhost:7860/api/ws
"""

import argparse
import asyncio
import audioop
import contextlib
import json
import os
import sys
import time
import uuid
import wave
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.genesys_audiohook.auth import sign_request

SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000
PING_INTERVAL_SECS = 5.0


def load_caller_audio(path: str | None, seconds: float) -> bytes:
    """Return ``seconds`` of caller audio as 8 kHz PCMU, padding with silence."""
    pcm = b""
    if path:
        with wave.open(path, "rb") as wav:
            if wav.getsampwidth() != 2:
                raise SystemExit("--input must be a 16-bit PCM WAV file")
            pcm = wav.readframes(wav.getnframes())
            if wav.getnchannels() == 2:
                pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
            if wav.getframerate() != SAMPLE_RATE:
                pcm, _ = audioop.ratecv(pcm, 2, 1, wav.getframerate(), SAMPLE_RATE, None)
    total_bytes = int(seconds * SAMPLE_RATE) * 2
    pcm = pcm[:total_bytes] + b"\x00" * max(0, total_bytes - len(pcm))
    return audioop.lin2ulaw(pcm, 2)


class AudioConnectorSimulator:
    """Genesys Cloud side of one Audio Connector session."""

    def __init__(self, args: argparse.Namespace):
        """Create a simulator for the parsed command-line arguments."""
        self.args = args
        self.session_id = str(uuid.uuid4())
        self.seq = 0
        self.server_seq = 0
        self.position_secs = 0.0
        self.opened = asyncio.Event()
        self.finished = asyncio.Event()
        self.close_sent = False
        self.events: list[str] = []
        self.packet_sizes: Counter[int] = Counter()
        self.bot_audio = bytearray()
        self.opened_at: float | None = None
        self.first_audio_at: float | None = None

    def message(self, msg_type: str, parameters: dict) -> str:
        """Build a client message with the next sequence number."""
        self.seq += 1
        return json.dumps(
            {
                "version": "2",
                "id": self.session_id,
                "type": msg_type,
                "seq": self.seq,
                "serverseq": self.server_seq,
                "position": f"PT{self.position_secs:.3f}S",
                "parameters": parameters,
            }
        )

    def headers(self) -> dict[str, str]:
        """Return the upgrade headers Genesys Cloud sends."""
        url = urlparse(self.args.url)
        headers = {
            "x-api-key": self.args.api_key,
            "audiohook-organization-id": self.args.organization_id,
            "audiohook-session-id": self.session_id,
            "audiohook-correlation-id": str(uuid.uuid4()),
        }
        if self.args.client_secret:
            headers |= sign_request(
                headers,
                request_target=url.path + (f"?{url.query}" if url.query else ""),
                authority=url.netloc,
                api_key=self.args.api_key,
                client_secret=self.args.client_secret,
            )
        return {name: value for name, value in headers.items() if value}

    def open_parameters(self) -> dict:
        """Return ``open`` parameters shaped like a Genesys Audio Connector call."""
        return {
            "organizationId": self.args.organization_id,
            "conversationId": str(uuid.uuid4()),
            "participant": {
                "id": str(uuid.uuid4()),
                "ani": self.args.ani,
                "aniName": "Simulated Caller",
                "dnis": "+15555550199",
            },
            "media": [{"type": "audio", "format": "PCMU", "channels": ["external"], "rate": SAMPLE_RATE}],
            "language": self.args.language,
        }

    async def send_close(self, websocket, reason: str) -> None:
        """Start the close transaction; no audio may follow it."""
        if self.close_sent:
            return
        self.close_sent = True
        with contextlib.suppress(Exception):
            await websocket.send(self.message("close", {"reason": reason}))

    async def receive(self, websocket) -> None:
        """Collect bot audio and react to server control messages."""
        try:
            await self._receive_messages(websocket)
        except ConnectionClosed as exc:
            print(f"connection closed by server (code={exc.rcvd.code if exc.rcvd else 'none'})")
        self.finished.set()

    async def _receive_messages(self, websocket) -> None:
        async for data in websocket:
            if isinstance(data, bytes):
                self.first_audio_at = self.first_audio_at or time.monotonic()
                self.packet_sizes[len(data)] += 1
                self.bot_audio.extend(data)
                continue
            message = json.loads(data)
            self.server_seq = message.get("seq", self.server_seq)
            msg_type = message.get("type", "?")
            self.events.append(msg_type)
            if msg_type != "pong":
                print(f"<- {msg_type} {json.dumps(message.get('parameters', {}))}")
            if msg_type == "opened":
                self.opened_at = time.monotonic()
                self.opened.set()
            elif msg_type == "disconnect":
                await self.send_close(websocket, "disconnect")
            elif msg_type == "closed":
                break

    async def stream(self, websocket, caller_audio: bytes) -> None:
        """Send caller audio in real time, with periodic pings."""
        start = next_ping = time.monotonic()
        for index, offset in enumerate(range(0, len(caller_audio), FRAME_BYTES)):
            if self.close_sent or self.finished.is_set():
                return
            await websocket.send(caller_audio[offset : offset + FRAME_BYTES])
            self.position_secs += FRAME_MS / 1000
            if time.monotonic() >= next_ping:
                next_ping = time.monotonic() + PING_INTERVAL_SECS
                await websocket.send(self.message("ping", {}))
            await asyncio.sleep(max(0.0, start + (index + 1) * FRAME_MS / 1000 - time.monotonic()))

    async def run(self) -> int:
        """Run one session and return a process exit code."""
        caller_audio = load_caller_audio(self.args.input, self.args.seconds)
        try:
            return await self._run(caller_audio)
        except InvalidStatus as exc:
            print(
                f"WebSocket upgrade rejected with HTTP {exc.response.status_code}; check the URL path and credentials"
            )
            return 1

    async def _run(self, caller_audio: bytes) -> int:
        async with connect(self.args.url, additional_headers=self.headers(), max_size=None) as websocket:
            receiver = asyncio.create_task(self.receive(websocket))
            try:
                await websocket.send(self.message("open", self.open_parameters()))
                opened = asyncio.create_task(self.opened.wait())
                await asyncio.wait({opened, receiver}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
                opened.cancel()
                if self.opened.is_set():
                    await self.stream(websocket, caller_audio)
                    await self.send_close(websocket, "end")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.finished.wait(), timeout=10)
            except ConnectionClosed:
                pass  # The receiver reports why the server closed the connection.
            finally:
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver
        return self.report()

    def report(self) -> int:
        """Print a summary, save bot audio, and return the exit code."""
        audio_secs = len(self.bot_audio) / SAMPLE_RATE
        latency = f"{self.first_audio_at - self.opened_at:.2f}s" if self.first_audio_at and self.opened_at else "n/a"
        print(f"events: {' '.join(self.events)}")
        print(f"bot audio: {audio_secs:.1f}s in packets {dict(self.packet_sizes)}; first audio after opened: {latency}")
        if self.args.output and self.bot_audio:
            with wave.open(self.args.output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(audioop.ulaw2lin(bytes(self.bot_audio), 2))
            print(f"saved bot audio to {self.args.output}")
        return 0 if self.opened.is_set() and "closed" in self.events else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="AudioHook WebSocket URL, for example ws://localhost:7860/api/ws")
    parser.add_argument("--api-key", default=os.getenv("GENESYS_AUDIOHOOK_API_KEY", ""), help="X-API-KEY value")
    parser.add_argument(
        "--client-secret",
        default=os.getenv("GENESYS_AUDIOHOOK_CLIENT_SECRET", ""),
        help="Base64 client secret; when set, requests are signed like Genesys Cloud",
    )
    parser.add_argument("--organization-id", default=str(uuid.uuid4()), help="Genesys organization ID")
    parser.add_argument("--input", help="16-bit PCM WAV file with caller audio (default: silence)")
    parser.add_argument("--output", help="Write the bot audio to this WAV file")
    parser.add_argument("--seconds", type=float, default=15.0, help="Length of the call in seconds")
    parser.add_argument("--language", default="en-US", help="Language code sent in the open message")
    parser.add_argument("--ani", default="+15555550100", help="Caller number sent in the open message")
    return parser.parse_args(argv)


def main() -> None:
    """Run the simulator from the command line."""
    sys.exit(asyncio.run(AudioConnectorSimulator(parse_args()).run()))


if __name__ == "__main__":
    main()
