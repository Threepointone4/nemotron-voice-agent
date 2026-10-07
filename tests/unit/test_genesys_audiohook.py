# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103
"""Genesys Audio Connector example: authentication, AudioHook protocol, and session lifecycle."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml
from pipecat.frames.frames import (
    CancelFrame,
    CancelWorkerFrame,
    EndFrame,
    EndTaskFrame,
    EndWorkerFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.genesys import GenesysAudioHookSerializer
from pipecat.tests.utils import run_test

from examples.genesys_audiohook.auth import (
    AudioHookAuthConfig,
    NonceCache,
    SignatureError,
    authenticate_request,
    request_target_from_scope,
    sign_request,
    unauthorized_disconnect_message,
    verify_signature,
)
from examples.genesys_audiohook.serializer import AudioConnectorSerializer
from examples.genesys_audiohook.session import AudioHookSessionProcessor
from examples.genesys_audiohook.transport import (
    AUDIOHOOK_OUT_10MS_CHUNKS,
    AUDIOHOOK_PACKET_BYTES,
    AUDIOHOOK_SAMPLE_RATE,
    create_audiohook_transport,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# "Example 1" from the Genesys AudioHook protocol documentation, also used by the
# official reference implementation (purecloudlabs/audiohook-reference-implementation).
DOC_API_KEY = "SGVsbG8sIEkgYW0gdGhlIEFQSSBrZXkh"
DOC_CLIENT_SECRET = "TXlTdXBlclNlY3JldEtleVRlbGxOby0xITJAMyM0JDU="
DOC_TARGET = "/api/v1/voicebiometrics/ws"
DOC_NOW = 1641013205.0
DOC_HEADERS = {
    "host": "audiohook.example.com",
    "audiohook-organization-id": "d7934305-0972-4844-938e-9060eef73d05",
    "audiohook-session-id": "30b0e395-84d3-4570-ac13-9a62d8f514c0",
    "audiohook-correlation-id": "e160e428-53e2-487c-977d-96989bf5c99d",
    "x-api-key": DOC_API_KEY,
    "signature": "sig1=:NZBwyBHRRyRoeLqy1IzOa9VYBuI8TgMFt2GRDkDuJh4=:",
    "signature-input": (
        'sig1=("@request-target" "@authority" "audiohook-organization-id" "audiohook-session-id" '
        '"audiohook-correlation-id" "x-api-key");keyid="SGVsbG8sIEkgYW0gdGhlIEFQSSBrZXkh";'
        'nonce="VGhpc0lzQVVuaXF1ZU5vbmNl";alg="hmac-sha256";created=1641013200;expires=3282026430'
    ),
}

SESSION_ID = "30b0e395-84d3-4570-ac13-9a62d8f514c0"
OPEN_PARAMETERS = {
    "organizationId": "d7934305-0972-4844-938e-9060eef73d05",
    "conversationId": "090eaa2f-72fa-480a-83e0-8667ff89c0ec",
    "participant": {"id": "883efee8-3d6c-4537-b500-6d7ca4b92fa0", "ani": "+15555550100", "aniName": "", "dnis": ""},
    "media": [{"type": "audio", "format": "PCMU", "channels": ["external"], "rate": 8000}],
}


def _client_message(msg_type: str, seq: int, parameters: dict | None = None) -> str:
    return json.dumps(
        {
            "version": "2",
            "id": SESSION_ID,
            "type": msg_type,
            "seq": seq,
            "serverseq": 0,
            "position": "PT0S",
            "parameters": parameters or {},
        }
    )


def _verify(headers=DOC_HEADERS, *, now=DOC_NOW, secret=DOC_CLIENT_SECRET, target=DOC_TARGET, nonces=None) -> None:
    verify_signature(
        headers,
        request_target=target,
        authority=headers.get("host", ""),
        api_key=DOC_API_KEY,
        client_secret=secret,
        now=now,
        nonces=nonces or NonceCache(),
    )


def _new_serializer() -> AudioConnectorSerializer:
    return AudioConnectorSerializer(GenesysAudioHookSerializer.InputParams(sample_rate=16000))


def _pcm_frame(samples: int = 1600) -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=b"\x00\x00" * samples, sample_rate=AUDIOHOOK_SAMPLE_RATE, num_channels=1)


class SignatureTests(unittest.TestCase):
    def test_accepts_genesys_documentation_example(self) -> None:
        _verify()

    def test_header_order_and_case_do_not_matter(self) -> None:
        headers = {name.upper(): value for name, value in reversed(list(DOC_HEADERS.items()))}
        result = authenticate_request(
            headers,
            request_target=DOC_TARGET,
            authority="audiohook.example.com",
            config=AudioHookAuthConfig(api_key=DOC_API_KEY, client_secret=DOC_CLIENT_SECRET),
            now=DOC_NOW,
            nonces=NonceCache(),
        )
        self.assertTrue(result.ok, result.reason)

    def test_rejects_tampered_header_target_or_secret(self) -> None:
        with self.assertRaisesRegex(SignatureError, "mismatch"):
            _verify({**DOC_HEADERS, "audiohook-session-id": "00000000-0000-0000-0000-000000000000"})
        with self.assertRaisesRegex(SignatureError, "mismatch"):
            _verify(target="/api/ws")
        with self.assertRaisesRegex(SignatureError, "mismatch"):
            _verify(secret="d3Jvbmctc2VjcmV0LXZhbHVlLTEyMzQ1Njc4OTAxMjM=")

    def test_rejects_stale_signature_and_replayed_nonce(self) -> None:
        with self.assertRaisesRegex(SignatureError, "too old"):
            _verify(now=DOC_NOW + 60)
        nonces = NonceCache()
        _verify(nonces=nonces)
        with self.assertRaisesRegex(SignatureError, "replayed"):
            _verify(nonces=nonces)

    def test_rejects_signature_that_skips_required_components(self) -> None:
        signature_input = DOC_HEADERS["signature-input"].replace(' "x-api-key"', "")
        with self.assertRaisesRegex(SignatureError, "does not cover x-api-key"):
            _verify({**DOC_HEADERS, "signature-input": signature_input})

    def test_sign_request_round_trips(self) -> None:
        headers = {name: value for name, value in DOC_HEADERS.items() if not name.startswith("signature")}
        headers |= sign_request(
            headers,
            request_target="/api/ws/demo",
            authority="voice.example.com",
            api_key=DOC_API_KEY,
            client_secret=DOC_CLIENT_SECRET,
            created=1_700_000_000,
            nonce="n" * 24,
        )
        verify_signature(
            headers,
            request_target="/api/ws/demo",
            authority="voice.example.com",
            api_key=DOC_API_KEY,
            client_secret=DOC_CLIENT_SECRET,
            now=1_700_000_001,
            nonces=NonceCache(),
        )


class ApiKeyTests(unittest.TestCase):
    def _auth(self, headers: dict, **config):
        return authenticate_request(
            headers,
            request_target="/api/ws/demo",
            authority="voice.example.com",
            config=AudioHookAuthConfig(**config),
            nonces=NonceCache(),
        )

    def test_api_key_must_match(self) -> None:
        self.assertTrue(self._auth({"X-API-KEY": "secret-key"}, api_key="secret-key").ok)
        self.assertFalse(self._auth({"x-api-key": "wrong"}, api_key="secret-key").ok)
        self.assertFalse(self._auth({}, api_key="secret-key").ok)

    def test_fails_closed_unless_unauthenticated_access_is_explicit(self) -> None:
        rejected = self._auth({"x-api-key": "anything"})
        self.assertFalse(rejected.ok)
        self.assertIn("GENESYS_AUDIOHOOK_API_KEY", rejected.reason)
        allowed = self._auth({}, allow_unauthenticated=True)
        self.assertTrue(allowed.ok)
        self.assertIn("unauthenticated", allowed.reason)

    def test_client_secret_requires_a_signature(self) -> None:
        result = self._auth({"x-api-key": DOC_API_KEY}, api_key=DOC_API_KEY, client_secret=DOC_CLIENT_SECRET)
        self.assertFalse(result.ok)
        self.assertIn("signature", result.reason)

    def test_config_from_env(self) -> None:
        config = AudioHookAuthConfig.from_env(
            {"GENESYS_AUDIOHOOK_API_KEY": " key ", "GENESYS_AUDIOHOOK_ALLOW_UNAUTHENTICATED": "TRUE"}
        )
        self.assertEqual(config.api_key, "key")
        self.assertTrue(config.allow_unauthenticated)

    def test_request_target_uses_raw_path_and_query(self) -> None:
        scope = {"raw_path": b"/api/ws/a%20b", "path": "/api/ws/a b", "query_string": b"x=1"}
        self.assertEqual(request_target_from_scope(scope), "/api/ws/a%20b?x=1")

    def test_unauthorized_disconnect_message(self) -> None:
        message = unauthorized_disconnect_message(SESSION_ID)
        self.assertEqual((message["type"], message["id"]), ("disconnect", SESSION_ID))
        self.assertEqual(message["parameters"], {"reason": "unauthorized"})
        self.assertNotEqual(unauthorized_disconnect_message("not-a-uuid")["id"], "not-a-uuid")


class SerializerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.serializer = _new_serializer()
        await self.serializer.setup(SimpleNamespace(audio_in_sample_rate=16000))
        frame = await self.serializer.deserialize(_client_message("open", 1, OPEN_PARAMETERS))
        self.assertIsInstance(frame, OutputTransportMessageUrgentFrame)
        self.opened = frame.message

    async def test_open_selects_external_pcmu(self) -> None:
        self.assertEqual(self.opened["type"], "opened")
        self.assertEqual(self.opened["parameters"]["media"], OPEN_PARAMETERS["media"])
        self.assertTrue(self.serializer.is_open)

    async def test_audio_is_converted_in_both_directions(self) -> None:
        inbound = await self.serializer.deserialize(b"\xff" * 1600)
        self.assertIsInstance(inbound, InputAudioRawFrame)
        self.assertEqual(inbound.sample_rate, 16000)
        outbound = await self.serializer.serialize(_pcm_frame(1600))
        self.assertEqual(len(outbound), AUDIOHOOK_PACKET_BYTES)

    async def test_client_paused_and_resumed_gate_audio_and_barge_in(self) -> None:
        events: list[str] = []

        async def record(serializer, message):
            events.append(message["type"])

        self.serializer.add_event_handler("on_paused", record)
        self.serializer.add_event_handler("on_resumed", record)
        self.assertIsNone(await self.serializer.deserialize(_client_message("paused", 2)))
        self.assertTrue(self.serializer.is_paused)
        self.assertIsNone(await self.serializer.serialize(_pcm_frame()))
        self.assertIsNone(await self.serializer.serialize(InterruptionFrame()))
        self.assertIsNone(await self.serializer.deserialize(b"\xff" * 1600))
        resumed = _client_message("resumed", 3, {"start": "PT1S", "discarded": "PT0.5S"})
        self.assertIsNone(await self.serializer.deserialize(resumed))
        await asyncio.sleep(0.01)
        self.assertFalse(self.serializer.is_paused)
        self.assertEqual(events, ["paused", "resumed"])
        self.assertIsNotNone(await self.serializer.serialize(_pcm_frame()))

    async def test_discarded_is_accepted_without_state_change(self) -> None:
        discarded = _client_message("discarded", 2, {"start": "PT1S", "discarded": "PT0.2S"})
        self.assertIsNone(await self.serializer.deserialize(discarded))
        self.assertFalse(self.serializer.is_paused)

    async def test_end_frame_sends_one_disconnect_without_pipecat_action_variable(self) -> None:
        payload = json.loads(await self.serializer.serialize(EndFrame()))
        self.assertEqual((payload["type"], payload["parameters"]), ("disconnect", {"reason": "completed"}))
        self.assertIsNone(await self.serializer.serialize(EndFrame()))

    async def test_cancel_frame_reports_error_with_output_variables(self) -> None:
        self.serializer.set_output_variables({"intent": "billing"})
        payload = json.loads(await self.serializer.serialize(CancelFrame()))
        self.assertEqual(payload["parameters"]["reason"], "error")
        self.assertEqual(payload["parameters"]["outputVariables"], {"intent": "billing"})

    async def test_nothing_is_sent_after_genesys_closes(self) -> None:
        closed = await self.serializer.deserialize(_client_message("close", 2, {"reason": "end"}))
        self.assertEqual(closed.message["type"], "closed")
        self.assertTrue(self.serializer.close_received)
        self.assertIsNone(await self.serializer.serialize(EndFrame()))
        self.assertIsNone(await self.serializer.serialize(CancelFrame()))
        self.assertIsNone(await self.serializer.serialize(_pcm_frame()))

    async def test_ping_is_answered_with_client_sequence(self) -> None:
        pong = await self.serializer.deserialize(_client_message("ping", 7))
        self.assertEqual((pong.message["type"], pong.message["clientseq"]), ("pong", 7))


class SessionProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def _opened_serializer(self) -> AudioConnectorSerializer:
        serializer = _new_serializer()
        await serializer.deserialize(_client_message("open", 1, OPEN_PARAMETERS))
        return serializer

    async def test_end_request_becomes_disconnect_while_session_is_open(self) -> None:
        session = AudioHookSessionProcessor(
            await self._opened_serializer(), open_timeout_secs=0, disconnect_timeout_secs=60
        )
        down, _ = await run_test(
            session,
            frames_to_send=[EndWorkerFrame(reason="user inactive")],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_down_frames=[OutputTransportMessageUrgentFrame],
            expected_up_frames=[],
        )
        self.assertEqual(down[0].message["type"], "disconnect")
        self.assertEqual(down[0].message["parameters"], {"reason": "completed", "info": "user inactive"})
        self.assertTrue(session.hangup_requested)
        # The shared activity check still pushes the deprecated EndTaskFrame subclass.
        self.assertTrue(issubclass(EndTaskFrame, EndWorkerFrame))

    async def test_end_request_passes_through_before_open(self) -> None:
        session = AudioHookSessionProcessor(_new_serializer(), open_timeout_secs=0)
        await run_test(
            session,
            frames_to_send=[EndWorkerFrame()],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_down_frames=[],
            expected_up_frames=[EndWorkerFrame],
        )

    async def test_open_starts_the_session_once(self) -> None:
        serializer = _new_serializer()
        opened: list[str] = []

        async def on_open(message: dict) -> None:
            opened.append(message["id"])

        AudioHookSessionProcessor(serializer, on_session_open=on_open, open_timeout_secs=0)
        await serializer.deserialize(_client_message("open", 1, OPEN_PARAMETERS))
        await serializer.deserialize(_client_message("open", 2, OPEN_PARAMETERS))
        await asyncio.sleep(0.01)
        self.assertEqual(opened, [SESSION_ID])

    async def test_missing_open_cancels_the_pipeline(self) -> None:
        session = AudioHookSessionProcessor(_new_serializer(), open_timeout_secs=0.01)
        pushed: list[tuple] = []

        async def capture(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append((frame, direction))

        session.push_frame = capture
        await session._expire_open()
        self.assertIsInstance(pushed[0][0], CancelWorkerFrame)
        self.assertEqual(pushed[0][1], FrameDirection.UPSTREAM)


class DeploymentWiringTests(unittest.TestCase):
    def test_packets_are_200_ms_of_8_khz_pcmu(self) -> None:
        self.assertEqual(AUDIOHOOK_SAMPLE_RATE * 10 * AUDIOHOOK_OUT_10MS_CHUNKS // 1000, AUDIOHOOK_PACKET_BYTES)
        transport = create_audiohook_transport(SimpleNamespace(), _new_serializer(), audio_in_sample_rate=16000)
        self.assertEqual(transport._params.fixed_audio_packet_size, AUDIOHOOK_PACKET_BYTES)
        self.assertEqual(transport._params.audio_out_sample_rate, AUDIOHOOK_SAMPLE_RATE)

    def test_registry_compose_and_catalog_agree(self) -> None:
        entry = yaml.safe_load((REPO_ROOT / "examples_registry.yaml").read_text())["examples"][
            "genesys-audiohook-assistant"
        ]
        self.assertEqual(entry["bot"], "examples.genesys_audiohook.pipeline:bot")
        prompts = yaml.safe_load((REPO_ROOT / "src/examples/genesys_audiohook/prompts.yaml").read_text())
        self.assertIn(entry["defaults"]["prompt"][0], prompts)
        catalog = yaml.safe_load((REPO_ROOT / "services.yaml").read_text())["server"]
        for slot, service_ids in entry["services"].items():
            for service_id in service_ids:
                self.assertIn(service_id, catalog[slot])
        services = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())["services"]
        environment = services["genesys-audiohook-assistant"]["environment"]
        self.assertEqual(environment["EXAMPLE_SELECTION"], "genesys-audiohook-assistant")
        self.assertEqual(environment["TRANSPORT_SELECTION"], "websocket")
        self.assertEqual(environment["SERVICE_RECIPE"], "cloud")
