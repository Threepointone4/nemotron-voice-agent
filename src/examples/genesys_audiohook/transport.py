# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket transport settings for Genesys Audio Connector sessions."""

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from examples.genesys_audiohook.serializer import AudioConnectorSerializer

AUDIOHOOK_SAMPLE_RATE = 8000
# Pipecat recommends 1,600-byte packets (200 ms of 8 kHz PCMU) to stay under Genesys message rate limits.
AUDIOHOOK_PACKET_BYTES = 1600
# 20 x 10 ms of 8 kHz PCM per write serializes to exactly one packet.
AUDIOHOOK_OUT_10MS_CHUNKS = 20


def create_audiohook_transport(
    websocket,
    serializer: AudioConnectorSerializer,
    *,
    audio_in_sample_rate: int,
) -> FastAPIWebsocketTransport:
    """Create the WebSocket transport for one Audio Connector session.

    Bot audio is resampled once, from the TTS rate to 8 kHz, in the transport.
    Each 200 ms write then serializes to one 1,600-byte PCMU packet.

    Args:
        websocket: Accepted FastAPI WebSocket from Genesys Cloud.
        serializer: AudioHook serializer for the session.
        audio_in_sample_rate: Pipeline input rate that caller audio is resampled to.

    Returns:
        The configured transport.
    """
    return FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=audio_in_sample_rate,
            audio_out_enabled=True,
            audio_out_sample_rate=AUDIOHOOK_SAMPLE_RATE,
            audio_out_10ms_chunks=AUDIOHOOK_OUT_10MS_CHUNKS,
            add_wav_header=False,
            serializer=serializer,
            fixed_audio_packet_size=AUDIOHOOK_PACKET_BYTES,
        ),
    )
