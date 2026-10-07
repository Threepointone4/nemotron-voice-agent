# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Genesys Audio Connector serializer with AudioHook protocol fixes.

Pipecat's :class:`~pipecat.serializers.genesys.GenesysAudioHookSerializer`
handles the AudioHook handshake, keep-alives, and PCMU audio. As of Pipecat
1.12.0 it deviates from the protocol in two ways that matter for a voice bot:

* Genesys reports pauses with client ``paused`` and ``resumed`` messages
  (``pause`` and ``resume`` are server requests). The base class ignores them,
  so the bot keeps sending audio into a paused stream.
* Every ``EndFrame`` or ``CancelFrame`` emits ``disconnect`` with a
  Pipecat-specific ``action: transfer`` output variable, even after Genesys has
  already closed the session.

This subclass fixes both and adds :meth:`AudioConnectorSerializer.build_disconnect_message`
so the session processor can end calls with the ``disconnect``, ``close``,
``closed`` handshake.
"""

import json
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, InterruptionFrame
from pipecat.serializers.genesys import AudioHookMessageType, GenesysAudioHookSerializer

_STREAM_STATE_MESSAGES = ("paused", "resumed", "discarded")


class AudioConnectorSerializer(GenesysAudioHookSerializer):
    """AudioHook serializer for Genesys Cloud Audio Connector voice bots.

    Event handlers available in addition to the base class events:

    - on_paused(serializer, message): Genesys paused the audio stream.
    - on_resumed(serializer, message): Genesys resumed the audio stream.
    """

    def __init__(self, params: GenesysAudioHookSerializer.InputParams | None = None, **kwargs):
        """Initialize the serializer.

        Args:
            params: Base serializer configuration.
            **kwargs: Additional arguments passed to the base serializer.
        """
        super().__init__(params, **kwargs)
        self._close_received = False
        self._disconnect_sent = False
        self._register_event_handler("on_paused")
        self._register_event_handler("on_resumed")

    @property
    def close_received(self) -> bool:
        """Whether Genesys has started the close transaction."""
        return self._close_received

    @property
    def disconnect_sent(self) -> bool:
        """Whether a ``disconnect`` message has been sent to Genesys."""
        return self._disconnect_sent

    def build_disconnect_message(
        self,
        reason: str = "completed",
        *,
        info: str | None = None,
        output_variables: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a ``disconnect`` message that asks Genesys to close the session.

        Args:
            reason: AudioHook disconnect reason: ``completed``, ``unauthorized``, or ``error``.
            info: Optional human-readable detail.
            output_variables: Optional values returned to the Architect flow.

        Returns:
            The protocol message, to send in an ``OutputTransportMessageUrgentFrame``.
        """
        parameters: dict[str, Any] = {"reason": reason}
        if info:
            parameters["info"] = info
        if output_variables:
            parameters["outputVariables"] = dict(output_variables)
        self._disconnect_sent = True
        return self._create_message(AudioHookMessageType.DISCONNECT, parameters=parameters)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a frame, dropping messages the protocol does not allow in the current state."""
        if isinstance(frame, (EndFrame, CancelFrame)):
            if self._disconnect_sent or self._close_received or not self.is_open:
                return None
            if isinstance(frame, EndFrame):
                message = self.build_disconnect_message("completed", output_variables=self.output_variables)
            else:
                message = self.build_disconnect_message(
                    "error", info="pipeline cancelled", output_variables=self.output_variables
                )
            return json.dumps(message)
        if isinstance(frame, InterruptionFrame) and (self.is_paused or not self.is_open):
            return None
        return await super().serialize(frame)

    async def _handle_control_message(self, message: dict[str, Any]) -> Frame | None:
        msg_type = message.get("type", "")
        if msg_type == AudioHookMessageType.CLOSE.value:
            self._close_received = True
        elif msg_type in _STREAM_STATE_MESSAGES:
            self._client_seq = message.get("seq", self._client_seq)
            if "position" in message:
                self._position = self._parse_position(message["position"])
            await self._handle_stream_state(msg_type, message)
            return None
        return await super()._handle_control_message(message)

    async def _handle_stream_state(self, msg_type: str, message: dict[str, Any]) -> None:
        if msg_type == "paused" and not self._is_paused:
            self._is_paused = True
            logger.info("Genesys paused the AudioHook stream")
            await self._call_event_handler("on_paused", message)
        elif msg_type == "resumed" and self._is_paused:
            self._is_paused = False
            logger.info("Genesys resumed the AudioHook stream")
            await self._call_event_handler("on_resumed", message)
        elif msg_type == "discarded":
            logger.debug(f"Genesys discarded audio: {message.get('parameters', {})}")
