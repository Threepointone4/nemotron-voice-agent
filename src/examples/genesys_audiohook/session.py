# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AudioHook session lifecycle for Genesys Audio Connector calls."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    CancelWorkerFrame,
    EndWorkerFrame,
    Frame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.genesys_audiohook.serializer import AudioConnectorSerializer

SessionOpenCallback = Callable[[dict[str, Any]], Awaitable[None]]


class AudioHookSessionProcessor(FrameProcessor):
    """Drive the AudioHook session lifecycle from inside the pipeline.

    Place it directly after ``transport.input()``. It:

    - starts the conversation when Genesys sends ``open``, and ends the pipeline if
      ``open`` never arrives (for example, a browser connected to the shared route);
    - turns upstream end requests, such as the activity check's inactivity hang-up,
      into the AudioHook ``disconnect``, ``close``, ``closed`` handshake, so Genesys
      takes the success path and receives any output variables;
    - interrupts the bot when Genesys pauses the stream.
    """

    def __init__(
        self,
        serializer: AudioConnectorSerializer,
        *,
        on_session_open: SessionOpenCallback | None = None,
        open_timeout_secs: float = 10.0,
        disconnect_timeout_secs: float = 5.0,
        close_timeout_secs: float = 3.0,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            serializer: Serializer attached to the Genesys WebSocket transport.
            on_session_open: Called with the ``open`` message once the session starts.
            open_timeout_secs: End the pipeline if no ``open`` arrives in time; 0 disables.
            disconnect_timeout_secs: Time Genesys has to close the session after ``disconnect``.
            close_timeout_secs: Time Genesys has to drop the WebSocket after ``closed``.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._serializer = serializer
        self._on_session_open = on_session_open
        self._open_timeout_secs = open_timeout_secs
        self._disconnect_timeout_secs = disconnect_timeout_secs
        self._close_timeout_secs = close_timeout_secs
        self._session_opened = False
        self._hangup_requested = False
        self._open_timer: asyncio.Task | None = None
        self._end_timer: asyncio.Task | None = None
        serializer.add_event_handler("on_open", self._handle_open)
        serializer.add_event_handler("on_close", self._handle_close)
        serializer.add_event_handler("on_paused", self._handle_paused)

    @property
    def hangup_requested(self) -> bool:
        """Whether the bot has asked Genesys to end the session."""
        return self._hangup_requested

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Arm the open timeout and intercept end requests while the session is open."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._open_timeout_secs > 0 and not self._session_opened:
                self._open_timer = self.create_task(self._expire_open(), "audiohook-open-timeout")
            return
        if direction == FrameDirection.UPSTREAM and isinstance(frame, EndWorkerFrame) and self._serializer.is_open:
            await self.hangup(info=getattr(frame, "reason", None) or None)
            return
        await self.push_frame(frame, direction)

    async def hangup(
        self,
        reason: str = "completed",
        *,
        info: str | None = None,
        output_variables: Mapping[str, Any] | None = None,
    ) -> None:
        """Ask Genesys to end the session, then let it run the close transaction.

        Args:
            reason: AudioHook disconnect reason: ``completed`` or ``error``.
            info: Optional human-readable detail.
            output_variables: Values returned to the Architect flow. Defaults to the
                serializer's output variables.
        """
        if self._hangup_requested:
            return
        self._hangup_requested = True
        if not self._serializer.is_open or self._serializer.close_received:
            await self._force_end("the AudioHook session is not open")
            return
        variables = self._serializer.output_variables if output_variables is None else output_variables
        message = self._serializer.build_disconnect_message(reason, info=info, output_variables=variables)
        logger.info(f"Asking Genesys to end the session (reason={reason}, info={info or '-'})")
        await self.push_frame(OutputTransportMessageUrgentFrame(message=message))
        self._schedule_end(self._disconnect_timeout_secs, "Genesys did not close the session after disconnect")

    async def cleanup(self) -> None:
        """Cancel pending timers."""
        for timer in (self._open_timer, self._end_timer):
            if timer is not None and not timer.done():
                await self.cancel_task(timer)
        await super().cleanup()

    async def _handle_open(self, serializer: AudioConnectorSerializer, message: dict[str, Any]) -> None:
        if self._session_opened:
            return
        self._session_opened = True
        if self._open_timer is not None and not self._open_timer.done():
            self._open_timer.cancel()
        if self._on_session_open is not None:
            await self._on_session_open(message)

    async def _handle_close(self, serializer: AudioConnectorSerializer, message: dict[str, Any]) -> None:
        reason = (message.get("parameters") or {}).get("reason", "unknown")
        logger.info(f"Genesys closed the AudioHook session (reason={reason})")
        self._schedule_end(self._close_timeout_secs, "Genesys did not drop the WebSocket after closed")

    async def _handle_paused(self, serializer: AudioConnectorSerializer, message: dict[str, Any]) -> None:
        await self.broadcast_interruption()

    def _schedule_end(self, delay_secs: float, reason: str) -> None:
        if self._end_timer is not None and not self._end_timer.done():
            self._end_timer.cancel()
        self._end_timer = self.create_task(self._end_after(delay_secs, reason), "audiohook-end-timeout")

    async def _end_after(self, delay_secs: float, reason: str) -> None:
        await asyncio.sleep(delay_secs)
        await self._force_end(reason)

    async def _expire_open(self) -> None:
        await asyncio.sleep(self._open_timeout_secs)
        if not self._session_opened:
            await self._force_end(f"no AudioHook open message within {self._open_timeout_secs:g}s")

    async def _force_end(self, reason: str) -> None:
        logger.warning(f"Ending the Genesys pipeline: {reason}")
        await self.push_frame(CancelWorkerFrame(reason=reason), FrameDirection.UPSTREAM)
