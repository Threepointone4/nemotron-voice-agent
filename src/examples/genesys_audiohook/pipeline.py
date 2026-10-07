# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Genesys Cloud Audio Connector pipeline: Nemotron ASR -> Nemotron LLM -> Magpie TTS over AudioHook.

Genesys Cloud connects to the shared ``/api/ws`` WebSocket route. Each connection is authenticated
with the Genesys API key (and request signature, when a client secret is set)
before any audio is processed. :class:`AudioConnectorSerializer` speaks the
AudioHook protocol and :class:`AudioHookSessionProcessor` manages the call
lifecycle. The speech pipeline mirrors the Generic Assistant without its
browser-only RTVI features.
"""

import asyncio

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.services.nvidia.stt import NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.genesys_audiohook.auth import AudioHookAuthConfig, authenticate_websocket, reject_unauthorized
from examples.genesys_audiohook.serializer import AudioConnectorSerializer
from examples.genesys_audiohook.session import AudioHookSessionProcessor
from examples.genesys_audiohook.transport import create_audiohook_transport
from examples.shared.activity_check import create_activity_check_processor
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.nvidia_force_eou_stt import NvidiaForceEouSTTService
from examples.shared.nvidia_streaming_llm import NvidiaStreamingLLMService, StreamingLLMUserAggregator
from examples.shared.pipeline_utils import (
    PIPELINE_AUDIO_IN_SAMPLE_RATE,
    apply_pinned_prompt_summary,
    build_context_messages,
    build_pipeline_params,
    build_user_aggregator_params,
)
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    is_streaming_llm_url,
    load_ipa_dictionary,
    load_service_entry,
    normalize_lang_code,
    nvidia_api_key,
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
    resolve_prompt,
)

load_dotenv(override=True)
CHAT_HISTORY_RECENT_TURNS = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)

EXAMPLE_KEY = "genesys-audiohook-assistant"
INTRO_PROMPT = "Greet the caller briefly, introduce yourself as Nemotron, and ask how you can help."


async def bot(runner_args: RunnerArguments) -> None:
    """Authenticate one Genesys Audio Connector connection and run the call."""
    websocket = getattr(runner_args, "websocket", None)
    if websocket is None:
        raise TypeError(f"Genesys AudioHook needs WebSocket runner arguments, got {type(runner_args).__name__}")
    body = runner_args.body if isinstance(runner_args.body, dict) else {}

    auth = authenticate_websocket(websocket, AudioHookAuthConfig.from_env())
    if not auth.ok:
        logger.warning(f"Rejecting Genesys AudioHook connection: {auth.reason}")
        await reject_unauthorized(websocket)
        return
    if auth.reason:
        logger.warning(auth.reason)

    serializer = AudioConnectorSerializer()
    transport = create_audiohook_transport(websocket, serializer, audio_in_sample_rate=PIPELINE_AUDIO_IN_SAMPLE_RATE)
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    prompt_key, base_system_content = resolve_prompt(
        __file__,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    logger.info(f"Starting Genesys AudioHook pipeline (prompt={prompt_key})")
    default_llm = load_service_entry("llm", "")
    default_tts = load_service_entry("tts", "")
    default_asr = load_service_entry("asr", "")

    asr_server = body.get("asr_server", "") or default_asr.get("server", "grpc.nvcf.nvidia.com:443")
    asr_ssl = is_nvcf(asr_server)
    asr_kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": asr_server,
        "use_ssl": asr_ssl,
    }
    asr_function_id = body.get("asr_function_id", "") or default_asr.get("function_id", "")
    asr_model = body.get("asr_model", "") or default_asr.get("model", "")
    asr_language_code = body.get("asr_language_code", "") or default_asr.get("language_code", "")
    asr_automatic_punctuation = str(body.get("asr_automatic_punctuation", "true")).lower() != "false"
    if asr_function_id or asr_model:
        asr_kwargs["model_function_map"] = {
            "function_id": asr_function_id,
            "model_name": asr_model or "custom-asr",
        }
    asr_kwargs["settings"] = NvidiaSTTSettings(automatic_punctuation=asr_automatic_punctuation)
    if asr_language_code:
        asr_kwargs["settings"].language = asr_language_code
    stt = NvidiaForceEouSTTService(**asr_kwargs, stop_history=400)
    logger.info(
        f"ASR: server={asr_server}, ssl={asr_ssl}, function_id={asr_function_id or '(default)'}, "
        f"language={asr_language_code or '(default)'}, punctuation={asr_automatic_punctuation}"
    )

    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3.5-lightning-30b-a3b")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        label="extra_params",
    )
    raw_temperature = body.get("temperature", "")
    if raw_temperature in ("", None):
        raw_temperature = default_llm.get("temperature", "")
    llm_temperature = None
    if raw_temperature not in ("", None):
        try:
            llm_temperature = float(raw_temperature)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid temperature={raw_temperature!r}")
    llm_settings = NvidiaLLMSettings(model=model_id)
    max_tokens = body.get("max_tokens", "") or default_llm.get("max_tokens", "")
    if max_tokens not in ("", None):
        try:
            llm_settings.max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid max_tokens={max_tokens!r}")
    if llm_temperature is not None:
        llm_settings.temperature = llm_temperature
    if extra_params:
        llm_settings.extra = extra_params
    logger.info(
        f"LLM: model={model_id}, base_url={base_url}, "
        f"system_prompt={'<' + system_prompt + '>' if system_prompt else '(none)'}, "
        f"temperature={llm_temperature if llm_temperature is not None else '(default)'}, "
        f"extra_params={extra_params or '(none)'}"
    )
    streaming = is_streaming_llm_url(base_url)
    llm_service = NvidiaStreamingLLMService if streaming else NvidiaLLMService
    llm = llm_service(api_key=nvidia_api_key(), base_url=base_url, settings=llm_settings)

    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    tts_synthesis_mode = body.get("tts_synthesis_mode", "") or default_tts.get("synthesis_mode", "")
    raw_tts_function_id = body.get("tts_function_id")
    tts_function_id = (
        str(raw_tts_function_id) if raw_tts_function_id is not None else default_tts.get("function_id", "")
    )
    tts_model = body.get("tts_model", "") or default_tts.get("model", "")
    tts_language_code = body.get("tts_language_code", "") or default_tts.get("language_code", "")
    if tts_language_code:
        tts_language_code = normalize_lang_code(tts_language_code)
    tts_settings_kwargs: dict = {"voice": tts_voice}
    if tts_synthesis_mode:
        tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode
    if tts_language_code:
        tts_settings_kwargs["language"] = tts_language_code
    tts_kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": tts_server,
        "settings": NvidiaTTSSettings(**tts_settings_kwargs),
        "use_ssl": tts_ssl,
        "text_filters": [NemotronSpeechTextFilter()],
        "custom_dictionary": load_ipa_dictionary(),
    }
    if tts_function_id or tts_model:
        tts_kwargs["model_function_map"] = {"function_id": tts_function_id, "model_name": tts_model}
    tts = NvidiaTTSService(**tts_kwargs)
    logger.info(
        f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"model={tts_model or '(pipecat default)'}, function_id={tts_function_id or '(pipecat default)'}, "
        f"language={tts_language_code or '(pipecat default)'}"
    )

    messages = build_context_messages(base_system_content, system_prompt)
    context = LLMContext(messages)
    preserve_prompt_messages = len(messages)

    user_params = build_user_aggregator_params(welcome_enabled)
    if parse_env_bool("GENESYS_GREETING_INTERRUPTIBLE", default=True):
        # Phone callers often talk over the greeting ("Hello?"), so let them interrupt it.
        user_params.user_mute_strategies = []
    if streaming:
        user_aggregator = StreamingLLMUserAggregator(context, params=user_params)
        assistant_aggregator = LLMAssistantAggregator(context)
    else:
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context, user_params=user_params)

    audio_recorder = create_audio_recorder()

    async def queue_llm_run() -> None:
        await task.queue_frame(LLMRunFrame())

    activity_check = create_activity_check_processor(
        examples_registry.activity_check_config(body.get("pipeline_mode", EXAMPLE_KEY)),
        context=context,
        queue_llm_run=queue_llm_run,
        instruction_role="user",
    )
    logger.info(f"Proactive activity checks: {'enabled' if activity_check else 'disabled'}")

    async def on_session_open(message: dict) -> None:
        parameters = message.get("parameters") or {}
        logger.info(
            f"AudioHook session opened (session={message.get('id')}, "
            f"conversation={parameters.get('conversationId') or '-'})"
        )
        if audio_recorder:
            await audio_recorder.start_recording()
        if activity_check:
            activity_check.start()
        if not welcome_enabled:
            logger.info("Welcome message disabled; waiting for the caller to speak first")
            return
        context.add_message({"role": "user", "content": INTRO_PROMPT})
        await task.queue_frame(LLMRunFrame())

    session = AudioHookSessionProcessor(
        serializer,
        on_session_open=on_session_open,
        open_timeout_secs=parse_env_float("GENESYS_OPEN_TIMEOUT_SECS", 10.0, min_value=0.0),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            session,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            *([activity_check] if activity_check else []),
            *([audio_recorder] if audio_recorder else []),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    summary_lock = asyncio.Lock()

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        async with summary_lock:
            await apply_pinned_prompt_summary(
                context=context,
                llm=llm,
                preserve_prompt_messages=preserve_prompt_messages,
                recent_turns=CHAT_HISTORY_RECENT_TURNS,
                summary_system_prompt=system_prompt,
            )

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech(observer, latency):
        logger.info(f"First bot speech latency: {latency:.3f}s")

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):
        logger.info(f"User→Bot latency: {latency:.3f}s")

    task = PipelineWorker(
        pipeline,
        params=build_pipeline_params(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        cancel_on_idle_timeout=False,
        observers=[latency_observer],
        enable_rtvi=False,
        enable_tracing=IS_TRACING_ENABLED,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
        setup_timeout_secs=120.0,
    )

    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(worker):
        await session.hangup(info="pipeline idle timeout")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Genesys WebSocket disconnected")
        await task.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()
