# Genesys Cloud Audio Connector Assistant

This example connects Genesys Cloud call flows to the cascaded Nemotron voice pipeline through
[Audio Connector](https://help.genesys.cloud/articles/about-audio-connector/), which streams call audio over the
AudioHook protocol. It also serves as a reference for integrating any WebSocket-based telephony platform with
Pipecat. The following diagram shows the call path:

```text
Caller -> Genesys Cloud -> Architect "Call Audio Connector" action
                              |  WSS, AudioHook v2, PCMU 8 kHz, X-API-KEY (+ signature)
                              v
                   TLS ingress with a public certificate
                              |  ws://<app>:7860/api/ws
                              v
        AudioConnectorSerializer -> Nemotron ASR -> Nemotron LLM -> Magpie TTS
```

Audio Connector uses a bidirectional WebSocket, so Pipecat handles the protocol directly. Unlike the gRPC-based Cisco
Webex BYOVA integration, this example does not need an external adapter, and it does not change any shared server
code. Audio Connector itself is a Genesys Cloud product, and this example provides only the server that it connects
to.

## What This Example Reuses

This example adds a thin Genesys layer to the existing Nemotron voice pipeline instead of building a new voice agent.
The following diagram shows which parts come from Pipecat, which come from this repository, and which are new:

```text
Genesys call
    |
    v
New      auth.py, session.py, transport.py,              Genesys-specific code
         serializer.py (protocol fixes)
    |
    v
Reused   GenesysAudioHookSerializer and                  From Pipecat
         FastAPIWebsocketTransport
    |
    v
Reused   Nemotron ASR -> Nemotron LLM -> Magpie TTS,     From this repository, the same pipeline
         server.py, /api/ws route, shared helpers         as the Generic Assistant
    |
    v
Reused   Nemotron models on NVIDIA NIM                   From the repository service catalog
```

Each layer has a distinct role:

- **Pipecat** provides the Genesys protocol support, including the `open` handshake, keep-alive messages, μ-law audio
  conversion, and barge-in events. The example subclasses the Pipecat serializer only to fix three protocol gaps.
- **This repository** provides the server and its `/api/ws` route, the speech and language services, prompt loading,
  activity checks, and the shared helpers in `src/examples/shared/`. The example does not modify any of them.
- **NVIDIA NIM** serves the models: Nemotron automatic speech recognition (ASR), the Nemotron large language model
  (LLM), and Magpie text-to-speech (TTS). The cloud recipe calls NVIDIA-hosted endpoints, and the repository service
  catalog also defines self-hosted NIM containers for the same models.
- **The example** adds Genesys credential checks, phone-call start and hang-up rules, audio packet settings, and a
  simulator for testing without Genesys.

The service setup in `pipeline.py` repeats the Generic Assistant setup, because each example in this repository is
self-contained.

## How It Works

The following table summarizes how the example handles each part of a call.

| Concern | Behavior |
| --- | --- |
| Routing | Genesys connects to the shared `/api/ws` WebSocket route. The Compose profile locks the server to this example, so every connection runs this pipeline. |
| Authentication | Every connection must send an `X-API-KEY` header that matches `GENESYS_AUDIOHOOK_API_KEY`. When `GENESYS_AUDIOHOOK_CLIENT_SECRET` is set, the server also verifies the HMAC-SHA256 HTTP message signature and rejects signatures older than 10 seconds or with a replayed nonce. Rejected connections receive `disconnect` with reason `unauthorized`. Without a configured API key, the server rejects every connection. |
| Protocol | [`serializer.py`](serializer.py) extends the Pipecat `GenesysAudioHookSerializer`. It answers `open`, `ping`, and `close`, honors the client `paused` and `resumed` messages, and never sends `disconnect` after Genesys closes the session. |
| Audio | Caller audio arrives as 8 kHz μ-law (PCMU) and is resampled to 16 kHz for ASR. TTS audio is resampled once to 8 kHz and sent in 1,600-byte packets of 200 ms, the size Pipecat recommends for Genesys rate limits. |
| Call lifecycle | [`session.py`](session.py) greets the caller after `open`, ends the pipeline if `open` never arrives, and turns inactivity and idle-timeout hang-ups into the `disconnect`, `close`, and `closed` handshake. The Architect action then takes its success path. |

## Build a Similar Integration with Pipecat

Most contact center and telephony platforms that stream call audio over a WebSocket fit the same pattern. Pipecat
ships serializers for several of them, including Twilio, Telnyx, Plivo, Vonage, Exotel, and Genesys. The following
table maps each platform concern to the Pipecat building block that handles it and to the file in this example that
implements it.

| Platform concern | Pipecat building block | File in this example |
| --- | --- | --- |
| WebSocket connection and audio pacing | `FastAPIWebsocketTransport` | [`transport.py`](transport.py) |
| Wire protocol, such as `open`, `ping`, `close`, and audio frames | A `FrameSerializer`: `deserialize()` turns messages into frames, and `serialize()` turns frames into messages | [`serializer.py`](serializer.py) |
| Codec and sample rate | Resampling in the serializer and the transport | [`serializer.py`](serializer.py), [`transport.py`](transport.py) |
| Barge-in | An `InterruptionFrame`, serialized as a platform event | [`serializer.py`](serializer.py) |
| Call start and end | A `FrameProcessor` that reacts to serializer events and intercepts `EndWorkerFrame` | [`session.py`](session.py) |
| Credentials | A check that runs before the pipeline starts | [`auth.py`](auth.py) |
| Data returned to the call flow | `outputVariables` in the `disconnect` message | [`serializer.py`](serializer.py) |

In simplified form, the wiring in [`pipeline.py`](pipeline.py) looks like the following example. The serializer and
the session processor are the only Genesys-specific pieces in the pipeline.

```python
serializer = AudioConnectorSerializer()
transport = create_audiohook_transport(websocket, serializer, audio_in_sample_rate=16000)
session = AudioHookSessionProcessor(serializer, on_session_open=start_conversation)

pipeline = Pipeline(
    [transport.input(), session, stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator]
)
```

To apply the pattern to another platform, complete the following steps:

1. **Start from the protocol reference.** List every message in each direction, the order the platform expects, and
   which side can close the session.
2. **Choose or write a serializer.** Use the Pipecat serializer when one exists, and test it against the protocol
   reference. Subclass it to fix gaps, as `serializer.py` does for pause handling and duplicate `disconnect` messages.
3. **Match audio at the edges.** Telephony audio is usually 8 kHz μ-law, and the pipeline runs at higher rates.
   Resample once in each direction, and size outbound packets to the platform limits.
4. **Model the session lifecycle explicitly.** Start the conversation only after the platform handshake. End calls
   with the platform close transaction instead of dropping the socket, and put a timeout on every wait.
5. **Authenticate before you process audio.** Verify the platform credentials when the connection opens, and reject
   unauthenticated connections by default.
6. **Test without the platform.** Unit-test the serializer with protocol messages, and write a small client, like
   [`simulator.py`](simulator.py), that plays the platform side of a call.

## Run the Example

Set `NVIDIA_API_KEY` and `GENESYS_AUDIOHOOK_API_KEY` in `.env`. Then start the cloud recipe, which uses NVIDIA cloud
endpoints and does not need a GPU:

```bash
docker compose --profile genesys-audiohook-assistant up -d
```

The profile serves plain HTTP and WebSocket on port 7860 (`PIPELINE_TLS=false`), because Genesys connects only to
endpoints with a publicly trusted certificate. Terminate TLS at your ingress and forward traffic to port 7860. Do not
expose port 7860 directly to the internet.

## Smoke Test Without Genesys

[`simulator.py`](simulator.py) plays the Genesys side of a call. It sends the same headers, runs the `open`
transaction, streams caller audio, answers `disconnect` with a `close` transaction, and saves the bot audio. Run the
following command from the repository root:

```bash
uv run python src/examples/genesys_audiohook/simulator.py \
  --url ws://localhost:7860/api/ws --seconds 20 --output bot.wav
```

The simulator reads `GENESYS_AUDIOHOOK_API_KEY` and `GENESYS_AUDIOHOOK_CLIENT_SECRET` from the environment, and it
signs requests when a client secret is set. A healthy session prints `opened`, reports bot audio in 1,600-byte packets,
and ends with `closed`. To talk to the bot, pass `--input caller.wav` with a 16-bit PCM WAV file.

## Configure Genesys Cloud

Audio Connector is a premium Genesys Cloud application that Genesys bills per minute of streaming. It requires a
Genesys Cloud CX license and the Bring Your Own Technology (BYOT) Rate E subscription item, which you request from
Genesys. For details, refer to [Audio Connector pricing](https://help.genesys.cloud/articles/audio-connector-pricing/).

Complete the following steps in Genesys Cloud:

1. In **Admin > Integrations**, add an **Audio Connector** integration.
   - **Base Connection URI**: `wss://voice.example.com/api`.
   - **Credentials**: the API key, and optionally a client secret, that match `GENESYS_AUDIOHOOK_API_KEY` and
     `GENESYS_AUDIOHOOK_CLIENT_SECRET`.
2. In Architect, add a **Call Audio Connector** action from the **Bot** category to an inbound call flow. Select the
   integration and enter `ws` as the connector ID. Genesys appends the connector ID to the base URI, so it connects to
   `wss://voice.example.com/api/ws`, the shared WebSocket route.
3. Route the success and failure paths of the action, publish the flow, and place a test call.

If the first call fails while connecting, check the request path in your ingress access log. Adjust the base URI so
that the final path is `/api/ws`, or rewrite the path at the ingress. A rewritten path invalidates request signatures,
so use API-key authentication in that case.

## Configuration

The following environment variables configure the example.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GENESYS_AUDIOHOOK_API_KEY` | None (required) | API key that Genesys sends in `X-API-KEY`. |
| `GENESYS_AUDIOHOOK_CLIENT_SECRET` | Unset | Base64 client secret. When set, the server requires request signatures. |
| `GENESYS_AUDIOHOOK_ALLOW_UNAUTHENTICATED` | `false` | Accepts connections without credentials. Use only for local testing. |
| `GENESYS_OPEN_TIMEOUT_SECS` | `10` | Ends the session if Genesys does not send `open` in time. |
| `GENESYS_GREETING_INTERRUPTIBLE` | `true` | Lets callers interrupt the greeting. |
| `SILERO_VAD_STOP_SECS` | `0.8` in the Compose profile | Silence that ends a caller turn. Lower it only after you test real phone audio. |
| `ENABLE_WELCOME_MESSAGE` | `true` | Greets the caller when the session opens. |
| `PIPELINE_IDLE_TIMEOUT_SECS` | `600` | Idle time before the bot ends the call. |

The registry entry also enables proactive activity checks. After 30 seconds of caller silence, the bot checks in. It
checks again after 15 more seconds and then ends the call.

## Return Data to Architect

The example sends values from `serializer.set_output_variables()` or `session.hangup(output_variables=...)` as
`outputVariables` in the `disconnect` and `closed` messages. Map them to flow variables in the Call Audio Connector
action, for example to route the call to a queue after the bot finishes. The example does not include a transfer tool.
To add one, register an LLM function whose handler calls
`session.hangup(output_variables={"action": "transfer"})`.

## Production Checklist

Before you take production calls, complete the following tasks:

- Terminate TLS at an ingress with a publicly trusted certificate. If you enable request signatures, configure the
  ingress to preserve the original `Host` header and request path.
- Allowlist the Genesys Audio Connector IP ranges or rate-limit connections at the ingress. The shared WebSocket route
  runs LLM, ASR, and TTS readiness checks before authentication on every new connection.
- Size replicas for peak concurrent calls, because each call runs its own pipeline. Set connection limits at the
  ingress.
- Deploy in or near your Genesys Cloud region. Measure first-response latency, barge-in, and packet delivery on real
  calls before you tune settings.

## Limitations

The example has the following limitations:

- The example ships a cloud Compose recipe only. A recipe for self-hosted NIM services can follow the pattern of the
  other examples in `docker-compose.yml`.
- The default ASR model supports English, and the example ignores language `update` messages from Genesys.
- With `selection: all`, the example appears in the browser UI, but browsers do not speak AudioHook. A browser
  session ends after the open timeout.
- The example supports Audio Connector only. AudioHook Monitor only receives audio and cannot carry bot audio.

## Files

The example contains the following files.

| File | Purpose |
| --- | --- |
| `pipeline.py` | Builds the pipeline and handles call setup. |
| `auth.py` | Verifies API keys and request signatures. |
| `serializer.py` | Fixes AudioHook protocol handling on top of the Pipecat Genesys serializer. |
| `session.py` | Handles the open timeout, graceful hang-ups, and pauses. |
| `transport.py` | Configures the WebSocket transport and packet size. |
| `simulator.py` | Simulates Genesys Audio Connector for local tests. |
| `prompts.yaml` | Defines the phone-oriented system prompt. |
