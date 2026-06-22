"""60db (60db.ai) Text-to-Speech plugin for the LiveKit Agents framework.

This is a self-contained TTS plugin that talks to the 60db streaming WebSocket
TTS API (`wss://api.60db.ai/ws/tts`). It implements the same `livekit.agents.tts`
interface that the built-in plugins (deepgram, cartesia, ...) implement, so it can
be dropped straight into an ``AgentSession(tts=...)``.

We use the WebSocket API with ``LINEAR16`` (16-bit signed little-endian PCM, mono)
output because LiveKit consumes raw PCM frames directly — no client-side decoding
needed, which keeps the path low-latency and reliable.

Protocol (see https://docs.60db.ai/websocket-api/tts):
    -> create_context   (voice + audio config)
    -> send_text        (incremental text, once per sentence)
    -> flush_context    (synthesize the buffered text)
    <- audio_chunk       (base64 PCM, possibly many per flush)
    <- flush_completed   (this flush is done)

Authentication is via the ``apiKey`` query parameter on the WebSocket URL.

Environment variables:
    SIXTYDB_API_KEY   - your 60db API key (required if ``api_key`` not passed)
    SIXTYDB_VOICE_ID  - default voice_id UUID (required if ``voice_id`` not passed)
    SIXTYDB_BASE_URL  - override the API base URL (optional)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

DEFAULT_BASE_URL = "https://api.60db.ai"
WS_PATH = "/ws/tts"

# 60db LINEAR16 == 16-bit signed little-endian PCM, mono. LiveKit wants raw PCM.
DEFAULT_ENCODING = "LINEAR16"
DEFAULT_SAMPLE_RATE = 24000  # supported: 8000, 16000, 24000, 48000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    api_key: str
    voice_id: str
    base_url: str
    encoding: str
    sample_rate: int
    speed: float
    stability: float
    similarity: float

    def get_ws_url(self) -> str:
        # https -> wss, http -> ws
        base = self.base_url.replace("http", "ws", 1)
        return f"{base}{WS_PATH}?apiKey={self.api_key}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        stability: float = 50.0,
        similarity: float = 75.0,
        encoding: str = DEFAULT_ENCODING,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """Create a new 60db TTS.

        Args:
            voice_id: 60db voice UUID (from ``GET /myvoices``). Falls back to the
                ``SIXTYDB_VOICE_ID`` environment variable.
            api_key: 60db API key. Falls back to ``SIXTYDB_API_KEY``.
            sample_rate: PCM sample rate in Hz (8000/16000/24000/48000).
            speed: speech speed multiplier, 0.5-2.0.
            stability: 0-100, lower = more expressive, higher = more consistent.
            similarity: 0-100, voice match fidelity.
            encoding: audio encoding sent to 60db. Only ``LINEAR16``/``PCM`` are
                supported here (raw PCM is what LiveKit consumes directly).
            base_url: override the API base URL (or ``SIXTYDB_BASE_URL``).
            http_session: reuse an existing aiohttp session.
            tokenizer: sentence tokenizer used to flush text progressively.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        sixtydb_api_key = api_key if is_given(api_key) else os.environ.get("SIXTYDB_API_KEY")
        if not sixtydb_api_key:
            raise ValueError(
                "60db API key is required, either pass api_key= or set the "
                "SIXTYDB_API_KEY environment variable"
            )

        sixtydb_voice_id = voice_id if is_given(voice_id) else os.environ.get("SIXTYDB_VOICE_ID")
        if not sixtydb_voice_id:
            raise ValueError(
                "60db voice_id is required, either pass voice_id= or set the "
                "SIXTYDB_VOICE_ID environment variable (list voices via GET /myvoices)"
            )

        if encoding.upper() not in ("LINEAR16", "PCM"):
            raise ValueError(
                f"unsupported encoding {encoding!r}; this plugin streams raw PCM, "
                "use 'LINEAR16' or 'PCM'"
            )

        self._opts = _TTSOptions(
            api_key=sixtydb_api_key,
            voice_id=sixtydb_voice_id,
            base_url=base_url if is_given(base_url) else os.environ.get("SIXTYDB_BASE_URL", DEFAULT_BASE_URL),
            encoding=encoding.upper(),
            sample_rate=sample_rate,
            speed=speed,
            stability=stability,
            similarity=similarity,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        return "60db"

    @property
    def provider(self) -> str:
        return "60db"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        # one socket per synthesis: dropping it cleanly stops 60db from
        # synthesizing/billing an abandoned context after a barge-in.
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(self._opts.get_ws_url()), timeout
        )

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
        similarity: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed
        if is_given(stability):
            self._opts.stability = stability
        if is_given(similarity):
            self._opts.similarity = similarity

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        # 60db is streaming-only here; reuse the stream path for one-shot synthesis.
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _create_context_msg(self, context_id: str) -> str:
        return json.dumps(
            {
                "create_context": {
                    "context_id": context_id,
                    "voice_id": self._opts.voice_id,
                    "audio_config": {
                        "audio_encoding": self._opts.encoding,
                        "sample_rate_hertz": self._opts.sample_rate,
                    },
                    "speed": self._opts.speed,
                    "stability": self._opts.stability,
                    "similarity": self._opts.similarity,
                }
            }
        )

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        context_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        # one LiveKit segment per stream; 60db flushes just chunk it internally
        output_emitter.start_segment(segment_id=context_id)

        input_started = asyncio.Event()
        input_done = asyncio.Event()
        flushes_sent = 0
        flushes_completed = 0

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()

        async def _tokenize_input() -> None:
            # fan raw LLM tokens / flush sentinels into the sentence tokenizer
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal flushes_sent
            await ws.send_str(self._create_context_msg(context_id))
            async for ev in sent_tokenizer_stream:
                await ws.send_str(
                    json.dumps(
                        {"send_text": {"context_id": context_id, "text": ev.token + " "}}
                    )
                )
                await ws.send_str(json.dumps({"flush_context": {"context_id": context_id}}))
                flushes_sent += 1
                self._mark_started()
                input_started.set()
            input_done.set()
            input_started.set()  # unblock recv even when there was no text

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal flushes_completed
            await input_started.wait()
            while True:
                # done once every flush we sent has been acknowledged
                if input_done.is_set() and flushes_completed >= flushes_sent:
                    break

                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "60db connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)

                if "audio_chunk" in data:
                    chunk = data["audio_chunk"]
                    # filter by context_id so a pooled/reused socket never mixes
                    # audio from a previous (e.g. interrupted) context
                    if chunk.get("context_id") == context_id and chunk.get("audioContent"):
                        output_emitter.push(base64.b64decode(chunk["audioContent"]))
                elif "flush_completed" in data:
                    if data["flush_completed"].get("context_id") == context_id:
                        flushes_completed += 1
                elif "error" in data:
                    raise APIError(f"60db returned error: {data['error']}")
                # connection_established / context_created / context_closed -> ignore

            output_emitter.end_input()

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._tts.connect_ws(self._conn_options.timeout)
            tasks = [
                asyncio.create_task(_tokenize_input()),
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                input_started.set()
                await sent_tokenizer_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None and not ws.closed:
                await ws.close()
