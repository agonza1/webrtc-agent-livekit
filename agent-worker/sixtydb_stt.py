"""60db (60db.ai) Speech-to-Text plugin for the LiveKit Agents framework.

60db's STT is a REST, file-upload (batch) API (`POST https://api.60db.ai/stt`),
not a live websocket like Deepgram. So this plugin implements the **non-streaming**
`livekit.agents.stt.STT` interface (`recognize()` / `_recognize_impl()`).

When you pass a non-streaming STT to an ``AgentSession`` that also has a VAD
(this project uses Silero VAD), the framework automatically wraps it with a
``StreamAdapter``: the VAD slices the mic audio into utterances and calls
``recognize()`` once per utterance. So it works in a real-time agent, but with
higher latency than a streaming STT — Deepgram remains the recommended default;
60db STT is offered here as a drop-in alternative.

Each recognize call uploads the utterance as a WAV file and parses the JSON
transcript (see https://docs.60db.ai/api-reference/stt/speech-to-text).

Environment variables:
    SIXTYDB_API_KEY   - your 60db API key (required if ``api_key`` not passed)
    SIXTYDB_BASE_URL  - override the API base URL (optional)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

DEFAULT_BASE_URL = "https://api.60db.ai"
STT_PATH = "/stt"
DEFAULT_LANGUAGE = "en"  # fallback when 60db returns no detected language


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    language: str  # ISO 639-1 code, or "auto" for auto-detection
    diarize: bool


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        language: str = "auto",
        diarize: bool = False,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new 60db STT.

        Args:
            api_key: 60db API key. Falls back to ``SIXTYDB_API_KEY``.
            language: ISO 639-1 language code, or ``"auto"`` for auto-detection.
            diarize: enable speaker diarization (adds speaker labels to segments).
            base_url: override the API base URL (or ``SIXTYDB_BASE_URL``).
            http_session: reuse an existing aiohttp session.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        sixtydb_api_key = api_key if is_given(api_key) else os.environ.get("SIXTYDB_API_KEY")
        if not sixtydb_api_key:
            raise ValueError(
                "60db API key is required, either pass api_key= or set the "
                "SIXTYDB_API_KEY environment variable"
            )

        self._opts = _STTOptions(
            api_key=sixtydb_api_key,
            base_url=base_url if is_given(base_url) else os.environ.get("SIXTYDB_BASE_URL", DEFAULT_BASE_URL),
            language=language,
            diarize=diarize,
        )
        self._session = http_session

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

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        diarize: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(diarize):
            self._opts.diarize = diarize

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language

        # combine the utterance frames into a single WAV (sample rate is embedded)
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        if lang and lang != "auto":
            form.add_field("language", lang)
        if self._opts.diarize:
            form.add_field("diarize", "true")

        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}{STT_PATH}",
                data=form,
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ServerTimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

        text = data.get("text", "") or ""
        detected_lang = data.get("language") or (lang if lang != "auto" else None) or DEFAULT_LANGUAGE

        # average per-segment confidence when available
        segments = data.get("segments") or []
        confidences = [s["confidence"] for s in segments if isinstance(s.get("confidence"), (int, float))]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=data.get("request_id", ""),
            alternatives=[
                stt.SpeechData(
                    language=detected_lang,
                    text=text,
                    confidence=confidence,
                )
            ],
        )
