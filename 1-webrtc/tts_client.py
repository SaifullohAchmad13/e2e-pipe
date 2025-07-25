from typing import AsyncGenerator, Optional

from loguru import logger
from openai import AsyncOpenAI, BadRequestError

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# Temporary
class CustomTTSService(TTSService):
    OPENAI_SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        api_key: Optional[str] = "dummy",
        base_url: Optional[str] = None,
        model: str = "f5tts",
        voice: Optional[str] = None,
        sample_rate: Optional[int] = None,
        instructions: Optional[str] = None,
        **kwargs,
    ):
        if sample_rate and sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS only supports {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {sample_rate}Hz may cause issues."
            )
        super().__init__(sample_rate=sample_rate, aggregate_sentences=False, **kwargs)
        self.set_model_name(model)
        self.set_voice(voice or "ono")
        self._instructions = instructions
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        self.set_model_name(model)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self.sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS requires {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()

            # Setup extra body parameters
            extra_body = {}
            if self._instructions:
                extra_body["instructions"] = self._instructions

            async with self._client.audio.speech.with_streaming_response.create(
                input=text,
                model=self.model_name,
                voice=self._voice_id,
                response_format="wav",
                extra_body=extra_body,
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                CHUNK_SIZE = self.chunk_size
                logger.info(f"{self}: Using chunk size {CHUNK_SIZE}")

                yield TTSStartedFrame()
                async for chunk in r.iter_bytes(CHUNK_SIZE):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                        yield frame
                yield TTSStoppedFrame()
        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")