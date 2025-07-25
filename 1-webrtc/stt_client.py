from typing import Optional
from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from pipecat.transcriptions.language import Language

class CustomSTTService(BaseWhisperSTTService):
    def __init__(
        self,
        *,
        model: str = "dummy",
        api_key: Optional[str] = "dummy",
        base_url: Optional[str] = None,
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            prompt=prompt,
            temperature=temperature,
            **kwargs,
        )

    async def _transcribe(self, audio: bytes) -> Transcription:
        assert self._language is not None  # Assigned in the BaseWhisperSTTService class

        # Build kwargs dict with only set parameters
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": "dummy",
            "language": "id",
        }

        return await self._client.audio.transcriptions.create(**kwargs)
