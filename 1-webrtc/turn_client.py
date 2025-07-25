import aiohttp
from pipecat.audio.turn.smart_turn.http_smart_turn import HttpSmartTurnAnalyzer

class CustomSmartTurnAnalyzer(HttpSmartTurnAnalyzer):
    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str,
        **kwargs,
    ):
        url = f"{base_url}/audio/turn-detect"
        super().__init__(url=url, aiohttp_session=aiohttp_session, headers={}, **kwargs)