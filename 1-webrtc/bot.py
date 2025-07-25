import os
from dotenv import load_dotenv
from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from typing import List
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext
)
from llm_client import CustomLLMService, _stream_chat_completions_patched
from stt_client import CustomSTTService
import aiohttp

BaseOpenAILLMService._stream_chat_completions = _stream_chat_completions_patched
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.audio.filters.noisereduce_filter import NoisereduceFilter
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from tts_client import CustomTTSService
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor, RTVIServerMessageFrame
from pipecat.processors.user_idle_processor import UserIdleProcessor
from typing import Optional, List
from pipecat.frames.frames import TranscriptionMessage, TranscriptionUpdateFrame
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.frames.frames import LLMMessagesFrame, TTSSpeakFrame, TTSStoppedFrame, EndFrame
from turn_client import CustomSmartTurnAnalyzer
from pipecat.metrics.metrics import SmartTurnMetricsData
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from pipecat.frames.frames import Frame, MetricsFrame
from datetime import datetime

load_dotenv(override=True)

RECORDS_DIR = os.getenv("RECORDS_DIR")

# #### **Step 2: Follow the conversation**
# - Just follow the conversation. If the caller ends the conversation, **IMMEDIATELY** call the `terminate_call` function.
# - If the caller intention is marketing / spam / scam, just repond that you're not interested and you can end the conversation by calling the `terminate_call` function.

SYSTEM_PROMPT = """
Your name is {target}, act as a person who is friendly.

Follow these steps **EXACTLY**.

#### **Step 1: Greeting**
- Greet caller with: "Halo {target} disini, maaf siapa ya? ada yang bisa dibantu?"

#### **Step 2: Follow the conversation**
- Just follow the conversation. If the caller ends the conversation, Then end call by thanking and be friendly.
- If the caller intention is marketing / spam / scam, just repond that you're not interested and you can end the conversation.
- On end call, you can say "Okay saya tutup ya, terima kasih."

### **General Rules**
- Answers in Bahasa Indonesia and concise.
- Your responses should sound natural and consistent with spoken conversation.
- Your output will be converted to audio, so **do not include special characters or formatting.** or any markdown formatting.
- Do not use exclamation sentences such as Wah!, you can only use plain text end with . or question sentence end with ?
"""

class SmartTurnMetricsProcessor(FrameProcessor):
    """Processes the metrics data from Smart Turn Analyzer.

    This processor is responsible for handling smart turn metrics data
    and forwarding it to the client UI via RTVI.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle Smart Turn metrics.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Handle Smart Turn metrics
        if isinstance(frame, MetricsFrame):
            for metrics in frame.data:
                if isinstance(metrics, SmartTurnMetricsData):
                    logger.info(f"Smart Turn metrics: {metrics}")

                    # Create a payload with the smart turn prediction data
                    smart_turn_data = {
                        "type": "smart_turn_result",
                        "is_complete": metrics.is_complete,
                        "probability": metrics.probability,
                        "inference_time_ms": metrics.inference_time_ms,
                        "server_total_time_ms": metrics.server_total_time_ms,
                        "e2e_processing_time_ms": metrics.e2e_processing_time_ms,
                    }

                    # Send the data to the client via RTVI
                    rtvi_frame = RTVIServerMessageFrame(data=smart_turn_data)
                    await self.push_frame(rtvi_frame)

        await self.push_frame(frame, direction)


class TranscriptHandler:
    def __init__(self, target: str):
        self.messages: List[TranscriptionMessage] = []
        self.target: str = target
        # Create filename with voice name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.target}_{timestamp}.log"
        self.output_file: Optional[str] = RECORDS_DIR + "/" + filename
        logger.debug(
            f"TranscriptHandler initialized {'with output_file=' + self.output_file if self.output_file else 'with log output only'}"
        )

    async def save_message(self, message: TranscriptionMessage):
        mapped_role = {
            "user": "Caller",
            "assistant": self.target,
        }
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}{mapped_role.get(message.role, 'Unknown')}: {message.content}"
        logger.info(f"Transcript: {line}")

        if self.output_file:
            try:
                os.makedirs(RECORDS_DIR, exist_ok=True)
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.error(f"Error saving transcript message to file: {e}")

    async def on_transcript_update(
        self, processor: TranscriptProcessor, frame: TranscriptionUpdateFrame
    ):
        logger.debug(f"Received transcript update with {len(frame.messages)} new messages")
        for msg in frame.messages:
            self.messages.append(msg)
            await self.save_message(msg)


async def run_bot(webrtc_connection, target):
    # Temporary
    if target is None:
        target = "ono"

    logger.info(f"Starting bot for {target}")

    md_filter = MarkdownTextFilter(
        params=MarkdownTextFilter.InputParams(
            filter_code=True,
            filter_tables=True
        )
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_filter=NoisereduceFilter(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.3)),
            turn_analyzer=CustomSmartTurnAnalyzer(
                aiohttp_session=aiohttp.ClientSession(),
                base_url=os.getenv("BASE_URL_STT"),
            )
        ),
    )

    stt = CustomSTTService(
        base_url=os.getenv("BASE_URL_STT"),
    )

    llm = CustomLLMService(
        base_url=os.getenv("BASE_URL_LLM"),
    )

    tts = CustomTTSService(
        base_url=os.getenv("BASE_URL_TTS"),
        voice=target,
        text_filters=[md_filter]
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    smart_turn_metrics_processor = SmartTurnMetricsProcessor()

    async def handle_user_idle(user_idle: UserIdleProcessor, retry_count: int) -> bool:
        if retry_count == 1:
            await user_idle.push_frame(
                TTSSpeakFrame("Hai, masih disitu?")
            )
            return True
        elif retry_count == 2:
            await user_idle.push_frame(
                TTSSpeakFrame("Okay saya tutup ya, terima kasih.")
            )
            await task.queue_frame(EndFrame())
            return False

    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=15)

    async def terminate_call(
        task: PipelineTask,  # Pipeline task reference
        params: FunctionCallParams,
    ):
        """Function the bot can call to terminate the call."""
        # Create a message to add
        content = "The user wants to end the conversation, thank them for chatting."
        message = {
            "role": "system",
            "content": content,
        }
        # Append the message to the list
        messages.append(message)
        # Queue the message to the context
        await task.queue_frames([LLMMessagesFrame(messages)])

        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )
    messages = [
        {
            "role": "user",
            "content": SYSTEM_PROMPT.format(target=target),
        },
    ]
    logger.info(f"System prompt: {messages}")

    # tools = ToolsSchema(standard_tools=[terminate_call_function])
    # llm.register_function("terminate_call", lambda params: terminate_call(task, params))
    # context = OpenAILLMContext(messages, tools)
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    transcript = TranscriptProcessor()
    transcript_handler = TranscriptHandler(target=target)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            # stt_mute_processor,
            rtvi,
            smart_turn_metrics_processor,
            stt,
            user_idle,
            transcript.user(),  # User transcripts
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            transcript.assistant(),  # Assistant transcripts
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            start_metadata={
                "target": target
            }
        ),
        observers=[RTVIObserver(rtvi)],
    )


    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        messages = {
            "show_text_container": True,
            "show_video_container": False,
            "show_debug_container": True,
        }

        rtvi_frame = RTVIServerMessageFrame(data=messages)
        await task.queue_frames([rtvi_frame])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        messages.append({"role": "system", "content": f"Start the conversation with something like: Halo {target} disini, maaf siapa ya? ada yang bisa dibantu?"})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Client closed")
        await task.cancel()

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        await transcript_handler.on_transcript_update(processor, frame)

    runner = PipelineRunner(handle_sigint=True)

    await runner.run(task)
