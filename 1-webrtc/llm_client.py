from loguru import logger

import base64
from typing import List
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext
)
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

# Monkey patching
async def _stream_chat_completions_patched(
    self, context: OpenAILLMContext
) -> AsyncStream[ChatCompletionChunk]:

    def merge_consecutive_messages(messages):
        if not messages:
            return []
        merged = [messages[0].copy()]
        for msg in messages[1:]:
            if msg['role'] == merged[-1]['role']:
                # Merge content, handling both string and list content
                if isinstance(merged[-1]['content'], list) and isinstance(msg['content'], list):
                    merged[-1]['content'].extend(msg['content'])
                elif isinstance(merged[-1]['content'], list):
                    merged[-1]['content'].append(msg['content'])
                elif isinstance(msg['content'], list):
                    merged[-1]['content'] = [merged[-1]['content']] + msg['content']
                else:
                    merged[-1]['content'] += ' ' + msg['content']
            else:
                merged.append(msg.copy())
        return merged


    messages: List[ChatCompletionMessageParam] = context.get_messages()
    messages = merge_consecutive_messages(messages)

    logger.debug(f"{self}: Generating chat {messages}")

    # base64 encode any images
    for message in messages:
        if message.get("mime_type") == "image/jpeg":
            encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
            text = message["content"]
            message["content"] = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]
            del message["data"]
            del message["mime_type"]

    chunks = await self.get_chat_completions(context, messages)

    return chunks


import json
import os

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from loguru import logger
from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class CustomLLMService(OpenAILLMService):
    def __init__(
        self,
        *,
        api_key: str = "dummy",
        base_url: str = "dummy",
        model: str = "dummy",
        stream: bool = False,
        **kwargs,
    ):
        self.stream = stream
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(
            context
        )

        combined_text = ""
        async for chunk in WatchdogAsyncIterator(chunk_stream, manager=self.task_manager):
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                await self.start_llm_usage_metrics(tokens)

            if chunk.choices is None or len(chunk.choices) == 0:
                continue

            await self.stop_ttfb_metrics()

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.tool_calls:
                logger.debug(f"Tool call: {chunk.choices[0].delta.tool_calls}")
                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.index != func_idx:
                    functions_list.append(function_name)
                    arguments_list.append(arguments)
                    tool_id_list.append(tool_call_id)
                    function_name = ""
                    arguments = ""
                    tool_call_id = ""
                    func_idx += 1
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                if self.stream:
                    await self.push_frame(LLMTextFrame(text))
                else:
                    combined_text += text

        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            logger.debug(
                f"Function list: {functions_list}, Arguments list: {arguments_list}, Tool ID list: {tool_id_list}"
            )

            function_calls = []
            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                if function_name == "":
                    continue

                arguments = json.loads(arguments)

                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)

        if not self.stream:
            await self.push_frame(LLMTextFrame(combined_text))   