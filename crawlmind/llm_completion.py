from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
    TypedDict
)

# For real usage, install the openai package that supports the usage pattern you demonstrated.
# pip install openai
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ----------------------------------------------------------------------
# Type Definitions
# ----------------------------------------------------------------------
@dataclass
class LLMCompletionNonStreamingResponse:
    """ Equivalent to LLMCompletionNonStreamingResponse (TypeScript). """
    content: str
    functionCalls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMCompletionStreamingResponse:
    """
    Equivalent to LLMCompletionStreamingResponse (TypeScript).
    Contains an async iterator for chunked content.
    """
    content: AsyncIterator[str]
    functionCalls: Optional[List[Dict[str, Any]]] = None


# Union type for return values
LLMCompletionResponse = Union[LLMCompletionNonStreamingResponse, LLMCompletionStreamingResponse]


class LLMFunctionParameters(TypedDict, total=False):
    """ Equivalent to the TypeScript interface `LLMFunctionParameters`. """
    type: str
    properties: dict
    description: str
    required: List[str]


@dataclass
class LLMFunction:
    """
    Equivalent to the TypeScript interface `LLMFunction`.
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[LLMFunctionParameters] = None
    strict: Optional[bool] = None


@dataclass
class LLMTool:
    """
    Equivalent to the TypeScript interface `LLMTool`.
    Used to pass function-calling metadata to the model.
    """
    type: str  # typically 'function'
    function: LLMFunction


@dataclass
class LLMCompletionRequest:
    """
    Mirrors the properties from TypeScript's llmCompletion request object.
    """
    model: str
    system_prompt: str
    messages: List[Dict[str, str]]  # e.g. [{"role": "user", "content": "..."}]
    workspaceId: str
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[LLMTool]] = None
    max_tokens: Optional[int] = None


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

async def sleep(ms: int) -> None:
    """Sleep for the specified milliseconds (helper)."""
    await asyncio.sleep(ms / 1000)


# ----------------------------------------------------------------------
# Main LLM logic using the new OpenAI Python SDK usage pattern
# ----------------------------------------------------------------------
# crawlmind/llm_completion.py
# crawlmind/llm_completion.py

async def llmCompletionGeneral(
    request: LLMCompletionRequest,
    stream: bool = False
) -> LLMCompletionResponse:
    """
    Handles LLM completion requests, supporting both streaming and non-streaming responses.
    """
    client = OpenAI()  # Initialize your OpenAI client once per request

    model = request.model
    system_prompt = request.system_prompt
    messages = request.messages
    response_format = request.response_format
    tools = request.tools
    max_tokens = request.max_tokens

    # Build the messages array for the ChatCompletion
    final_messages = [{"role": "system", "content": system_prompt}] + messages

    if model.startswith("gpt-"):
        openai_tools = None
        tool_choice = None
        if tools:
            openai_tools = []
            tool_choice = "auto"
            for tool in tools:
                openai_tools.append({
                    "type": tool.type,
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters
                    }
                })

        if stream:
            # Streaming API call
            try:
                stream_iter = client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    stream=True,
                    tools=openai_tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                )

                async def stream_gen() -> AsyncIterator[str]:
                    # Correctly using async for to iterate over the async generator
                    async for chunk in stream_iter:
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            yield delta.content

                return LLMCompletionStreamingResponse(content=stream_gen())

            except Exception as e:
                logger.error(f"Failed to generate streaming completion: {e}")
                return LLMCompletionStreamingResponse(
                    content=async_empty_generator()
                )
        else:
            # Non-streaming API call
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    tools=openai_tools,
                    tool_choice="auto" if openai_tools else None,
                    max_tokens=max_tokens
                )
                content = ""
                function_calls = []
                if completion.choices and completion.choices[0].message:
                    content = completion.choices[0].message.content or ""

                    if getattr(completion.choices[0].message, "tool_calls", None):
                        for call in completion.choices[0].message.tool_calls:
                            fn_name = call.function.name
                            fn_args_str = call.function.arguments
                            try:
                                fn_args = json.loads(fn_args_str)
                            except json.JSONDecodeError:
                                fn_args = {}
                            function_calls.append({"name": fn_name, "args": fn_args})

                return LLMCompletionNonStreamingResponse(
                    content=content,
                    functionCalls=function_calls if function_calls else None
                )

            except Exception as e:
                logger.error(f"Failed to generate completion: {e}")
                return LLMCompletionNonStreamingResponse(content="")

    else:
        # Handle other model families if necessary
        logger.error(f"Unsupported model: {model}")
        raise ValueError(f"Unsupported model: {model}")

async def llmCompletion(
    request: LLMCompletionRequest
) -> LLMCompletionNonStreamingResponse:
    """
    A Python port of `llmCompletion` from TS. Returns a non-streaming response.
    """
    resp = await llmCompletionGeneral(request, stream=False)
    if isinstance(resp, LLMCompletionNonStreamingResponse):
        return resp
    # Should never happen if code is correct
    raise RuntimeError("llmCompletion: expected a non-streaming result.")


async def llmCompletionStreaming(
    request: LLMCompletionRequest
) -> LLMCompletionStreamingResponse:
    """
    A Python port of `llmCompletionStreaming` from TS, returning an async stream.
    """
    resp = await llmCompletionGeneral(request, stream=True)
    if isinstance(resp, LLMCompletionStreamingResponse):
        return resp
    # Should never happen if code is correct
    raise RuntimeError("llmCompletionStreaming: expected a streaming result.")


async def llmCompletionWithRetry(
    request: LLMCompletionRequest,
    retries: int = 2
) -> LLMCompletionNonStreamingResponse:
    """
    Port of `llmCompletionWithRetry` from TS:
    Retries on empty `content`.
    """
    if retries < 0:
        error_message = f"Failed to get LLM response after {retries} retries."
        logger.error(error_message)
        raise RuntimeError(error_message)

    response = await llmCompletion(request)
    if not response.content:
        logger.error("Received empty content; retrying ...")
        return await llmCompletionWithRetry(request, retries - 1)
    return response


async def llmChatCompletion(
    model: str,
    object_id: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    response_format: str,
    retries: int,
    workspace_id: str,
    tools: Optional[List[LLMTool]] = None,
    max_tokens: Optional[int] = None
) -> Any:
    """
    Port of `llmChatCompletion` from TS. If `response_format == "json_object"`,
    tries to parse the content as JSON; else returns the raw text.
    """
    if retries < 0:
        error_message = f"Failed to get GPT response after {retries} retries for object: {object_id}"
        logger.error(error_message)
        raise RuntimeError(error_message)

    request_obj = LLMCompletionRequest(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        workspaceId=workspace_id,
        response_format={"type": response_format},  # not used heavily here, but included
        tools=tools,
        max_tokens=max_tokens
    )

    response = await llmCompletion(request_obj)
    content = response.content
    if not content:
        logger.error(f"Empty LLM response for {object_id}; retrying...")
        return await llmChatCompletion(
            model=model,
            object_id=object_id,
            system_prompt=system_prompt,
            messages=messages,
            response_format=response_format,
            retries=retries - 1,
            workspace_id=workspace_id,
            tools=tools,
            max_tokens=max_tokens
        )

    if response_format == "json_object":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for object {object_id}: {e}. Retrying...")
            return await llmChatCompletion(
                model=model,
                object_id=object_id,
                system_prompt=system_prompt,
                messages=messages,
                response_format=response_format,
                retries=retries - 1,
                workspace_id=workspace_id,
                tools=tools,
                max_tokens=max_tokens
            )
    return content


# ----------------------------------------------------------------------
# Helper: async generator yielding nothing
# ----------------------------------------------------------------------
async def async_empty_generator() -> AsyncIterator[str]:
    if False:  # This will never run
        yield ""  # no-op
