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
    TypedDict,
    Iterable,
)

# For real usage, install the openai package that supports the usage pattern you demonstrated.
# pip install openai
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from typing import cast
from pydantic import ValidationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ----------------------------------------------------------------------
# Type Definitions
# ----------------------------------------------------------------------
@dataclass
class LLMCompletionNonStreamingResponse:
    """
    Original non-streaming response, now with an optional parsed field.
    """
    content: str
    functionCalls: Optional[List[Dict[str, Any]]] = None
    parsed: Optional[Any] = None  # <-- new field for structured pars

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
    messages: List[Dict[str, str]]  # e.g. [{"role": "user", "content": "..."}
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

async def llmCompletionGeneral(
    request: LLMCompletionRequest,
    stream: bool = False
) -> LLMCompletionResponse:
    """
    Handles LLM completion requests, supporting both streaming and non-streaming responses.
    Now also supports structured outputs (pydantic) if `request.response_format`
    indicates a pydantic model.
    """
    client = OpenAI()  # Initialize OpenAI client
    model = request.model
    system_prompt = request.system_prompt
    messages = request.messages
    response_format = request.response_format
    tools = request.tools
    max_tokens = request.max_tokens

    # Build the messages array for the ChatCompletion
    final_messages = [{"role": "system", "content": system_prompt}] + messages

    # --- Step A: If not a GPT model, raise
    if not model.startswith("gpt-"):
        logger.error(f"Unsupported model: {model}")
        raise ValueError(f"Unsupported model: {model}")

    # Attempt to build openai_tools if any
    openai_tools = None
    tool_choice = None
    if tools:
        from openai.types.chat import ChatCompletionToolParam
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

    # Check if user wants a Pydantic-based structured parse
    if (
        response_format is not None
        and isinstance(response_format, dict)
        and response_format.get("type") == "pydantic_model"
        and "pydantic_class" in response_format
    ):
        # The new structured parse flow
        if stream:
            # .parse() doesn't currently support streaming
            raise ValueError("Structured parse cannot be used with streaming.")

        pydantic_model = response_format["pydantic_class"]
        try:
            # Beta API for structured parse:
            # If your installed openai library has .beta.chat.completions.parse,
            # we can call it directly.
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=final_messages,
                response_format=pydantic_model,  # The Pydantic class
            )

            # Extract the parsed object (a Pydantic instance)
            parsed_obj = completion.choices[0].message.parsed
            if parsed_obj is None:
                logger.error("Structured parse returned no parsed object.")
                return LLMCompletionNonStreamingResponse(content="")
            
            # logger.info("Parsed object:", parsed_obj)

            # Return a normal LLMCompletionNonStreamingResponse with
            # empty .content (unless you want to do something else),
            # and the parsed object in .parsed
            return LLMCompletionNonStreamingResponse(
                content="", 
                functionCalls=None,
                parsed=parsed_obj
            )

        except ValidationError as ve:
            logger.error(f"Pydantic validation error: {ve}")
            return LLMCompletionNonStreamingResponse(content="")
        except Exception as e:
            logger.error(f"Failed to generate structured parse: {e}")
            return LLMCompletionNonStreamingResponse(content="")

    # If not using pydantic structured parse, follow the old logic
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
        # Non-streaming old code path
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
    if not response.content and response.functionCalls is None and response.parsed is None:
        logger.error(f"Received empty content; retrying ..., response so far: {response}, request: {request}")
        return await llmCompletionWithRetry(request, retries - 1)
    return response


# ----------------------------------------------------------------------
# Helper: async generator yielding nothing
# ----------------------------------------------------------------------
async def async_empty_generator() -> AsyncIterator[str]:
    if False:  # This will never run
        yield ""  # no-op
