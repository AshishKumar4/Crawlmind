# crawlmind/tests/test_llm_completion.py

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from crawlmind.llm_completion import (
    llmCompletion,
    llmCompletionStreaming,
    llmCompletionWithRetry,
    LLMCompletionRequest,
    LLMTool,
    LLMFunction,
    LLMCompletionNonStreamingResponse,
    LLMCompletionStreamingResponse
)


@pytest.mark.asyncio
async def test_llmCompletion_non_streaming_success():
    """
    Test a successful non-streaming completion with content returned.
    """
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_response_content = "Hello from non-streaming GPT response!"

        # Prepare the .create(...) call to return a simulated success
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=mock_response_content,
                        tool_calls=[]  # no function calls
                    )
                )
            ]
        )

        request = LLMCompletionRequest(
            model="gpt-4o",
            system_prompt="system prompt",
            messages=[{"role": "user", "content": "Hello?"}],
        )
        response = await llmCompletion(request)
        assert isinstance(response, LLMCompletionNonStreamingResponse)
        assert response.content == mock_response_content
        assert response.functionCalls is None
        # Check that the underlying OpenAI client was invoked
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_llmCompletion_non_streaming_empty():
    """
    Test a case where the completion returns an empty string.
    """
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="",
                        tool_calls=[]
                    )
                )
            ]
        )

        request = LLMCompletionRequest(
            model="gpt-4o",
            system_prompt="system prompt",
            messages=[{"role": "user", "content": "Hello?"}],
        )
        response = await llmCompletion(request)
        assert response.content == ""
        assert response.functionCalls is None


@pytest.mark.asyncio
async def test_llmCompletion_non_streaming_with_function_calls():
    """
    Test capturing function calls from the model's response.
    """
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value

        # Create a mock function with name and arguments as strings
        function_mock = MagicMock()
        function_mock.name = "get_current_weather"
        function_mock.arguments = '{"location":"Boston"}'

        # Create a mock tool call containing the function_mock
        tool_call_mock = MagicMock(function=function_mock)

        # Set the return value of the create() method
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="The weather is sunny.",
                        tool_calls=[tool_call_mock]
                    )
                )
            ]
        )

        request = LLMCompletionRequest(
            model="gpt-4o",
            system_prompt="system prompt",
            messages=[{"role": "user", "content": "How's the weather?"}],
        )
        response = await llmCompletion(request)
        assert response.content == "The weather is sunny."
        assert response.functionCalls is not None
        assert response.functionCalls[0]["name"] == "get_current_weather"
        assert response.functionCalls[0]["args"] == {"location": "Boston"}


@pytest.mark.asyncio
async def test_llmCompletion_streaming_success():
    """
    Test a successful streaming completion where multiple chunks are returned.
    """
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value

        # Define an async generator to simulate streaming responses
        async def mock_stream_generator():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk1 "))])
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk2 "))])
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk3"))])

        # Create an AsyncMock for the streaming response
        mock_stream = AsyncMock()
        mock_stream.__aiter__.side_effect = mock_stream_generator

        # Assign the async generator to the create method
        mock_client.chat.completions.create.return_value = mock_stream

        request = LLMCompletionRequest(
            model="gpt-4o",
            system_prompt="system prompt",
            messages=[{"role": "user", "content": "Hello streaming?"}],
        )
        response = await llmCompletionStreaming(request)
        assert isinstance(response, LLMCompletionStreamingResponse)

        content_collected = ""
        async for chunk in response.content:
            content_collected += chunk
        print("content collected", content_collected)
        assert content_collected == "chunk1 chunk2 chunk3"


@pytest.mark.asyncio
async def test_llmCompletionWithRetry_success_on_second_attempt():
    """
    Test llmCompletionWithRetry where the first attempt returns empty, second attempt succeeds.
    """
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value

        # Simulate the first call returns empty, second returns data
        first_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        second_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Retry success!"))]
        )

        mock_client.chat.completions.create.side_effect = [first_response, second_response]

        request = LLMCompletionRequest(
            model="gpt-4o",
            system_prompt="system prompt",
            messages=[{"role": "user", "content": "Hello?"}],
        )
        resp = await llmCompletionWithRetry(request, retries=2)
        assert resp.content == "Retry success!"


@pytest.mark.asyncio
async def test_llmCompletionWithRetry_exhausted():
    """
    Test that llmCompletionWithRetry raises an exception if it exhausts retries.
    """
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value

        # Return empty content on every attempt
        always_empty = MagicMock(choices=[MagicMock(message=MagicMock(content=""))])
        mock_client.chat.completions.create.side_effect = [always_empty, always_empty, always_empty]

        request = LLMCompletionRequest(
            model="gpt-4o",
            system_prompt="system prompt",
            messages=[{"role": "user", "content": "Hello?"}],
        )
        with pytest.raises(RuntimeError) as excinfo:
            await llmCompletionWithRetry(request, retries=1)

        assert "Failed to get LLM response after" in str(excinfo.value)


@pytest.mark.asyncio
async def test_llmCompletion_json():
    """
    Test llmCompletion with JSON output.
    """
    response_json = {"key": "value"}
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value
        # The model returns a JSON string
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=json.dumps(response_json),
                        tool_calls=[]
                    )
                )
            ]
        )

        result = await llmCompletionWithRetry(
            request=LLMCompletionRequest(
                model="gpt-4o",
                system_prompt="system prompt",
                messages=[{"role": "user", "content": "Gimme JSON."}],
                response_format={"type": "json_object"},
            ),
            retries=1,
        )
        
        obj = json.loads(result.content)

        assert isinstance(obj, dict)
        assert obj["key"] == "value"


@pytest.mark.asyncio
async def test_llmCompletion_text():
    """
    Test llmCompletion with plain text output (no JSON parsing).
    """
    mock_text = "Just a plain text response."
    with patch("crawlmind.llm_completion.OpenAI") as mock_openai_class:
        mock_client = mock_openai_class.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=mock_text,
                        tool_calls=[]
                    )
                )
            ]
        )

        result = await llmCompletionWithRetry(
            request=LLMCompletionRequest(
                model="gpt-4o",
                system_prompt="system prompt",
                messages=[{"role": "user", "content": "Gimme text."}],
                response_format={"type": "text"},
            ),
            retries=1,
        )

        assert isinstance(result.content, str)
        assert result.content == mock_text


@pytest.mark.asyncio
async def test_llmCompletion_unsupported_model():
    """
    Test that calling a model that doesn't start with 'gpt-' throws a ValueError.
    """
    request = LLMCompletionRequest(
        model="other-model",
        system_prompt="Prompt",
        messages=[{"role": "user", "content": "Hello?"}],
    )

    with pytest.raises(ValueError) as excinfo:
        await llmCompletion(request)
    assert "Unsupported model" in str(excinfo.value)
