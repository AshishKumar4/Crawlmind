# crawlmind/tests/test_parser.py

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock
from crawlmind.parser import (
    ContentParser,
    StructuredContent,
    remove_small_words
)


@pytest.mark.asyncio
async def test_remove_small_words():
    words = ["the", "is", "cat", "python", "JS", "C++", "AI"]
    filtered = remove_small_words(words)
    # Expect 'the', 'cat', 'python', because "is", "JS", "AI" are length <= 2
    expected = ["the", "cat", "python", "C++"]
    print("filtered", filtered)
    assert filtered == expected


@pytest.mark.asyncio
@patch("crawlmind.parser.llmCompletionWithRetry")
async def test_parse_content_success(mock_llm):
    # Mock the LLM to return a well-formed JSON string
    mock_response_content = json.dumps({
        "context": "Sample context",
        "summary": "A very detailed summary of the content.",
        "technical_terms": ["Python", "JS", "C++", "API"],
        "unique_terminologies": ["GPT", "ai", "info"],
        "concepts_ideas": ["Neural Networks", "nlp", "CV"],
        "people_places_events": ["John Doe", "Jane Smith"],
        "dates_timelines": ["2023-01-01"],
        "links_references": ["http://example.com"],
        "other_keywords": ["extra"],
        "unstructured_content": "Leftover data"
    })

    # Mock llmCompletionWithRetry() response
    mock_llm.return_value = MagicMock(
        content=mock_response_content,
        functionCalls=None
    )

    parser = ContentParser(model="fake-model")
    raw_data = "some raw web content"
    result: StructuredContent = await parser.parse_content(raw_data)
    
    print("result", result)

    # Check that we got the right parsed content
    assert result.context == "Sample context"
    assert result.summary == "A very detailed summary of the content."
    # "C++" has length 3, so that stays
    assert "JS" not in result.technical_terms
    assert "ai" not in result.unique_terminologies
    assert "nlp" in result.concepts_ideas
    assert "Python" in result.technical_terms
    assert "C++" in result.technical_terms
    assert "Neural Networks" in result.concepts_ideas
    assert result.people_places_events == ["John Doe", "Jane Smith"]
    assert result.dates_timelines == ["2023-01-01"]
    assert result.links_references == ["http://example.com"]
    assert result.other_keywords == ["extra"]
    assert result.unstructured_content == "Leftover data"

    # Ensure mock was called with the correct prompt
    mock_llm.assert_called_once()
    call_args = mock_llm.call_args[0][0]  # The first positional arg of the llmCompletionWithRetry call
    assert raw_data in call_args.messages[0]["content"]


@pytest.mark.asyncio
@patch("crawlmind.parser.llmCompletionWithRetry")
async def test_parse_content_empty_response(mock_llm):
    """
    Test that parse_content handles an empty LLM response gracefully.
    """
    mock_llm.return_value = MagicMock(
        content="",
        functionCalls=None
    )

    parser = ContentParser(model="fake-model")
    result = await parser.parse_content("whatever raw content")

    # Expect an empty StructuredContent
    assert result.context == ""
    assert result.summary == ""
    assert result.technical_terms == []
    # etc...

@pytest.mark.asyncio
@patch("crawlmind.parser.llmCompletionWithRetry")
async def test_parse_content_invalid_json(mock_llm):
    """
    Test that parse_content handles a JSONDecodeError gracefully.
    """
    mock_llm.return_value = MagicMock(
        content="this is not valid json",
        functionCalls=None
    )

    parser = ContentParser(model="fake-model")
    result = await parser.parse_content("whatever raw content")

    # Expect an empty StructuredContent
    assert result.context == ""
    assert result.summary == ""
    assert result.technical_terms == []


@pytest.mark.asyncio
@patch("crawlmind.parser.llmCompletionWithRetry")
async def test_parse_final_content_success(mock_llm):
    """
    Test parse_final_content with a normal response.
    """
    mock_response_content = json.dumps({
        "context": "Combined context from multiple sites",
        "summary": "A very detailed consolidated summary.",
        "technical_terms": ["Python", "REST"],
        "unique_terminologies": ["QuantumComputing", "ai"],
        "concepts_ideas": ["Deep Learning", "nlp"],
        "people_places_events": ["PersonA", "PlaceB"],
        "dates_timelines": ["2024-05-10"],
        "links_references": ["http://example.com"],
        "other_keywords": ["Combined", "extra"],
        "unstructured_content": "Collective leftover info"
    })

    mock_llm.return_value = MagicMock(
        content=mock_response_content,
        functionCalls=None
    )

    parser = ContentParser(model="fake-model")
    input_data = "[ ... multiple structured content entries ...]"
    result = await parser.parse_final_content(input_data)

    # Some checks
    assert result.context == "Combined context from multiple sites"
    assert result.summary == "A very detailed consolidated summary."
    assert "Python" in result.technical_terms
    assert "REST" in result.technical_terms
    assert "ai" not in result.unique_terminologies  # because it's length 2
    assert "Deep Learning" in result.concepts_ideas
    assert "nlp" in result.concepts_ideas
    assert result.people_places_events == ["PersonA", "PlaceB"]
    assert result.dates_timelines == ["2024-05-10"]
    assert result.links_references == ["http://example.com"]
    assert "Combined" in result.other_keywords
    assert result.unstructured_content == "Collective leftover info"
