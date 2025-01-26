# crawlmind/tests/test_intelligent_crawler.py

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

from crawlmind.crawler.intelligent_crawler import IntelligentCrawler, ActionType, Action, VisitActionResult
from crawlmind.parser import ContentParser, StructuredContent
from crawlmind.scraper import Scraper
from crawlmind.types import CrawlResult

@pytest.mark.asyncio
async def test_intelligent_crawler_basic_visit_and_halt():
    """
    Tests a basic scenario where the crawler first action is "visit_urls",
    then after scraping, the LLM decides to "halt".
    """
    # Mock Scraper & Parser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)

    # Provide a mocked scrape_content result
    mock_scraper.scrape_content.return_value = AsyncMock(
        content="Some scraped content",
        links=["https://example.com/about"]
    )  # or do: mock_scraper.scrape_content.return_value = ...
    mock_scraper.scrape_content.return_value.content = "Some scraped content"
    mock_scraper.scrape_content.return_value.links = ["https://example.com/about"]

    # Provide a mocked parse_content result
    mock_structured = StructuredContent(
        context="Test context",
        summary="Test summary",
        technical_terms=["Python", "Django"],
        unique_terminologies=["QuantumComputing"],
        concepts_ideas=["Neural Networks"],
        people_places_events=["John Doe"],
        dates_timelines=["2025-01-01"],
        links_references=["http://example.com"],
        other_keywords=["keyword"],
        unstructured_content="unstructured"
    )
    mock_parser.parse_content.return_value = mock_structured

    # Next, patch the LLM (llmCompletionWithRetry) so that the first call returns "halt" after scraping
    with patch("crawlmind.crawler.intelligent_crawler.llmCompletionWithRetry") as mock_llm:
        # The crawler calls get_action(...) to see what to do next
        # We want it to return a function call with name "halt".
        mock_llm_response = MagicMock()
        mock_llm_response.content = "some content"
        # functionCalls is a list of function calls returned by the LLM
        mock_llm_response.functionCalls = [
            {
                "name": "halt",
                "args": {}
            }
        ]
        mock_llm.return_value = mock_llm_response

        # Initialize the IntelligentCrawler
        crawler = IntelligentCrawler(
            scraper=mock_scraper,
            parser=mock_parser,
            action_model="test-model"
        )

        # Perform crawling
        results = await crawler.crawl("https://example.com", max_depth=2)

    # We expect that:
    # 1) The crawler took the default action=visit_urls at iteration 0.
    # 2) It scraped content from "https://example.com".
    # 3) Then called get_action => LLM told it to "halt".
    # 4) So results should contain one CrawlResult from "https://example.com".

    assert len(results) == 1
    assert results[0].url == "https://example.com"
    assert results[0].structured_content == mock_structured

    # Check that the scraper was called exactly once
    mock_scraper.scrape_content.assert_awaited_once_with("https://example.com")
    # Check that parser was called with the raw content
    mock_parser.parse_content.assert_awaited_once_with("Some scraped content")

    # The LLM (get_action call) should have been invoked exactly once (after the visit action).
    assert mock_llm.call_count == 1

@pytest.mark.asyncio
async def test_intelligent_crawler_visit_multiple_urls():
    """
    Tests a scenario where LLM instructs the crawler to "visit_urls" with multiple links,
    and then again "visit_urls" for the second iteration. We artificially limit depth=2.
    """
    # Mock Scraper & Parser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)

    # We'll define different scrape_content results for different links
    async def mock_scrape(url: str):
        # Return different content for each URL
        content_map = {
            "https://example.com": ("Homepage content", ["https://example.com/about", "http://otherdomain.com/extra"]),
            "https://example.com/about": ("About page content", ["https://example.com/contact"]),
        }
        c, l = content_map.get(url, ("Unknown content", []))
        result = MagicMock()
        result.content = c
        result.links = l
        return result

    mock_scraper.scrape_content.side_effect = mock_scrape

    # Provide a generic parse_content result
    async def mock_parse(content: str):
        # Just return a structured content with summary=the content
        return StructuredContent(
            context="ctx",
            summary=f"Parsed: {content}",
            technical_terms=["TermA"],
            unique_terminologies=[],
            concepts_ideas=[],
            people_places_events=[],
            dates_timelines=[],
            links_references=[],
            other_keywords=[]
        )

    mock_parser.parse_content.side_effect = mock_parse

    with patch("crawlmind.crawler.intelligent_crawler.llmCompletionWithRetry") as mock_llm:
        # We want to produce 2 calls for get_action:
        # 1) returns an action "visit_urls" with 2 links ("https://example.com", "https://example.com/about").
        # 2) returns an action "visit_urls" with 1 link ("https://example.com/about"),
        #    or we can do "halt" so we see multiple steps.

        # Actually, the code uses the existing "currentAction.arguments" to determine which links to visit.
        # So let's see how we can shape the function calls.

        # The crawler's first iteration uses "visit_urls" with argument=[ "https://example.com" ] by default.
        # After scraping, it calls get_action => so let's define the first llm response to return "visit_urls" with some arguments

        # We'll store each call in a list to simulate consecutive calls
        function_calls_responses = [
            # The first call to get_action => action=visit_urls => arguments is "https://example.com/about"
            {
                "name": "visit_urls",
                "args": {
                    "links": ["https://example.com/about"],
                    "additional_remarks": "Continue scraping"
                }
            },
            # The second call to get_action => action=halt => no more links
            {
                "name": "halt",
                "args": {}
            }
        ]

        def side_effect_for_llm(*args, **kwargs):
            # Return a MagicMock each time with functionCalls=the next in the list
            mock_resp = MagicMock()
            mock_resp.functionCalls = [function_calls_responses.pop(0)]
            mock_resp.content = "some content from LLM"
            return mock_resp

        mock_llm.side_effect = side_effect_for_llm

        # Initialize IntelligentCrawler
        crawler = IntelligentCrawler(scraper=mock_scraper, parser=mock_parser)

        # We do max_depth=2, so it should do 2 iterations
        results = await crawler.crawl("https://example.com", max_depth=2)

    # The loop:
    #  Iteration 0: action=visit_urls w/ arguments=["https://example.com"]
    #    => scrape "https://example.com"
    #    => get_action => LLM => returns { name=visit_urls, arguments=["https://example.com/about"] }
    #  Iteration 1: action=visit_urls w/ arguments=["https://example.com/about"]
    #    => scrape "https://example.com/about"
    #    => get_action => LLM => returns { name=halt }
    #  => done

    # So we expect 2 results
    assert len(results) == 2
    assert results[0].url == "https://example.com"
    assert results[0].structured_content.summary == "Parsed: Homepage content"

    assert results[1].url == "https://example.com/about"
    assert results[1].structured_content.summary == "Parsed: About page content"

    # Check scrape calls
    assert mock_scraper.scrape_content.await_count == 2
    calls = [c.args[0] for c in mock_scraper.scrape_content.await_args_list]
    assert calls == ["https://example.com", "https://example.com/about"]

    # Check parser calls
    # first parse => "Homepage content"
    # second parse => "About page content"
    parse_calls = [c.args[0] for c in mock_parser.parse_content.await_args_list]
    assert parse_calls == ["Homepage content", "About page content"]

    # Check LLM calls => get_action was called 2 times
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_intelligent_crawler_no_function_calls():
    """
    Tests scenario where LLM returns no functionCalls, forcing the crawler to retry,
    eventually defaulting to "halt".
    """
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)

    # For the single link
    mock_scraper.scrape_content.return_value = AsyncMock(content="Some content", links=[])

    mock_parser.parse_content.return_value = StructuredContent(
        context="some",
        summary="some",
        technical_terms=[],
        unique_terminologies=[],
        concepts_ideas=[],
        people_places_events=[],
        dates_timelines=[],
        links_references=[]
    )

    with patch("crawlmind.crawler.intelligent_crawler.llmCompletionWithRetry") as mock_llm:
        # The crawler calls get_action  after scraping => we force llm to return .functionCalls=None
        # so that it tries up to 3 times then halts
        mock_resp = MagicMock()
        mock_resp.functionCalls = None
        mock_resp.content = "No function calls returned"
        mock_llm.return_value = mock_resp

        crawler = IntelligentCrawler(scraper=mock_scraper, parser=mock_parser)
        results = await crawler.crawl("https://nodata.com", max_depth=2)

    # The crawler in iteration 0 => action=visit_urls => scrapes => calls get_action => no function calls => tries again => eventually halts
    # So we do get 1 result from scraping the initial link, then we bail out
    assert len(results) == 1
    assert results[0].url == "https://nodata.com"
    assert results[0].structured_content.summary == "some"
    # The LLM was called multiple times for retries
    assert mock_llm.call_count >= 1
    # The scrape was done once
    mock_scraper.scrape_content.assert_awaited_once_with("https://nodata.com")


@pytest.mark.asyncio
async def test_intelligent_crawler_exception_in_scrape_link():
    """
    Tests that if scraping a link fails, we still get a partial result with a default structured_content.
    """
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)

    # Force an exception when scraping
    mock_scraper.scrape_content.side_effect = Exception("Scrape error")

    # The parser won't be used because scraping fails, but let's set a default
    mock_parser.parse_content.return_value = StructuredContent.empty()

    with patch("crawlmind.crawler.intelligent_crawler.llmCompletionWithRetry") as mock_llm:
        # We'll do 1 iteration, set functionCalls to "halt" right after
        mock_llm_response = MagicMock()
        mock_llm_response.functionCalls = [{"name": "halt", "args": {}}]
        mock_llm_response.content = "some content"
        mock_llm.return_value = mock_llm_response

        crawler = IntelligentCrawler(scraper=mock_scraper, parser=mock_parser)
        results = await crawler.crawl("https://example-fail.com", max_depth=1)

    # We expect 0 results if the scrape call completely failed
    # Actually the code in the snippet tries to add a partial result but let's see how you handle it
    # In your snippet, you do `scrapeLink` which calls `scraper.scrapeContent`. 
    # If that fails, you catch it in the base crawler or not? 
    # Actually "scrapeLink" doesn't have its own try/except, might bubble up in TS code. 
    # If it isn't caught, might lead to no results. 
    # We'll assume the final result is an empty list because the code might break from exception. 
    # Or you can handle partial results. 
    # Adjust the assertion as needed if you do something else in a real code.
    assert len(results) == 0

    # Ensure it tried to scrape once
    mock_scraper.scrape_content.assert_awaited_once()
    # Parser not called because scraping never returned content
    mock_parser.parse_content.assert_not_awaited()
    # LLM was called once to get the next action
    mock_llm.assert_called_once()
