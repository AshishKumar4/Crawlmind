# crawlmind/tests/test_crawler.py

import pytest
from unittest.mock import AsyncMock, patch, call
from crawlmind.crawler.crawler import Crawler
from crawlmind.crawler.simple_crawler import SimpleCrawler
from crawlmind.parser import ContentParser, StructuredContent
from crawlmind.scraper import Scraper, ScrapeResponse
from crawlmind.types import CrawlResult

@pytest.mark.asyncio
async def test_crawler_basic_functionality():
    """
    Test the basic functionality of the SimpleCrawler.
    """
    # Mock Scraper and ContentParser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)
    
    # Define mock responses using ScrapeResponse and StructuredContent
    mock_scraper.scrape_content = AsyncMock(side_effect=[
        ScrapeResponse(
            content='Sample content',
            links=['https://www.example.com/about', '/contact']
        ),
        ScrapeResponse(
            content='About page content',
            links=['https://www.example.com/contact']
        )
    ])
    
    mock_parser.parse_content = AsyncMock(side_effect=[
        StructuredContent(
            context='Sample context from LLM',
            summary='Sample content summary',
            technical_terms=['Python', 'React', 'API'],
            unique_terminologies=['Quantum Entanglement'],
            concepts_ideas=['Neural Networks'],
            people_places_events=['John Doe', 'OpenAI HQ'],
            dates_timelines=['2023-01-01'],
            links_references=['https://www.example.com/about', 'https://www.example.com/contact'],
            other_keywords=['Extra data'],
            unstructured_content='Some leftover content'
        ),
        StructuredContent(
            context='About page context from LLM',
            summary='About page summary',
            technical_terms=['Django', 'REST'],
            unique_terminologies=['OAuth'],
            concepts_ideas=['API Design'],
            people_places_events=['Jane Smith', 'GitHub'],
            dates_timelines=['2024-05-10'],
            links_references=['https://www.example.com/contact'],
            other_keywords=['Developer'],
            unstructured_content='More leftover content'
        )
    ])
    
    # Initialize SimpleCrawler
    crawler = SimpleCrawler(scraper=mock_scraper, parser=mock_parser)
    
    # Perform crawling
    start_url = "https://www.example.com"
    max_depth = 1
    results = await crawler.crawl(start_url, max_depth)
    
    # Assertions
    print('results:', results)
    assert len(results) == 2  # Expecting two CrawlResults
    
    # Verify the first CrawlResult
    result_start = results[0]
    assert result_start.url == start_url
    assert result_start.structured_content.summary == 'Sample content summary'
    assert result_start.links == ['https://www.example.com/about', 'https://www.example.com/contact']
    
    # Verify the second CrawlResult
    result_about = results[1]
    assert result_about.url == 'https://www.example.com/about'
    assert result_about.structured_content.summary== 'About page summary'
    assert result_about.links == ['https://www.example.com/contact']
    
    # Ensure scraper was called twice with correct URLs
    expected_calls = [
        call(start_url),
        call('https://www.example.com/about')
    ]
    mock_scraper.scrape_content.assert_has_awaits(expected_calls, any_order=False)
    assert mock_scraper.scrape_content.await_count == 2
    
    # Ensure parser was called twice with correct content
    expected_parser_calls = [
        call('Sample content'),
        call('About page content')
    ]
    mock_parser.parse_content.assert_has_awaits(expected_parser_calls, any_order=False)
    assert mock_parser.parse_content.await_count == 2

@pytest.mark.asyncio
async def test_crawler_depth_limitation():
    """
    Test that the crawler respects the maximum depth.
    """
    # Mock Scraper and ContentParser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)
    
    # Define mock responses for two levels using ScrapeResponse and StructuredContent
    mock_scraper.scrape_content.side_effect = [
        ScrapeResponse(
            content='Content level 0',
            links=['https://www.example.com/page1']
        ),
        ScrapeResponse(
            content='Content level 1',
            links=['https://www.example.com/page2']
        )
    ]
    
    mock_parser.parse_content.side_effect = [
        StructuredContent(
            context='Context level 0',
            summary='Summary level 0',
            technical_terms=['Term1'],
            unique_terminologies=['UniqueTerm1'],
            concepts_ideas=['Concept1'],
            people_places_events=['Person1'],
            dates_timelines=['2023-01-01'],
            links_references=['https://www.example.com/page1'],
            other_keywords=['Keyword1'],
            unstructured_content=''
        ),
        StructuredContent(
            context='Context level 1',
            summary='Summary level 1',
            technical_terms=['Term2'],
            unique_terminologies=['UniqueTerm2'],
            concepts_ideas=['Concept2'],
            people_places_events=['Person2'],
            dates_timelines=['2024-01-01'],
            links_references=['https://www.example.com/page2'],
            other_keywords=['Keyword2'],
            unstructured_content=''
        )
    ]
    
    # Initialize SimpleCrawler
    crawler = SimpleCrawler(scraper=mock_scraper, parser=mock_parser)
    
    # Perform crawling with max_depth=1
    start_url = "https://www.example.com"
    max_depth = 1
    results = await crawler.crawl(start_url, max_depth)
    
    # Assertions
    assert len(results) == 2
    assert results[0].url == start_url
    assert results[0].structured_content.summary== 'Summary level 0'
    assert results[0].links == ['https://www.example.com/page1']
    assert results[1].url == 'https://www.example.com/page1'
    assert results[1].structured_content.summary== 'Summary level 1'
    assert results[1].links == ['https://www.example.com/page2']
    
    # Ensure scraper and parser were called correctly
    assert mock_scraper.scrape_content.await_count == 2
    assert mock_parser.parse_content.await_count == 2
@pytest.mark.asyncio
async def test_crawler_invalid_links():
    """
    Test that the crawler filters out invalid links.
    """
    # Mock Scraper and ContentParser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)
    
    # Define mock responses with some invalid links using ScrapeResponse and StructuredContent
    mock_scraper.scrape_content = AsyncMock(side_effect=[
        ScrapeResponse(
            content='Sample content',
            links=['invalid_url', 'ftp://www.example.com/resource', 'https://www.example.com/about']
        ),
        ScrapeResponse(
            content='About page content',
            links=['invalid_url', 'https://www.example.com/about']  # Duplicate to test visited_urls
        )
    ])
    
    mock_parser.parse_content = AsyncMock(side_effect=[
        StructuredContent(
            context='Sample context from LLM',
            summary='Sample content summary',
            technical_terms=['Python', 'React', 'API'],
            unique_terminologies=['Quantum Entanglement'],
            concepts_ideas=['Neural Networks'],
            people_places_events=['John Doe', 'OpenAI HQ'],
            dates_timelines=['2023-01-01'],
            links_references=['https://www.example.com/about'],
            other_keywords=['Extra data'],
            unstructured_content='Some leftover content'
        ),
        StructuredContent(
            context='About page context from LLM',
            summary='About page summary',
            technical_terms=['Django', 'REST'],
            unique_terminologies=['OAuth'],
            concepts_ideas=['API Design'],
            people_places_events=['Jane Smith', 'GitHub'],
            dates_timelines=['2024-05-10'],
            links_references=['https://www.example.com/about'],
            other_keywords=['Developer'],
            unstructured_content='More leftover content'
        )
    ])
    
    # Initialize SimpleCrawler
    crawler = SimpleCrawler(scraper=mock_scraper, parser=mock_parser)
    
    # Perform crawling
    start_url = "https://www.example.com"
    max_depth = 1
    results = await crawler.crawl(start_url, max_depth)
    
    # Assertions
    print('results:', results)
    assert len(results) == 2  # Expecting two CrawlResults
    
    # Verify the first CrawlResult
    result_start = results[0]
    assert result_start.url == start_url
    assert result_start.structured_content.summary== 'Sample content summary'
    assert result_start.links == ['https://www.example.com/about']
    
    # Verify the second CrawlResult
    result_about = results[1]
    assert result_about.url == 'https://www.example.com/about'
    assert result_about.structured_content.summary== 'About page summary'
    assert result_about.links == ['https://www.example.com/about']
    
    # Ensure scraper was called twice with correct URLs
    expected_calls = [
        call(start_url),
        call('https://www.example.com/about')
    ]
    mock_scraper.scrape_content.assert_has_awaits(expected_calls, any_order=False)
    assert mock_scraper.scrape_content.await_count == 2
    
    # Ensure parser was called twice with correct content
    expected_parser_calls = [
        call('Sample content'),
        call('About page content')
    ]
    mock_parser.parse_content.assert_has_awaits(expected_parser_calls, any_order=False)
    assert mock_parser.parse_content.await_count == 2

@pytest.mark.asyncio
async def test_crawler_no_links():
    """
    Test that the crawler handles pages with no links gracefully.
    """
    # Mock Scraper and ContentParser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)
    
    # Define mock response with no links using ScrapeResponse and StructuredContent
    mock_scraper.scrape_content = AsyncMock(return_value=ScrapeResponse(
        content='Single page content',
        links=[]
    ))
    
    mock_parser.parse_content = AsyncMock(return_value=StructuredContent(
        context='Single page context',
        summary='Single page summary',
        technical_terms=[],
        unique_terminologies=[],
        concepts_ideas=[],
        people_places_events=[],
        dates_timelines=[],
        links_references=[],
        other_keywords=[],
        unstructured_content=''
    ))
    
    # Initialize SimpleCrawler
    crawler = SimpleCrawler(scraper=mock_scraper, parser=mock_parser)
    
    # Perform crawling
    start_url = "https://www.singlepage.com"
    max_depth = 1
    results = await crawler.crawl(start_url, max_depth)
    
    # Assertions
    assert len(results) == 1
    result = results[0]
    assert result.url == start_url
    assert result.structured_content.summary== 'Single page summary'
    assert result.links == []
    
    # Ensure scraper and parser were called correctly
    mock_scraper.scrape_content.assert_awaited_once_with(start_url)
    mock_parser.parse_content.assert_awaited_once_with('Single page content')

@pytest.mark.asyncio
async def test_crawler_handle_exception_during_scraping():
    """
    Test that the crawler handles exceptions during scraping gracefully.
    """
    # Mock Scraper and ContentParser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)
    
    # Define mock responses where scraping raises an exception
    mock_scraper.scrape_content = AsyncMock(side_effect=Exception("Scraping failed"))
    
    # Initialize SimpleCrawler
    crawler = SimpleCrawler(scraper=mock_scraper, parser=mock_parser)
    
    # Perform crawling
    start_url = "https://www.errorpage.com"
    max_depth = 1
    results = await crawler.crawl(start_url, max_depth)
    
    # Assertions
    assert len(results) == 0  # No successful crawl results
    
    # Ensure scraper was called once and parser was never called
    mock_scraper.scrape_content.assert_awaited_once_with(start_url)
    mock_parser.parse_content.assert_not_awaited()

@pytest.mark.asyncio
async def test_crawler_handle_exception_during_parsing():
    """
    Test that the crawler handles exceptions during parsing gracefully.
    """
    # Mock Scraper and ContentParser
    mock_scraper = AsyncMock(spec=Scraper)
    mock_parser = AsyncMock(spec=ContentParser)
    
    # Define mock responses where parsing raises an exception
    mock_scraper.scrape_content = AsyncMock(return_value=ScrapeResponse(
        content='Some content',
        links=['https://www.example.com/page']
    ))
    mock_parser.parse_content = AsyncMock(side_effect=Exception("Parsing failed"))
    
    # Initialize SimpleCrawler
    crawler = SimpleCrawler(scraper=mock_scraper, parser=mock_parser)
    
    # Perform crawling
    start_url = "https://www.example.com"
    max_depth = 1
    results = await crawler.crawl(start_url, max_depth)
    
    # Assertions
    assert len(results) == 0  # Parsing failed, so no crawl result added
    
    # Ensure scraper and parser were called correctly
    mock_scraper.scrape_content.assert_awaited_once_with(start_url)
    mock_parser.parse_content.assert_awaited_once_with('Some content')
