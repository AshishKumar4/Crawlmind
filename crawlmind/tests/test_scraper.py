# crawlmind/tests/test_scraper.py

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bs4 import BeautifulSoup, Tag  # Ensure Tag is imported

from crawlmind.scraper import Scraper, ScrapeResponse, ScraperOptions, Viewport

import re

def normalize_whitespace(s: str) -> str:
    """Normalize whitespace by replacing multiple spaces/newlines with a single space."""
    return re.sub(r'\s+', ' ', s).strip()
@pytest.mark.asyncio
async def test_scraper_scrape_content_success():
    """
    Test successful scraping of a webpage.
    """
    mock_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <main>
                <h1>Welcome to the Test Page</h1>
                <p>This is a sample paragraph.</p>
                <a href="https://www.example.com/about">About</a>
                <a href="https://www.example.com/contact">Contact</a>
            </main>
            <script>alert("Hello");</script>
            <style>body {font-family: Arial;}</style>
        </body>
    </html>
    """
    mock_url = "https://www.example.com"

    with patch('crawlmind.scraper.async_playwright') as mock_playwright:
        # Create AsyncMock instances for playwright components
        mock_playwright_instance = AsyncMock()
        mock_browser = AsyncMock()
        mock_page = MagicMock()

        # Configure the mock_playwright to return a mock_playwright_instance when start() is called
        mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)

        # Configure the mock_playwright_instance.chromium.launch() to return mock_browser
        mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)

        # Configure mock_browser.new_page() to return mock_page
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        # Mock page.goto to do nothing (i.e., succeed)
        mock_page.goto = AsyncMock()

        # Mock page.title and page.content
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.content = AsyncMock(return_value=mock_html)

        # Set page.url to return mock_url
        mock_page.url = mock_url

        # Mock page.on to prevent coroutine warnings
        mock_page.on = MagicMock()

        # Instantiate Scraper and perform scraping
        scraper = Scraper()
        response = await scraper.scrape_content(mock_url)
        print("Response:", response.content)
        # Assertions
        assert isinstance(response, ScrapeResponse)
        
        # Normalize whitespace for comparison
        actual_content = normalize_whitespace(response.content)
        
        print("Actual Content:", actual_content)
        # Assertions
        assert "## START OF PAGE CONTENT ##" in actual_content
        assert "Title: Test Page" in actual_content
        assert "<h1> Welcome to the Test Page</h1>" in actual_content
        assert "This is a sample paragraph." in actual_content
        assert '<a href="https://www.example.com/about">About</a>' in actual_content
        assert '<a href="https://www.example.com/contact">Contact</a>' in actual_content
        assert "## END OF PAGE CONTENT ##" in actual_content
        assert "https://www.example.com/about" in response.links
        assert "https://www.example.com/contact" in response.links

@pytest.mark.asyncio
async def test_scraper_scrape_content_error():
    """
    Test scraping when an error occurs during page navigation.
    """
    mock_url = "https://www.invalidurl.com"

    with patch('crawlmind.scraper.async_playwright') as mock_playwright:
        # Create AsyncMock instances for playwright components
        mock_playwright_instance = AsyncMock()
        mock_browser = AsyncMock()
        mock_page = MagicMock()

        # Configure the mock_playwright to return a mock_playwright_instance when start() is called
        mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)

        # Configure the mock_playwright_instance.chromium.launch() to return mock_browser
        mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)

        # Configure mock_browser.new_page() to return mock_page
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        # Mock page.goto to raise an exception
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation failed"))

        # Mock page.title and page.content (not used in this test)
        mock_page.title = AsyncMock(return_value="Invalid Page")
        mock_page.content = AsyncMock(return_value="")

        # Set page.url to return mock_url
        mock_page.url = mock_url

        # Mock page.on to prevent coroutine warnings
        mock_page.on = MagicMock()

        # Instantiate Scraper and perform scraping
        scraper = Scraper()
        response = await scraper.scrape_content(mock_url)

        # Assertions
        assert isinstance(response, ScrapeResponse)
        assert response.content == ""
        assert response.links == []

@pytest.mark.asyncio
async def test_scraper_handle_response_download():
    """
    Test that handle_response logs a download detection.
    """
    with patch('crawlmind.scraper.logger') as mock_logger:
        from crawlmind.scraper import Scraper

        scraper = Scraper()

        # Mock Response object
        mock_response = AsyncMock()
        mock_response.headers = {
            'content-disposition': 'attachment; filename="file.zip"',
            'content-length': '2048000',
            'content-type': 'application/octet-stream'
        }

        await scraper.handle_response(mock_response)

        # Assertions
        mock_logger.info.assert_called_with("DOWNLOAD: A file download has been detected")

@pytest.mark.asyncio
async def test_scraper_good_html():
    """
    Test the good_html method to ensure scripts and styles are removed and important elements are moved.
    """
    from crawlmind.scraper import Scraper

    html = """
    <html>
        <head>
            <title>Sample Page</title>
            <script src="app.js"></script>
            <style>body { background-color: #fff; }</style>
        </head>
        <body>
            <main>
                <h1>Main Content</h1>
                <p>Welcome to the main content.</p>
            </main>
            <div id="bodyContent">
                <p>Additional content here.</p>
            </div>
        </body>
    </html>
    """

    scraper = Scraper()
    soup = scraper.good_html(html)

    # Check that script and style tags are removed
    assert soup.find('script') is None
    assert soup.find('style') is None

    # Check that <div id="bodyContent"> is first, followed by <main>
    body_children = list(soup.body.children)
    # Remove any NavigableString (like '\n') for accurate checking
    body_children = [child for child in body_children if isinstance(child, Tag)]
    assert len(body_children) >= 2
    assert body_children[0].name == 'div'
    assert body_children[0].get('id') == 'bodyContent'
    assert body_children[1].name == 'main'

@pytest.mark.asyncio
async def test_scraper_ugly_chowder():
    """
    Test the ugly_chowder method for proper content extraction.
    """
    from crawlmind.scraper import Scraper

    html = """
    <body>
        <main>
            <h1>Header</h1>
            <p>This is a <strong>test</strong> paragraph.</p>
            <form action="/submit">
                <input type="text" placeholder="Enter name" />
            </form>
        </main>
        <div>
            <p>Another div content.</p>
        </div>
        <a href="https://www.example.com/page1">Page 1</a>
    </body>
    """

    expected_output = (
        "<h1> Header</h1> This is a test paragraph. <form> </form> Another div content. <a href=\"https://www.example.com/page1\">Page 1</a>"
    )

    scraper = Scraper()
    output = scraper.ugly_chowder(html)
    normalized_output = normalize_whitespace(output)
    normalized_expected = normalize_whitespace(expected_output)
    print("Output:", normalized_output, "Expected:", normalized_expected)
    assert normalized_output == normalized_expected
    
# @pytest.mark.asyncio
# async def test_scraper_make_tag():
#     """
#     Test the make_tag method for correct tag construction and attribute truncation.
#     """
#     from crawlmind.scraper import Scraper
#     from bs4 import BeautifulSoup

#     scraper = Scraper()
#     html = """
#     <div href="https://www.example.com/this-is-a-very-long-url-that-needs-to-be-truncated-because-it-is-too-long"
#          placeholder="This is a very long placeholder text that should be truncated after a certain length"
#          title="This title is also very long and should be truncated appropriately"
#          role="navigation"
#          value="some value"
#          pgpt-id="unique-id-12345">
#         Some sample text content that exceeds the character limit to ensure truncation works as expected.
#     </div>
#     """
#     soup = BeautifulSoup(html, 'html.parser')
#     div = soup.find('div')

#     tag_info = scraper.make_tag(div)

#     expected_tag = '<div href="https://www.example.com/this-is-a-very-long-url-[..]" ' \
#                   'placeholder="This is a very long placeholder te[..]" ' \
#                   'title="This title is also very long a[..]" ' \
#                   'role="navigation" ' \
#                   'value="some value" pgpt-id="unique-id-12345">'

#     # The text is under 200 characters, so no truncation expected
#     expected_tag += 'Some sample text content that exceeds the character limit to ensure truncation works as expected.</div>'
#     print("Tag Info:", tag_info, "Expected Tag:", expected_tag)
#     assert tag_info['tag'] == expected_tag
#     assert tag_info['text'] == 'Some sample text content that exceeds the character limit to ensure truncation works as expected.'
