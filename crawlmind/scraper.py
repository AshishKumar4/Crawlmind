# crawlmind/scraper.py

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, Page, Browser, Response
from bs4 import BeautifulSoup, Tag, NavigableString

from crawlmind.logger import logger  # Ensure logger is properly set up in logger.py

@dataclass
class ScrapeResponse:
    content: str
    links: List[str]

@dataclass
class Viewport:
    width: int
    height: int
    device_scale_factor: Optional[int] = 1

@dataclass
class ScraperOptions:
    headless: bool = True
    navigation_timeout: int = 10000  # in milliseconds
    viewport: Viewport = field(default_factory=lambda: Viewport(width=1200, height=800, device_scale_factor=1))

class Scraper:
    def __init__(self, options: Optional[ScraperOptions] = None) -> None:
        self.options = options or ScraperOptions()
        self.browser: Optional[Browser] = None
        self.playwright = None

    async def start_browser(self) -> Browser:
        if self.browser:
            return self.browser

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.options.headless,
            # executable_path='/usr/bin/google-chrome-stable'  # Adjust path as necessary
            # args=['--no-sandbox', '--disable-setuid-sandbox'],  # Uncomment if needed
        )
        logger.info('Browser started')
        return self.browser

    async def new_page(self) -> Page:
        await self.start_browser()
        page = await self.browser.new_page(
            viewport={
                "width": self.options.viewport.width,
                "height": self.options.viewport.height,
                "device_scale_factor": self.options.viewport.device_scale_factor,
            }
        )

        page.on('framenavigated', lambda frame: asyncio.create_task(self.handle_frame_navigated(page, frame)))
        page.on('response', lambda response: asyncio.create_task(self.handle_response(response)))
        return page

    async def handle_frame_navigated(self, page: Page, frame: Any) -> None:
        if frame == page.main_frame:
            await asyncio.sleep(0.5)  # Wait for potential dynamic content to load

    async def handle_response(self, response: Response) -> None:
        headers = response.headers
        content_disposition = headers.get('content-disposition', '')
        content_length = int(headers.get('content-length', '0'))
        content_type = headers.get('content-type', '')

        if ('attachment' in content_disposition.lower() or
            content_length > 1024 * 1024 or
            content_type == "application/octet-stream"):
            await asyncio.sleep(2)  # Non-blocking wait
            logger.info("DOWNLOAD: A file download has been detected")

    def good_html(self, html: str) -> BeautifulSoup:
        html = re.sub(r'</', r' </', html)  # Ensure space before closing tags
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Define important selectors
        important_selectors = ['main', '[role="main"]', '#bodyContent', '#search', '#searchform', '.kp-header']

        for selector in important_selectors:
            for element in soup.select(selector):
                soup.body.insert(0, element.extract())

        return soup

    def is_tag(self, element: Any) -> bool:
        return isinstance(element, Tag)

    def is_text(self, element: Any) -> bool:
        return isinstance(element, NavigableString)

    def ugly_chowder(self, html: str) -> str:
        soup = self.good_html('<body>' + html + '</body>')

        def traverse(element: Any) -> str:
            output = ""
            if self.is_tag(element):
                tag_name = element.name
                children = element.contents

                if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    output += f"<{tag_name}>"

                if tag_name == "form":
                    output += f"\n<{tag_name}>\n"

                if tag_name in ["div", "section", "main"]:
                    output += "\n"

                the_tag = self.make_tag(element)

                if element.has_attr("pgpt-id") or element.has_attr("href") or element.has_attr("title"):
                    output += f" {the_tag['tag']}"

                for child in children:
                    output += traverse(child)

                if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    output += f"</{tag_name}>"

                if tag_name == "form":
                    output += f"\n</{tag_name}>\n"

                if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6", "div", "section", "main"]:
                    output += "\n"

            elif self.is_text(element) and not element.parent.get("pgpt-id") and not element.parent.get("href"):
                text = element.strip()
                if text:
                    output += f" {text}"

            # Clean up whitespace
            output = re.sub(r'[^\S\n]+', ' ', output)
            output = re.sub(r' \n+', '\n', output)
            output = re.sub(r'[\n]+', '\n', output)
            return output

        return traverse(soup.body)

    def make_tag(self, element: Tag) -> Dict[str, Any]:
        text_content = ' '.join(element.get_text(separator=' ').split())
        placeholder = element.get("placeholder", "")
        tag_name = element.name
        title = element.get("title", "")
        value = element.get("value", "")
        role = element.get("role", "")
        type_attr = element.get("type", "")
        href = element.get("href", "")
        pgpt_id = element.get("pgpt-id", "")

        # Truncate attributes if necessary
        if href and len(href) > 32:
            href = href[:32] + "[..]"
        # if placeholder and len(placeholder) > 32:
        #     placeholder = placeholder[:32] + "[..]"
        # if title and len(title) > 32:
        #     title = title[:32] + "[..]"
        # if text_content and len(text_content) > 200:
        #     text_content = text_content[:200] + "[..]"

        tag = f"<{tag_name}"
        if href:
            tag += f' href="{href}"'
        if type_attr:
            tag += f' type="{type_attr}"'
        if placeholder:
            tag += f' placeholder="{placeholder}"'
        if title:
            tag += f' title="{title}"'
        if role:
            tag += f' role="{role}"'
        if value:
            tag += f' value="{value}"'
        if pgpt_id:
            tag += f' pgpt-id="{pgpt_id}"'
        tag += '>'

        obj = {"tag": tag}

        if text_content:
            obj["text"] = text_content
            obj["tag"] += f"{text_content}</{tag_name}>"

        return obj

    async def get_page_content(self, page: Page) -> ScrapeResponse:
        title = await page.title()
        html = await page.content()
        content = f"## START OF PAGE CONTENT ##\nTitle: {title}\n\n{self.ugly_chowder(html)}\n## END OF PAGE CONTENT ##"

        soup = BeautifulSoup(html, 'html.parser')
        links = []
        base_url = page.url

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            if href:
                try:
                    absolute_url = urljoin(base_url, href)
                    parsed_url = urlparse(absolute_url)
                    if parsed_url.scheme in ['http', 'https']:
                        links.append(absolute_url)
                except Exception as e:
                    logger.error(f"Error parsing URL: {href} - {e}")
                    links.append(href)

        unique_links = list(set(links))
        return ScrapeResponse(content=content, links=unique_links)

    async def close_browser(self) -> None:
        if self.browser:
            await self.browser.close()
            self.browser = None
            await self.playwright.stop()
            logger.info('Browser closed')

    async def scrape_content(self, url: str) -> ScrapeResponse:
        page = await self.new_page()
        logger.info(f"Scraping {url}")
        try:
            await page.goto(url, wait_until='load', timeout=self.options.navigation_timeout)
            response = await self.get_page_content(page)
            return response
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ScrapeResponse(content='', links=[])

    async def __aenter__(self) -> Scraper:
        await self.start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close_browser()