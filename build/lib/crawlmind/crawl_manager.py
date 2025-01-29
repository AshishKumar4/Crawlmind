from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from crawlmind.scraper import Scraper, ScraperOptions, Viewport
from crawlmind.parser import ContentParser, StructuredContent
from crawlmind.crawler.intelligent_crawler import IntelligentCrawler
from crawlmind.types import CrawlResult

logger = logging.getLogger(__name__)


@dataclass
class AggregatedCrawlResult:
    """
    Python version of the TS AggregatedCrawlResult interface.
    Combines raw results, final structured content, and optional ranked keywords.
    """
    raw_results: List[CrawlResult]
    structured_content: StructuredContent


class CrawlManager:
    """
    Python port of TS CrawlManager. Sets up scraper, content parsers, ranker,
    and orchestrates a multi-step crawl with the IntelligentCrawler.
    """

    def __init__(
        self,
        summary_options: Dict[str, Any],
        action_options: Dict[str, Any],
        parser_options: Dict[str, Any],
    ) -> None:
        """
        Args:
            summary_options:    { "model": str, "additional_instructions": str } used for final parsing
            action_options:     { "model": str, "additional_instructions": str } used for the IntelligentCrawler
            parser_options:     { "model": str, "additional_instructions": str } used for the main crawl parser
        """
        # Create the scraper with default config, similar to TS { headless: true, navigationTimeout: 10000, ... }
        scraper_opts = ScraperOptions(
            headless=True,
            navigation_timeout=10000,
            viewport=Viewport(width=1200, height=1200, device_scale_factor=1),
        )
        self.scraper = Scraper(scraper_opts)

        # The parser used on each page's raw content
        self.crawl_parser = ContentParser(parser_options["model"])
        # A second parser used to produce a final combined parse
        self.final_parser = ContentParser(summary_options["model"])

        self.action_options = action_options

    async def crawl(self, url: str, max_depth: int) -> AggregatedCrawlResult:
        """
        Orchestrates an IntelligentCrawler run, merges final content, and ranks keywords.

        Returns:
            AggregatedCrawlResult with:
            - raw_results: list of all per-link crawl results
            - structured_content: final aggregated parse
        """
        # Construct the LLM-driven crawler
        crawler = IntelligentCrawler(
            scraper=self.scraper,
            parser=self.crawl_parser,
            action_model=self.action_options["model"],
            additional_instructions=self.action_options["additional_instructions"],
        )

        # 1) Perform the multi-step crawl
        results: List[CrawlResult] = await crawler.crawl(url, max_depth)

        # 2) Convert raw results to JSON for final parsing
        #    Since each result is (url, structured_content), we replicate TS's JSON.stringify(results)
        raw_json = json.dumps(
            [
                {
                    "url": r.url,
                    "structured_content": r.structured_content.__dict__,
                }
                for r in results
            ],
            ensure_ascii=False,
        )

        # 3) Parse that combined data for a final "aggregated" structured content
        structured_content: StructuredContent = await self.final_parser.parse_final_content(raw_json)

        # 5) Return an aggregated result
        return AggregatedCrawlResult(
            raw_results=results,
            structured_content=structured_content,
        )

    async def close_browser(self) -> None:
        """
        Closes the underlying Scraper's browser session.
        """
        await self.scraper.close_browser()
