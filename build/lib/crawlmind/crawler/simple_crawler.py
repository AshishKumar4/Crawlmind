# crawlmind/simple_crawler.py

from typing import List
from crawlmind.crawler.crawler import Crawler
from crawlmind.types import CrawlResult
import logging
from collections import deque

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed

class SimpleCrawler(Crawler):
    async def crawl(self, url: str, max_depth: int) -> List[CrawlResult]:
        """
        Performs a breadth-first crawl starting from the given URL up to the specified depth.
        
        Args:
            url (str): The starting URL for crawling.
            max_depth (int): The maximum depth to crawl.
        
        Returns:
            List[CrawlResult]: A list of crawl results.
        """
        crawl_results: List[CrawlResult] = []
        queue = deque()
        queue.append((url, 0))
        self.visited_urls.add(url)
        
        while queue:
            current_url, depth = queue.popleft()
            logger.info(f"Crawling URL: {current_url} at depth: {depth}")
            
            if "://" in current_url:
                # Ensure the URL is valid
                if not self._is_valid_url(current_url):
                    logger.debug(f"Invalid URL: {current_url}. Skipping.")
                    continue
            
            if depth > max_depth:
                logger.debug(f"Max depth {max_depth} reached for URL: {current_url}. Skipping.")
                continue
            
            # Scrape the content
            try:
                scrape_response = await self.scraper.scrape_content(current_url)
                logger.debug(f"Scraped content from {current_url}.")
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {e}")
                continue  # Skip to the next URL
            
            # Parse the content
            try:
                parsed_content = await self.parser.parse_content(scrape_response.content)
                logger.debug(f"Parsed content from {current_url}.")
            except Exception as e:
                logger.error(f"Error parsing content from {current_url}: {e}")
                continue  # Skip to the next URL
            
            # Create CrawlResult
            crawl_result = CrawlResult(
                url=current_url,
                structured_content=parsed_content,  # Assuming 'summary' contains relevant content
                links=parsed_content.links_references  # Use 'links_references' instead of 'links'
            )
            crawl_results.append(crawl_result)
            logger.info(f"Added CrawlResult for {current_url}.")
            
            # Filter and enqueue new links
            new_links = self._filter_links(scrape_response.links, current_url)
            for link in new_links:
                if link not in self.visited_urls:
                    queue.append((link, depth + 1))
                    self.visited_urls.add(link)
                    logger.debug(f"Enqueued new link: {link} at depth: {depth + 1}")
        
        logger.info(f"Crawling completed. Total pages crawled: {len(crawl_results)}.")
        return crawl_results
