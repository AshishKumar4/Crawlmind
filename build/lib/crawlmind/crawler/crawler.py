# crawlmind/crawler.py

from abc import ABC, abstractmethod
from urllib.parse import urlparse, urljoin
from typing import List, Set
from crawlmind.scraper import Scraper
from crawlmind.parser import ContentParser
from crawlmind.types import CrawlResult
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

class Crawler(ABC):
    def __init__(self, scraper: Scraper, parser: ContentParser):
        self.scraper = scraper
        self.parser = parser
        self.visited_urls: Set[str] = set()
        logger.info("Crawler initialized with scraper and parser.")

    def _get_base_domain(self, url: str) -> str:
        """
        Extracts the base domain from a given URL.
        
        Args:
            url (str): The URL to extract the domain from.
        
        Returns:
            str: The hostname of the URL.
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.hostname or ""
            # logger.debug(f"Extracted base domain '{domain}' from URL '{url}'.")
            return domain
        except Exception as e:
            logger.error(f"Error extracting base domain from URL '{url}': {e}")
            return ""

    def _is_valid_url(self, url: str) -> bool:
        """
        Validates whether a URL is properly formatted and uses HTTP or HTTPS scheme.
        
        Args:
            url (str): The URL to validate.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            parsed = urlparse(url)
            is_valid = parsed.scheme in ("http", "https") and bool(parsed.netloc)
            # logger.debug(f"URL '{url}' validation result: {is_valid}.")
            return is_valid
        except Exception as e:
            logger.error(f"Invalid URL '{url}': {e}")
            return False

    def _filter_links(self, raw_links: List[str], base_url: str) -> List[str]:
        """
        Filters raw links based on validity, domain, and visit status.
        
        Args:
            raw_links (List[str]): List of raw links extracted from a page.
            base_url (str): The base URL to resolve relative links and filter by domain.
        
        Returns:
            List[str]: A list of filtered, absolute URLs.
        """
        base_domain = self._get_base_domain(base_url)
        filtered_links = []
        
        for link in raw_links:
            if not self._is_valid_url(link):
                logger.debug(f"Link '{link}' is invalid. Skipping.")
                continue
            absolute_url = urljoin(base_url, link)
            link_domain = self._get_base_domain(absolute_url)
            if link_domain != base_domain:
                logger.debug(f"Link '{absolute_url}' domain '{link_domain}' does not match base domain '{base_domain}'. Skipping.")
                continue
            if absolute_url in self.visited_urls:
                logger.debug(f"Link '{absolute_url}' has already been visited. Skipping.")
                continue
            filtered_links.append(absolute_url)
            logger.debug(f"Link '{absolute_url}' added to filtered links.")
        
        logger.info(f"Filtered {len(filtered_links)} links from {len(raw_links)} raw links. Filtered links: {filtered_links}")
        return filtered_links

    @abstractmethod
    async def crawl(self, url: str, max_depth: int) -> List[CrawlResult]:
        """
        Abstract method to perform crawling starting from a given URL up to a specified depth.
        
        Args:
            url (str): The starting URL for crawling.
            max_depth (int): The maximum depth to crawl.
        
        Returns:
            List[CrawlResult]: A list of crawl results.
        """
        pass
