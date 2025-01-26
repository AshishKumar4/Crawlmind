from abc import ABC, abstractmethod
from crawlmind.scraper import Scraper
from crawlmind.parser import ContentParser

class Crawler(ABC):
    def __init__(self, scraper: Scraper, parser: ContentParser):
        self.scraper = scraper
        self.parser = parser
        self.visitedUrls = set()
        
    def getBaseDomain(self, url: str) -> str:
        return 