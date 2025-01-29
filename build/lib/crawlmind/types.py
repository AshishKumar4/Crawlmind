from dataclasses import dataclass, field
from typing import List
from crawlmind.parser import StructuredContent

@dataclass
class CrawlResult:
    url: str
    structured_content: StructuredContent
    links: List[str] = field(default_factory=list)
