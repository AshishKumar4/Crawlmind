# crawlmind/crawler/intelligent_crawler.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional
import asyncio
from urllib.parse import urljoin

from crawlmind.crawler.crawler import Crawler  # import your base Crawler class
from crawlmind.scraper import Scraper
from crawlmind.parser import ContentParser, StructuredContent
from crawlmind.llm_completion import llmCompletionWithRetry, LLMCompletionRequest, LLMCompletionNonStreamingResponse, LLMTool, LLMFunction, LLMFunctionParameters
from crawlmind.types import CrawlResult

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Additional data structures
# -------------------------------------------------------------------------

class ActionType(str, Enum):
    VISIT_URLS = "visit_urls"
    HALT = "halt"
    NOTHING = ""  # fallback default

@dataclass
class Action:
    """Represents the next action to be taken by the IntelligentCrawler."""
    id: Optional[str] = None
    name: ActionType = ActionType.NOTHING
    arguments: List[str] = field(default_factory=list)
    additional_remarks: str = ""

@dataclass
class ScrapeResults:
    """Holds the result of scraping and parsing a single link."""
    url: str
    depth: Optional[int] = None
    raw_content: Optional[str] = None
    structured_content: StructuredContent = field(default_factory=StructuredContent.empty)
    raw_links: List[str] = field(default_factory=list)

@dataclass
class VisitActionResult:
    """Holds the result of an action='visit_urls'."""
    action: Action
    results: Optional[List[ScrapeResults]] = None
    additional_remarks: str = ""

ActionResult = VisitActionResult  # In TS, same type

# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------

ACTION_PROMPT = """
You are an agent tasked with guiding the process of scraping and crawling through the domain: BASE_DOMAIN to gather extensive information
and do extensive osint research. Your job is to take an action on the result of the previous action taken by you, and
the context of the current state of the crawling process. The results from the previous action depends on the type of action performed.
You must take the actions by calling the function tools provided to you.

The information provided by the user is as follows:
- Previous action's result: JSON structured object with the previous action's results
- All previously visited URLs: JSON array of all previously visited URLs

You are only allowed to take 1 function call at a time.

Your ultimate goal is to gather as much open source intelligence as possible from the domain BASE_DOMAIN.
If it's a product, gather thorough information about the product, its features, the product domain, documentations etc.
If it's a company or an organization, we would need detailed and exhaustive information about the employees, the company's history, etc.
If it's a personal portfolio, we would need detailed information about the person, their work, their projects, project descriptions in detail,
their skills, experiences, etc.
You may use public web based osint tools to gather information as well, for example google, linkedin etc.

ADDITIONAL_INSTRUCTIONS
""".strip()


class IntelligentCrawler(Crawler):
    """
    Python port of the TypeScript IntelligentCrawler class.
    Extends the Crawler base class to implement "intelligent" action-based crawling using an LLM.
    """
    def __init__(
        self,
        scraper: Scraper,
        parser: ContentParser,
        action_model: str = "gpt-4o",
        additional_instructions: str = ""
    ) -> None:
        super().__init__(scraper, parser)
        self.model: str = action_model
        self.previous_action_results: List[ActionResult] = []
        self.additional_instructions: str = additional_instructions

    async def get_action(
        self,
        base_domain: str,
        current_result: ActionResult,
        retry: int = 0
    ) -> Action:
        """
        Calls the LLM to decide the next action (visit_urls or halt).
        Retries up to 3 times if there's an error or no function calls are returned.
        """
        try:
            # Build the system prompt by injecting domain and any custom instructions
            prompt = ACTION_PROMPT.replace("BASE_DOMAIN", base_domain)
            prompt = prompt.replace("ADDITIONAL_INSTRUCTIONS", self.additional_instructions)

            logger.info(
                f"Getting recommended actions. "
                f"Len of previous action results: {len(str(self.previous_action_results))}, "
                f"Len of current action result: {len(str(current_result))}, "
                f"visited URLs length: {len(str(list(self.visited_urls)))}"
            )

            # Build the request
            request = LLMCompletionRequest(
                model=self.model,
                max_tokens=10000,
                system_prompt=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Previous action's result: {current_result}\n\n"
                            f"All previously visited URLs: {list(self.visited_urls)}"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Older action results: {self.previous_action_results}"
                    }
                ],
                response_format={"type": "json_object"},
                tools=[
                    LLMTool(
                        type="function",
                        function=LLMFunction(
                            name="visit_urls",
                            description=(
                                "Visit a set of URLs in order of priority and get the content and links. "
                                "You may use the 'additional_remarks' field to keep track of the crawling process."
                            ),
                            parameters=LLMFunctionParameters(
                                type="object",
                                properties={
                                    "links": {
                                        "type": "array",
                                        "description": "An array of absolute URLs to visit",
                                        "items": {"type": "string"}
                                    },
                                    "additional_remarks": {
                                        "type": "string"
                                    }
                                },
                                required=["links"]
                            )
                        )
                    ),
                    LLMTool(
                        type="function",
                        function=LLMFunction(
                            name="halt",
                            description=(
                                "Stop the crawling process if enough info has been gathered "
                                "or there are no more links to visit"
                            )
                            # parameters could remain None if you don't need them
                        )
                    )
                ],
            )
            response: LLMCompletionNonStreamingResponse = await llmCompletionWithRetry(request)
            function_calls = response.functionCalls

            if not function_calls:
                # No recognized function call in the LLM response
                logger.error(f"Failed to parse recommended actions. Retrying... {retry}")
                if retry >= 3:
                    # Return a forced 'halt' action if we cannot parse anything after 3 retries
                    return Action(
                        name=ActionType.HALT,
                        arguments=[],
                        additional_remarks=response.content or ""
                    )
                else:
                    return await self.get_action(base_domain, current_result, retry + 1)

            # Typically, we only look at the first function call
            fc = function_calls[0]
            action_name = fc["name"]
            args = fc["args"]  # should be a dictionary from the JSON

            logger.info(f"Recommended Action arguments: {args}")

            # Construct the next action
            next_action = Action(
                name=ActionType(action_name) if action_name in ActionType._value2member_map_ else ActionType.NOTHING,
                arguments=args.get("links", []),
                additional_remarks=args.get("additional_remarks", "")
            )
            logger.info(f"Recommended Actions: {next_action}")
            return next_action

        except Exception as e:
            logger.error(f"Failed to get recommended actions: {e} Retrying... {retry}")
            if retry >= 3:
                return Action(name=ActionType.HALT, arguments=[], additional_remarks="")
            return await self.get_action(base_domain, current_result, retry + 1)

    async def scrape_link(self, link: str, base_url: str) -> ScrapeResults:
        """
        Scrapes a single link, parses it, and returns a ScrapeResults object.
        """
        self.visited_urls.add(link)
        try:
            scraped = await self.scraper.scrape_content(link)
            content, all_links = scraped.content, scraped.links

            if not content:
                logger.error(f"Failed to scrape content for link {link}")
                return ScrapeResults(
                    url=link,
                    structured_content=StructuredContent(
                        context="",
                        summary="",
                        technical_terms=[],
                        unique_terminologies=[],
                        concepts_ideas=[],
                        people_places_events=[],
                        dates_timelines=[],
                        links_references=[],
                        other_keywords=[]
                    ),
                    raw_links=[]
                )

            logger.info(f"Parsing content for link {link}")
            structured_content = await self.parser.parse_content(content)

            # Filter raw links to keep within domain, etc.
            filtered_links = self._filter_links(all_links, base_url)

            return ScrapeResults(
                url=link,
                structured_content=structured_content,
                raw_links=filtered_links
            )
        except Exception as e:
            logger.error(f"Error scraping link {link}: {e}")
            return ScrapeResults(
                url=link,
                structured_content=StructuredContent(
                    context="",
                    summary="",
                    technical_terms=[],
                    unique_terminologies=[],
                    concepts_ideas=[],
                    people_places_events=[],
                    dates_timelines=[],
                    links_references=[],
                    other_keywords=[]
                ),
                raw_links=[]
            )

    async def scrape_links(self, links: List[str], base_url: str) -> List[ScrapeResults]:
        """
        Scrapes multiple links concurrently or in a batch, returning their results.
        """
        tasks = [self.scrape_link(link, base_url) for link in links]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def crawl(self, url: str, max_depth: int) -> List[CrawlResult]:
        """
        Main crawl loop. We do up to `max_depth` actions, each typically
        "visit_urls" or "halt".
        """
        base_url = url
        base_domain = self._get_base_domain(base_url)
        current_action = Action(
            name=ActionType.VISIT_URLS,
            arguments=[url],
            additional_remarks=""
        )

        all_results: List[CrawlResult] = []

        for _ in range(max_depth):
            logger.info(f"New action: {current_action}")

            if current_action.name == ActionType.VISIT_URLS:
                # Prepare the "visitActionResult"
                visit_action_result = VisitActionResult(
                    action=current_action,
                    additional_remarks=current_action.additional_remarks
                )

                if current_action.arguments:
                    # Convert any relative to absolute
                    links_to_visit = [urljoin(base_url, link) for link in current_action.arguments]
                    # Or more robustly:
                    # from urllib.parse import urljoin
                    # links_to_visit = [urljoin(base_url, l) for l in current_action.arguments]

                    logger.info(f"Visiting links: {links_to_visit}")
                    visit_results = await self.scrape_links(links_to_visit, base_url)

                    # Convert each ScrapeResults -> CrawlResult for final usage
                    for sr in visit_results:
                        if not sr or sr.structured_content.summary == "":
                            continue
                        cr = CrawlResult(
                            url=sr.url,
                            structured_content=sr.structured_content
                        )
                        all_results.append(cr)

                    visit_action_result.results = visit_results

                    logger.info(
                        "Action result: "
                        + str([
                            {
                                "url": r.url,
                                "structured_content": r.structured_content,
                                "raw_links": r.raw_links
                            }
                            for r in visit_results
                        ])
                    )

                next_action = await self.get_action(base_domain, visit_action_result)
                current_action = next_action
                self.previous_action_results.append(visit_action_result)

            elif current_action.name == ActionType.HALT:
                # Stop the crawling process
                return all_results

            else:
                # If we receive an unrecognized action, just break or halt
                logger.warning(f"Unrecognized action: {current_action.name}, halting.")
                return all_results

        # If we exit the for-loop without an explicit halt, return what we have
        return all_results
