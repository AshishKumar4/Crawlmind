from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Any, Optional

from crawlmind.llm_completion import llmCompletionWithRetry, LLMCompletionRequest, LLMCompletionNonStreamingResponse

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Structured Content
# -------------------------------------------------------------------------
@dataclass
class StructuredContent:
    context: str
    summary: str
    technical_terms: List[str]
    unique_terminologies: List[str]
    concepts_ideas: List[str]
    people_places_events: List[str]
    dates_timelines: List[str]
    links_references: List[str]
    other_keywords: List[str] = field(default_factory=list)
    unstructured_content: str = ""

    @staticmethod
    def empty() -> StructuredContent:
        """
        Returns an empty StructuredContent object to use as fallback.
        """
        return StructuredContent(
            context="",
            summary="",
            technical_terms=[],
            unique_terminologies=[],
            concepts_ideas=[],
            people_places_events=[],
            dates_timelines=[],
            links_references=[],
            other_keywords=[],
            unstructured_content=""
        )
    
# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------
structured_content_description = """
{
    "context": "The context of the content",
    "summary": "A concise summary of the content",
    "technical_terms": ["Technical Term 1", "Technical Term 2"],
    "unique_terminologies": ["Unique Terminology 1", "Unique Terminology 2"],
    "concepts_ideas": ["Concept/Idea 1", "Concept/Idea 2"],
    "people_places_events": ["Person/Place/Event 1", "Person/Place/Event 2"],
    "dates_timelines": ["Date/Timeline 1", "Date/Timeline 2"],
    "links_references": ["Link/Reference 1", "Link/Reference 2"],
    "other_keywords": ["Other Keyword 1", "Other Keyword 2"],
    "unstructured_content": "Any unstructured content that could not be categorized but is important"
}
""".strip()

parse_content_prompt = f"""
You are tasked with structuring raw web content, scraped from a website, for further processing. Given the content, structure it to include:
- The context of the content
- A highly detailed, extensive, cleaned representation of the content as summary. This should be as detailed as possible to capture all the information in the content.
- A list of important technical terms. Can include products names, programming languages, frameworks, tools, protocols, etc.
- Unique terminologies discussed in the content ranked by importance and relevance starting from highly relevant. Can include unique terms, jargon, or domain-specific terms.
- Important concepts and ideas ranked by importance and relevance starting from highly relevant.
- Important people, places, and events mentioned in the content ranked by importance and relevance starting from highly relevant. These should be very specific and not general.
- Important dates and timelines ranked by importance and relevance starting from highly relevant.
- Important links and references
No generic terms should be included in the structured content. The structured content should be concise and to the point.
Keywords and terminologies should not contain single syllable words or small length words or common words like 'the', 'and', 'is', etc.
For small abbreviations, the full form should be included in the structured content.

The response should overall capture all the information present in the web crawl data in a structured format.

In case any information is not present in the content, you can leave the corresponding field empty. If there is no content, you can leave the entire structured content empty.
Do not make up any information that is not present in the content.

The response should be structured in a JSON format as follows:
{structured_content_description}
""".strip()

final_parse_content_prompt = f"""
You are provided with a list of structured web crawl data, scraped from multiple websites related to a particular domain.
Your task is to conduct a thorough research and analysis of the data and provide a well structured, highly detailed report that includes:
- The context of the content in detail
- A concise summary of the content in detail
- A list of important technical terms in detail. Can include products names, programming languages, frameworks, tools, protocols, etc.
- Exhaustive list of all Unique terminologies discussed in the content in detail ranked by importance and relevance starting from highly relevant. Can include unique terms, jargon, or domain-specific terms.
- Exhaustive list of all Important concepts and ideas in detail ranked by importance and relevance starting from highly relevant.
- Exhaustive list of all Important people, places, and events mentioned in the content ranked by importance and relevance starting from highly relevant. These should be very specific and not general.
- Exhaustive list of all Important dates and timelines ranked by importance and relevance starting from highly relevant.
- Exhaustive list of all Important links and references ranked by importance and relevance starting from highly relevant.
No generic terms should be included in the structured content. The structured content should be concise, detailed and to the point.
Keywords and terminologies should not contain single syllable words or small length words or common words like 'the', 'and', 'is', etc.
For small abbreviations, the full form should be included in the structured content.
All the lists must be a super set of all the corresponding lists in the input data.

The response should overall capture all the information present in the web crawl data in a structured format.

In case any information is not present in the content, you can leave the corresponding field empty.
If there is no information present for a particular site, ignore that site.
If there is no content, you can leave the entire structured content empty.
Do not make up any information that is not present in the content.

The input data is already structured and is a JSON list of structured content in the following format:
[{structured_content_description}, ...]

You need to concatenate and combine all the lists in the input data to generate the corresponding lists in the output structured content.

The response should be structured in a JSON format as follows:
{structured_content_description}
""".strip()

# -------------------------------------------------------------------------
# Helper: remove_small_words
# -------------------------------------------------------------------------
def remove_small_words(wordlist: List[str]) -> List[str]:
    """
    Filter out words of length 2 or less.
    """
    return [w for w in wordlist if len(w) > 2]


# -------------------------------------------------------------------------
# ContentParser Class
# -------------------------------------------------------------------------
class ContentParser:
    """
    Python port of the TS ContentParser. Summarizes raw/structured content 
    into a final structured format. Uses an LLM to do so.
    """
    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    async def _summarize(self, prompt: str, data: str) -> StructuredContent:
        """
        Private helper method to handle the summarization logic via LLM.
        """
        try:
            request = LLMCompletionRequest(
                model=self.model,
                system_prompt=prompt,
                messages=[{"role": "user", "content": data}],
                response_format={"type": "json_object"}
            )

            response: LLMCompletionNonStreamingResponse = await llmCompletionWithRetry(request, retries=3)
            # If the response or content is missing, return empty
            if not response or not response.content:
                logger.error("Failed to generate structured content (empty).")
                return StructuredContent.empty()

            # Attempt to parse the content as JSON
            try:
                raw_data = json.loads(response.content)
            except json.JSONDecodeError as exc:
                if response.content.startswith("```json"):
                    # If the response is wrapped in a code block, remove it and try again
                    response.content = response.content[7:-3]
                    try:
                        raw_data = json.loads(response.content)
                    except json.JSONDecodeError as exc:
                        logger.error(f"JSON parse error (code block): {exc}, response: {response.content}, data: {data}")
                        return StructuredContent.empty()
                logger.error(f"JSON parse error: {exc}, response: {response.content}, data: {data}")
                return StructuredContent.empty()

            # Build a structured content from parsed JSON
            structured_content = StructuredContent(
                context=raw_data.get("context", ""),
                summary=raw_data.get("summary", ""),
                technical_terms=raw_data.get("technical_terms", []),
                unique_terminologies=raw_data.get("unique_terminologies", []),
                concepts_ideas=raw_data.get("concepts_ideas", []),
                people_places_events=raw_data.get("people_places_events", []),
                dates_timelines=raw_data.get("dates_timelines", []),
                links_references=raw_data.get("links_references", []),
                other_keywords=raw_data.get("other_keywords", []),
                unstructured_content=raw_data.get("unstructured_content", "")
            )

            # Filter out all small words from specific fields
            structured_content.technical_terms = remove_small_words(structured_content.technical_terms)
            structured_content.unique_terminologies = remove_small_words(structured_content.unique_terminologies)
            structured_content.concepts_ideas = remove_small_words(structured_content.concepts_ideas)

            return structured_content

        except Exception as e:
            logger.error(f"Failed to parse content: {e}")
            return StructuredContent.empty()

    async def parse_content(self, raw_content: str) -> StructuredContent:
        """
        Parse raw web content into a structured format using parseContent prompt.
        """
        return await self._summarize(parse_content_prompt, raw_content)

    async def parse_final_content(self, data: str) -> StructuredContent:
        """
        Parse final combined data (multiple structured content items) into a single StructuredContent
        using finalParseContentPrompt.
        """
        return await self._summarize(final_parse_content_prompt, data)
