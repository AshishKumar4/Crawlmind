from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

from crawlmind.llm_completion import llmCompletionWithRetry, LLMCompletionRequest

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1) Python Dataclass for final usage
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
# 2) Pydantic Model for the new structured outputs API
# -------------------------------------------------------------------------
class StructuredContentModel(BaseModel):
    context: str 
    summary: str
    technical_terms: List[str] = Field(default_factory=list)
    unique_terminologies: List[str] = Field(default_factory=list)
    concepts_ideas: List[str] = Field(default_factory=list)
    people_places_events: List[str] = Field(default_factory=list)
    dates_timelines: List[str] = Field(default_factory=list)
    links_references: List[str] = Field(default_factory=list)
    other_keywords: List[str] = Field(default_factory=list)
    unstructured_content: str

# -------------------------------------------------------------------------
# 3) Prompts and helper
# -------------------------------------------------------------------------

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

""".strip()

final_parse_content_prompt = f"""
You are provided with a list of structured web crawl data, scraped from multiple websites related to a particular domain.
Your task is to conduct a thorough research and analysis of the data and provide a well structured, highly detailed report that includes:
- The context of the content in detail
- A highly descriptive, rich summary of the content in detail
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

You need to concatenate and combine all the lists in the input data to generate the corresponding lists in the output structured content.
""".strip()

def remove_small_words(wordlist: List[str]) -> List[str]:
    """Filter out words of length 2 or less."""
    return [w for w in wordlist if len(w) > 2]

# -------------------------------------------------------------------------
# 4) ContentParser Class (using new structured outputs API)
# -------------------------------------------------------------------------
class ContentParser:
    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        # Create a single OpenAI client here if you wish to reuse it
        self.client = OpenAI()

    async def _summarize(self, prompt: str, data: str) -> StructuredContent:
        """
        Private helper method that uses the new `client.beta.chat.completions.parse`
        to get a typed, validated `StructuredContentModel`.
        """
        try:
            # 1) Make an async call to OpenAI's new structured output API
            response = await llmCompletionWithRetry(
                request=LLMCompletionRequest(
                    model=self.model,
                    system_prompt=prompt,
                    messages=[{"role": "user", "content": data}],
                    response_format={"type": "pydantic_model", "pydantic_class": StructuredContentModel},
                ),
                retries=3,
            )

            # 2) Extract the parsed Pydantic model
            parsed_model: StructuredContentModel = response.parsed
            if not parsed_model:
                logger.error("Structured output parse returned null or empty.")
                return StructuredContent.empty()

        except ValidationError as ve:
            logger.error(f"Pydantic validation error: {ve}")
            return StructuredContent.empty()
        except Exception as e:
            logger.error(f"Failed to parse content via structured outputs: {e}")
            return StructuredContent.empty()
        # 3) Convert to our final StructuredContent dataclass
        structured_content = StructuredContent(
            context=parsed_model.context or "",
            summary=parsed_model.summary or "",
            technical_terms=parsed_model.technical_terms or [],
            unique_terminologies=parsed_model.unique_terminologies or [],
            concepts_ideas=parsed_model.concepts_ideas or [],
            people_places_events=parsed_model.people_places_events or [],
            dates_timelines=parsed_model.dates_timelines or [],
            links_references=parsed_model.links_references or [],
            other_keywords=parsed_model.other_keywords or [],
            unstructured_content=parsed_model.unstructured_content or ""
        )

        # 4) Filter out small words from certain fields
        structured_content.technical_terms = remove_small_words(structured_content.technical_terms)
        structured_content.unique_terminologies = remove_small_words(structured_content.unique_terminologies)
        structured_content.concepts_ideas = remove_small_words(structured_content.concepts_ideas)

        return structured_content

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
