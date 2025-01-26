from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Any, Optional
import asyncio

# If you have a custom llmCompletion function, import it:
# from utils import llm_completion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class StructuredContent:
    """
    Data class mirroring the StructuredContent interface in TypeScript.
    """
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
        Returns an empty StructuredContent object to use as a fallback.
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


# These prompts mirror the TypeScript constants in contentParser.ts
STRUCTURED_CONTENT_DESCRIPTION = """
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
"""

PARSE_CONTENT_PROMPT = f"""
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
{STRUCTURED_CONTENT_DESCRIPTION}
"""

FINAL_PARSE_CONTENT_PROMPT = f"""
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
All the lists must be a superset of all the corresponding lists in the input data.

In case any information is not present in the content, you can leave the corresponding field empty. 
If there is no information present for a particular site, ignore that site.
If there is no content, you can leave the entire structured content empty. 
Do not make up any information that is not present in the content.

The input data is already structured and is a JSON list of structured content in the following format:
[{STRUCTURED_CONTENT_DESCRIPTION}, ...]

You need to concatenate and combine all the lists in the input data to generate the corresponding lists in the output structured content.

The response should be structured in a JSON format as follows:
{STRUCTURED_CONTENT_DESCRIPTION}
"""


def remove_small_words(wordlist: List[str]) -> List[str]:
    """
    Filter out words of length 2 or less.
    """
    return [word for word in wordlist if len(word) > 2]


class ContentParser:
    """
    Python equivalent of the TypeScript ContentParser class.
    """
    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    async def _summarize(self, prompt: str, data: str) -> StructuredContent:
        """
        Private helper method to handle the summarization logic via LLM.
        """
        # Replace this call with the actual llmCompletion or your LLM invocation
        # The 'llmCompletion' function in TypeScript presumably returns:
        # {
        #   content: "<json string>"
        # }
        try:
            # Example placeholder implementation; replace with your LLM call:
            # resp = await llm_completion(
            #     model=self.model,
            #     system_prompt=prompt,
            #     messages=[{"role": "user", "content": data}],
            #     response_format={"type": "json_object"}
            # )
            #
            # For demonstration, let's just pretend it returned the following:
            resp = {
                "content": json.dumps({
                    "context": "Sample context from LLM",
                    "summary": "Sample summary from LLM",
                    "technical_terms": ["Python", "React", "API"],
                    "unique_terminologies": ["Quantum Entanglement"],
                    "concepts_ideas": ["Neural Networks"],
                    "people_places_events": ["John Doe", "OpenAI HQ"],
                    "dates_timelines": ["2023-01-01"],
                    "links_references": ["http://example.com"],
                    "other_keywords": ["Extra data"],
                    "unstructured_content": "Some leftover content"
                })
            }

            # If resp is None or empty, handle gracefully
            if not resp or "content" not in resp:
                logger.error("Failed to generate structured content (empty response).")
                return StructuredContent.empty()

            raw_json_str: str = resp["content"]
            parsed_data: Any = json.loads(raw_json_str)

            # Build a StructuredContent object safely
            structured_content = StructuredContent(
                context=parsed_data.get("context", ""),
                summary=parsed_data.get("summary", ""),
                technical_terms=parsed_data.get("technical_terms", []),
                unique_terminologies=parsed_data.get("unique_terminologies", []),
                concepts_ideas=parsed_data.get("concepts_ideas", []),
                people_places_events=parsed_data.get("people_places_events", []),
                dates_timelines=parsed_data.get("dates_timelines", []),
                links_references=parsed_data.get("links_references", []),
                other_keywords=parsed_data.get("other_keywords", []),
                unstructured_content=parsed_data.get("unstructured_content", "")
            )

            # Filter out all small words in certain fields
            structured_content.technical_terms = remove_small_words(structured_content.technical_terms)
            structured_content.unique_terminologies = remove_small_words(structured_content.unique_terminologies)
            structured_content.concepts_ideas = remove_small_words(structured_content.concepts_ideas)

            return structured_content
        except Exception as exc:
            logger.error(f"Failed to parse content: {exc}")
            return StructuredContent.empty()

    async def parse_content(self, raw_content: str) -> StructuredContent:
        """
        Asynchronously parse raw web content into a StructuredContent object using the parse prompt.
        """
        return await self._summarize(PARSE_CONTENT_PROMPT, raw_content)

    async def parse_final_content(self, data: str) -> StructuredContent:
        """
        Asynchronously parse final combined data (multiple structured content items) into a single StructuredContent.
        """
        return await self._summarize(FINAL_PARSE_CONTENT_PROMPT, data)


# Example usage (in an async context):
# async def main():
#     parser = ContentParser(model="gpt-4o")
#     structured = await parser.parse_content("Some raw content from the web.")
#     print(structured)
#
# if __name__ == "__main__":
#     asyncio.run(main())
