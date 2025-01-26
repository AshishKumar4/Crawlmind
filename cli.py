import asyncio
import json
import logging

from crawlmind.logger import logger
from crawlmind.crawl_manager import CrawlManager

async def main() -> None:
    # The starting URL (same as in the TS code)
    # start_url = input("Enter the URL to crawl: ")
    start_url = "https://ashishkumarsingh.com"

    # Build the CrawlManager with four sets of options
    crawler = CrawlManager(
        summary_options={
            "model": "gpt-4o",
            "additional_instructions": ""
        },
        action_options={
            "model": "gpt-4o",
            "additional_instructions": ""
        },
        parser_options={
            "model": "gpt-4o-mini",
            "additional_instructions": ""
        },
    )

    logger.info(f"Starting crawl on {start_url}")

    # Perform crawl with maxDepth=20 (as in TS code)
    results = await crawler.crawl(start_url, 20)

    # Log the crawl results in JSON
    logger.info(f"\n\n\nCrawl complete. Results: {json.dumps(results, default=lambda o: o.__dict__, indent=2)}")

    # Close the browser
    await crawler.close_browser()
    logger.info("Browser closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"CLI encountered an error: {e}")
