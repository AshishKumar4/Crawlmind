from scraper import Scraper, ScraperOptions
import asyncio

async def main():
    options = ScraperOptions()
    async with Scraper(options) as scraper:
        response = await scraper.scrape_content("https://ashishkumarsingh.com")
        print(response.content)
        print(response.links)

if __name__ == "__main__":
    asyncio.run(main())