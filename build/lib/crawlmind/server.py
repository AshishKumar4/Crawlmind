import logging
import uuid
import threading
import requests

from flask import Flask, request, jsonify
from crawlmind.logger import logger  # your enhanced logger
from crawlmind.crawl_manager import CrawlManager

app = Flask(__name__)

# A dictionary to hold crawl job state in memory.
# Keys: job_id (string)
# Value: { "url": str, "status": str, "result": any, "job": Optional[threading.Thread] }
crawl_jobs = {}

@app.route("/crawl", methods=["POST"])
def create_crawl():
    """
    POST /crawl
    Body JSON: {
      "url": str,
      "webhook": str,
      "summary_options"?: dict,
      "action_options"?: dict,
      "parser_options"?: dict
    }
    Returns: {
      "success": bool,
      "id": string
    }
    """
    data = request.get_json() or {}
    url = data.get("url")
    webhook = data.get("webhook")
    summary_options = data.get("summary_options")
    action_options = data.get("action_options")
    parser_options = data.get("parser_options")

    if not url or not webhook:
        return jsonify({"success": False, "message": "Missing url or webhook in request body"}), 400

    crawl_id = str(uuid.uuid4())
    crawl_jobs[crawl_id] = {"status": "pending", "result": None, "url": url, "job": None}

    # Respond immediately with the job ID
    response = {"success": True, "id": crawl_id}
    logger.info(f"Received /crawl request for {url}, assigned job id {crawl_id}")
    # Start background thread to do the actual crawl
    job_thread = threading.Thread(
        target=launch_job,
        args=(crawl_id, url, webhook, summary_options, action_options, parser_options),
        daemon=True
    )
    job_thread.start()

    crawl_jobs[crawl_id]["job"] = job_thread

    return jsonify(response)

@app.route("/crawl/<string:crawl_id>", methods=["GET"])
def get_crawl(crawl_id):
    """
    GET /crawl/<crawl_id>
    Returns the status and, if completed, the results
    """
    if crawl_id not in crawl_jobs:
        return jsonify({"success": False, "message": "Crawl job not found"}), 404

    job_data = crawl_jobs[crawl_id]
    return jsonify({
        "success": True,
        "status": job_data["status"],
        "result": job_data["result"]
    })

def launch_job(
    crawl_id: str,
    url: str,
    webhook: str,
    summary_options: dict,
    action_options: dict,
    parser_options: dict
):
    """
    Background function that orchestrates the crawl, updates the job state, and sends a webhook.
    """
    logger.info(f"Starting crawl job {crawl_id} for {url}")

    default_options = {
        "model": "gpt-4o",
        "additional_instructions": ""
    }

    # Build the manager with the provided options or defaults
    crawler = CrawlManager(
        summary_options or default_options,
        action_options or default_options,
        parser_options or {"model": "gpt-4o", "additional_instructions": ""},
    )

    # Mark the job as in progress
    crawl_jobs[crawl_id]["status"] = "in_progress"

    try:
        # Do the crawl with a fixed maxDepth=20
        results = asyncio_run_crawl(crawler, url)
        crawl_jobs[crawl_id]["status"] = "completed"
        crawl_jobs[crawl_id]["result"] = results

        logger.info(f"Crawl job {crawl_id} complete. Sending webhook to {webhook}")

        # Send the webhook
        try:
            requests.post(webhook, json={
                "id": crawl_id,
                "results": results,  # Possibly large
                "url": url
            })
            logger.info(f"Webhook for crawl job {crawl_id} sent successfully")
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook}: {e}")

    except Exception as e:
        crawl_jobs[crawl_id]["status"] = "failed"
        crawl_jobs[crawl_id]["result"] = None
        logger.error(f"Crawl job {crawl_id} failed: {e}")

    # Always close the browser
    try:
        asyncio_run_close(crawler)
    except Exception as e:
        logger.error(f"Error closing browser for crawl job {crawl_id}: {e}")
    logger.info(f"Browser for crawl job {crawl_id} closed")

def asyncio_run_crawl(crawler: CrawlManager, url: str):
    """
    Synchronous wrapper for an async method `crawler.crawl(...)`.
    This blocks the thread until the async call completes.
    """
    import asyncio
    return asyncio.run(crawler.crawl(url, 20))

def asyncio_run_close(crawler: CrawlManager):
    """
    Synchronous wrapper for the async method `crawler.close_browser()`.
    """
    import asyncio
    return asyncio.run(crawler.close_browser())

if __name__ == "__main__":
    # Start the Flask server
    app.run(host="0.0.0.0", port=5000, debug=True)
