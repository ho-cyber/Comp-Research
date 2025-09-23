import os
import time
import logging
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import asyncio
import requests

# ------------------- SETUP -------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

HF_MODEL = "sshleifer/distilbart-cnn-12-6"

app = FastAPI()

# ✅ CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "https://yourfrontend.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- PLAYWRIGHT SCRAPER -------------------
async def scrape_website_playwright(base_url: str, max_depth=5, max_pages=100, max_time=180, delay=1):
    visited = set()
    headings_all = []
    paragraphs_all = []
    pages_scanned = 0
    start_time = time.time()

    async def crawl(url, depth):
        nonlocal pages_scanned
        if (
            depth > max_depth
            or url in visited
            or pages_scanned >= max_pages
            or time.time() - start_time > max_time
        ):
            return

        visited.add(url)
        pages_scanned += 1
        logging.info(f"🔹 Crawling: {url} (Depth {depth})")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=30000)
                await asyncio.sleep(delay)  # wait for JS to load
                content = await page.content()
                await browser.close()
        except Exception as e:
            logging.warning(f"⚠️ Failed to load {url}: {e}")
            return

        soup = BeautifulSoup(content, "html.parser")
        headings_all.extend([h.get_text(strip=True) for h in soup.find_all(['h1','h2','h3'])])
        paragraphs_all.extend([p.get_text(strip=True) for p in soup.find_all('p')])

        # internal links
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                await crawl(full_url, depth + 1)

    await crawl(base_url, 0)
    content = "\n".join(paragraphs_all)
    return {"headings": list(dict.fromkeys(headings_all)), "content": content}

# ------------------- SUMMARIZATION -------------------
def hf_summarize(text: str):
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": text, "parameters": {"max_length": 150, "min_length": 30}}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "summary_text" in result[0]:
            return result[0]["summary_text"]
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def chunk_text(text, chunk_size=800):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def nlp_extract_features(content: str):
    if not content:
        return []
    chunks = chunk_text(content)
    bullets = []
    for chunk in chunks[:5]:  # limit for speed
        summary = hf_summarize(chunk)
        bullets.extend([s.strip() for s in summary.split('.') if s.strip()])
    return list(dict.fromkeys(bullets))

# ------------------- LLM MARKDOWN -------------------
def llm_format_markdown(name, link, headings, features, summary):
    headings_text = "\n".join(headings) if headings else "No headings found."
    features_text = "\n".join(features) if features else "Not available."
    prompt = f"""
You are a professional business analyst. 
You are given raw competitor data and must generate a structured Markdown report.

Competitor Name: {name}
Website: {link}

Headings found on website:
{headings_text}

Extracted Features / Insights:
{features_text}

Website Summary (raw content, may be messy):
{summary}

Task:
- Generate a structured Markdown report with these sections:
  1. **Overview** (2–3 sentences)
  2. **Products & Services** (list or "Not available")
  3. **Pricing** (or "Not available")
  4. **Features / USPs** (bullet points or "Not available")
  5. **Notes** (observations or gaps)

Rules:
- NEVER leave a section blank. Explicitly state "Not available" if missing.
- Output ONLY valid Markdown.
"""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        logging.error(f"Gemini failed: {e}")
        return "Error: Could not parse Gemini output."

# ------------------- PIPELINE -------------------
async def competitor_research(name: str, link: str):
    result = {"name": name, "link": link, "headings": [], "features": [], "summary": "", "markdown": ""}
    scraped = await scrape_website_playwright(link)
    result["headings"] = scraped["headings"]
    result["summary"] = scraped["content"][:4000]
    result["features"] = nlp_extract_features(scraped["content"])
    result["markdown"] = llm_format_markdown(
        name=name,
        link=link,
        headings=result["headings"],
        features=result["features"],
        summary=result["summary"]
    )
    return result

# ------------------- FASTAPI -------------------
@app.get("/markdown", response_class=PlainTextResponse)
async def get_markdown(
    name: str = Query(..., description="Competitor name"),
    link: str = Query(..., description="Competitor website URL")
):
    logging.info(f"📥 Request received: /markdown?name={name}&link={link}")
    research_result = await competitor_research(name, link)
    logging.info(f"✅ Successfully generated markdown for {name}")
    return research_result.get("markdown", "No markdown generated.")
