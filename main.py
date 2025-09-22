import os
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from google import genai

# ------------------- ENV + CLIENT -------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")
if not HUGGINGFACE_API_KEY:
    raise ValueError("❌ HUGGINGFACE_API_KEY not found in environment variables!")

client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------- WEBSITE SCRAPER -------------------
def scrape_website(link: str):
    try:
        response = requests.get(link, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        content = "\n".join(paragraphs)
        return {"headings": headings, "content": content}
    except Exception as e:
        return {"error": str(e)}

# ------------------- HUGGING FACE SUMMARIZATION -------------------
HF_MODEL = "sshleifer/distilbart-cnn-12-6"

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
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def nlp_extract_features(content: str):
    if not content:
        return []
    chunks = chunk_text(content)
    bullets = []
    for chunk in chunks:
        summary = hf_summarize(chunk)
        bullets.extend([s.strip() for s in summary.split('.') if s.strip()])
    bullets = list(dict.fromkeys(bullets))
    return bullets

# ------------------- LLM FORMATTING -------------------
def llm_format_markdown(name, link, headings, features, summary):
    headings_text = "\n".join(headings) if headings else "No headings found."
    features_text = "\n".join(features) if features else "No features extracted."
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
  1. **Overview** → Write a 2–3 sentence overview of the company.
  2. **Products & Services** → List all products/services mentioned in headings/content. If none, write "Not available".
  3. **Pricing** → If no pricing found, explicitly state "Not available on the website."
  4. **Features / USPs** → Turn extracted features into clear bullet points. If none, say "Not available".
  5. **Notes** → Add any insights, gaps, or observations.

Rules:
- NEVER leave a section blank. Explicitly state "Not available" if missing.
- Output ONLY valid Markdown.
"""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.candidates[0].content.parts[0].text
    except Exception:
        return "Error: Could not parse Gemini output."

# ------------------- FULL PIPELINE -------------------
def competitor_research(name: str, link: str):
    result = {"name": name, "link": link, "headings": [], "features": [], "summary": "", "markdown": ""}
    scraped = scrape_website(link)
    if "error" in scraped:
        result["error"] = scraped["error"]
        return result
    result["headings"] = scraped["headings"]
    result["summary"] = scraped["content"][:4000]
    result["features"] = nlp_extract_features(scraped["content"])
    result["markdown"] = llm_format_markdown(
        name=result["name"],
        link=result["link"],
        headings=result["headings"],
        features=result["features"],
        summary=result["summary"]
    )
    return result

# ------------------- FASTAPI -------------------
app = FastAPI()

@app.get("/markdown", response_class=PlainTextResponse)
def get_markdown(
    name: str = Query(..., description="Competitor name"),
    link: str = Query(..., description="Competitor website URL")
):
    research_result = competitor_research(name, link)
    if "error" in research_result:
        return f"Error: {research_result['error']}"
    return research_result.get("markdown", "No markdown generated.")

@app.get("/research")
def get_full_research(
    name: str = Query(..., description="Competitor name"),
    link: str = Query(..., description="Competitor website URL")
):
    return competitor_research(name, link)

@app.get("/test")
def test():
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Respond with just 'ok'"
    )
    return {"result": response.candidates[0].content.parts[0].text}

# ------------------- MAIN -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

