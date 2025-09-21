import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from google import genai

# ------------------- ENV + CLIENT -------------------
# Load .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables!")

client = genai.Client(api_key=api_key)

# ------------------- WEBSITE SCRAPER -------------------
def scrape_website(link: str):
    """
    Fetch website content + headings
    """
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

# ------------------- NLP LAYER -------------------
nlp_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, chunk_size=800):
    """
    Split text into smaller chunks
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def nlp_extract_features(content: str):
    """
    Extract bullet-point features from text using NLP summarization
    """
    if not content:
        return []

    chunks = chunk_text(content)
    bullets = []
    for chunk in chunks:
        summary = nlp_summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        bullets.extend([s.strip() for s in summary.split('.') if s.strip()])
    # Remove duplicates
    bullets = list(dict.fromkeys(bullets))
    return bullets

# ------------------- LLM FORMATTING LAYER -------------------
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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
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
    result["summary"] = scraped["content"][:4000]  # limit size
    result["features"] = nlp_extract_features(scraped["content"])
    result["markdown"] = llm_format_markdown(
        name=result["name"],
        link=result["link"],
        headings=result["headings"],
        features=result["features"],
        summary=result["summary"]
    )
    return result

# ------------------- SAVE TO README -------------------
def save_competitor_to_readme(research_result: dict, filename: str = "README.md"):
    with open(filename, "w", encoding="utf-8") as f:
        if research_result.get("markdown"):
            f.write(research_result["markdown"])
        else:
            f.write(f"# Competitor Research: {research_result['name']}\n\n")
            f.write(f"**Website:** [{research_result['link']}]({research_result['link']})\n\n")
            if research_result.get("summary"):
                f.write("## Summary\n")
                f.write(f"{research_result['summary']}\n\n")
            if research_result.get("headings"):
                f.write("## Headings Found\n")
                for h in research_result["headings"]:
                    f.write(f"- {h}\n")
            if research_result.get("features"):
                f.write("## Features / Insights\n")
                for ftr in research_result["features"]:
                    f.write(f"- {ftr}\n")
        f.write("\n---\nGenerated via Competitor Research Dashboard\n")

# ------------------- FASTAPI APP -------------------
app = FastAPI()

@app.get("/markdown", response_class=PlainTextResponse)
def get_markdown(
    name: str = Query(..., description="Competitor name"),
    link: str = Query(..., description="Competitor website URL")
):
    """
    Run competitor research and return ONLY the Markdown report.
    """
    research_result = competitor_research(name, link)
    if "error" in research_result:
        return f"Error: {research_result['error']}"
    return research_result.get("markdown", "No markdown generated.")

@app.get("/research")
def get_full_research(
    name: str = Query(..., description="Competitor name"),
    link: str = Query(..., description="Competitor website URL")
):
    """
    Run competitor research and return the full JSON object.
    """
    return competitor_research(name, link)

@app.get("/test")
def test():
    """
    Simple test route to check if Gemini API works.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Respond with just 'ok'"
    )
    return {"result": response.candidates[0].content.parts[0].text}

# ------------------- MAIN -------------------
if __name__ == "__main__":
    competitor = competitor_research(
        name="Brandwell AI",
        link="https://brandwell.ai/"
    )
    # save_competitor_to_readme(competitor, filename="Competitor_Example.md")
    # print("✅ Competitor research saved to Competitor_Example.md")
