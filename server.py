from fastapi import FastAPI
from main import competitor_research

app = FastAPI()

@app.get("/")
async def root():
    # Example competitor research call
    result = competitor_research(
        name="Brandwell AI",
        link="https://brandwell.ai/"
    )
    return result
