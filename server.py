import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="ChatNUR API")
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise RuntimeError("Set OPENAI_API_KEY as a private server environment variable.")
client = OpenAI(api_key=key)
model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
instructions = Path(__file__).with_name("personality.txt").read_text(encoding="utf-8")

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message or len(message) > 12000:
        raise HTTPException(status_code=400, detail="Message must contain 1–12000 characters")
    try:
        result = client.responses.create(
            model=model,
            instructions=instructions,
            input=message,
        )
        return {"reply": result.output_text}
    except Exception:
        raise HTTPException(status_code=502, detail="AI service temporarily unavailable")
