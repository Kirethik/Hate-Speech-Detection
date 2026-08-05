from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path
import os
import traceback

# Silence HF symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ensure the repository root is in the path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model_b_generation.gating import moderate

app = FastAPI(title="Civitas AI Moderation API")

# Setup CORS to allow the Vite React app to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModerationRequest(BaseModel):
    text: str
    language: str

@app.post("/api/moderate")
async def moderate_endpoint(req: ModerationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Map friendly language names to internal codes if necessary, or accept codes directly
    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "en": "en",
        "hi": "hi",
        "ta": "ta"
    }
    lang_code = lang_map.get(req.language)
    if not lang_code:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {req.language}")

    try:
        result = moderate(req.text, lang_code)
        return result
    except Exception as e:
        print(f"Error during moderation: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during moderation.")

if __name__ == "__main__":
    import uvicorn
    # Run server on port 8000
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
