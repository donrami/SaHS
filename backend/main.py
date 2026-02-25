import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Add the parent directory so we can import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io

# Import from our pipeline
from pipeline import search_hs_code, step_b_translate_and_map, step_c_generate_csv

try:
    from google import genai
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception:
    gemini_client = None

app = FastAPI(title="HS Code Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class InvoiceItem(BaseModel):
    item_description: str
    quantity: Optional[str] = "1"
    country_of_origin: Optional[str] = ""

class ClassifyRequest(BaseModel):
    items: List[InvoiceItem]

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class HSCodeResult(BaseModel):
    hs_code: str
    desc_en: str
    desc_ar: str
    distance: float
    hierarchy_path: str = ""
    duty_rate: str = ""
    procedures: str = ""

# --- Routes ---

@app.get("/")
def root():
    return {"status": "HS Code Classification API is running"}

@app.post("/search", response_model=List[HSCodeResult])
def search(req: SearchRequest):
    """Directly search the vector DB for HS codes."""
    try:
        results = search_hs_code(req.query, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Classify a list of invoice items using ChromaDB + Gemini.
    Returns enriched items with hs_code, arabic_description, and reasoning.
    """
    try:
        items_as_dicts = [item.model_dump() for item in req.items]
        mapped = step_b_translate_and_map(items_as_dicts, gemini_client=gemini_client)
        return {"results": mapped}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/csv")
async def classify_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with a column 'item_description' and get back a classified CSV.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), dtype=str)
        if "item_description" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have an 'item_description' column")

        items = df.to_dict(orient="records")
        mapped = step_b_translate_and_map(items, gemini_client=gemini_client)

        out_path = "/tmp/classified_output.csv"
        step_c_generate_csv(mapped, output_filename=out_path)
        return FileResponse(out_path, media_type="text/csv", filename="classified_output.csv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
def export_last_result():
    """Download the last generated CSV."""
    path = "processed_invoice.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No processed invoice found. Run classify first.")
    return FileResponse(path, media_type="text/csv", filename="processed_invoice.csv")
