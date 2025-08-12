# analyzer_app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os, json

# If OPENAI_API_KEY is present, use OpenAI vision; otherwise return a safe fallback
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY", "").strip())

app = FastAPI(title="DocBot Image Analyzer", version="0.1.0")

class AnalyzeRequest(BaseModel):
    image_url: HttpUrl

class AnalyzeResponse(BaseModel):
    items: List[str]
    carbs_g: Optional[int]
    protein_g: Optional[int]
    fat_g: Optional[int]
    kcal: Optional[int]
    veredicto: str
    razon: str
    sugerencia: str

def analyze_fallback(url: str) -> AnalyzeResponse:
    return AnalyzeResponse(
        items=[],
        carbs_g=None,
        protein_g=None,
        fat_g=None,
        kcal=None,
        veredicto="gris",
        razon="Análisis básico sin visión activada.",
        sugerencia="Activa OPENAI_API_KEY para análisis real o envía la comida por texto."
    )

def analyze_with_openai(url: str) -> AnalyzeResponse:
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = (
            "Eres nutricionista. Analiza la foto de comida y responde SOLO un JSON con este esquema exacto: "
            "{"
            "\"items\":[\"string\"],"
            "\"carbs_g\": int | null,"
            "\"protein_g\": int | null,"
            "\"fat_g\": int | null,"
            "\"kcal\": int | null,"
            "\"veredicto\": \"verde\" | \"amarillo\" | \"rojo\" | \"gris\","
            "\"razon\": \"string\","
            "\"sugerencia\": \"string\""
            "}. "
            "Criterio: verde=adecuado para T2D; amarillo=moderar por carbohidratos/porción; "
            "rojo=alto en carbohidratos/azúcar/ultraprocesados; gris=incierto. "
            "No incluyas nada fuera del JSON."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user","content":[
                    {"type":"text","text": prompt},
                    {"type":"image_url","image_url":{"url": str(url)}}
                ]}
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        if "```" in content:
            # remove code fences if present
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.strip().startswith("json"):
                    content = content.strip()[4:].strip()
        data = json.loads(content)
        return AnalyzeResponse(**data)
    except Exception:
        return analyze_fallback(url)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if USE_OPENAI:
        return analyze_with_openai(str(req.image_url))
    else:
        return analyze_fallback(str(req.image_url))

@app.get("/health")
def health():
    return {"ok": True}
