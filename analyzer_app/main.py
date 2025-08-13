# analyzer_app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Tuple
import os, json, re

# Env flags
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY", "").strip())
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")  # 1ª pasada (rápida)
ESCALATE_MODEL = os.getenv("ESCALATE_MODEL", "gpt-4o")      # 2ª pasada (precisa)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))  # si < umbral, escalamos

app = FastAPI(title="DocBot Image Analyzer", version="0.2.0")

class AnalyzeRequest(BaseModel):
    image_url: HttpUrl

class AnalyzeResponse(BaseModel):
    items: List[str]
    carbs_g: Optional[int]
    protein_g: Optional[int]
    fat_g: Optional[int]
    kcal: Optional[int]
    veredicto: str  # "verde" | "amarillo" | "rojo" | "gris"
    razon: str
    sugerencia: str
    confidence: Optional[float] = None

# --- Reglas duras (heurísticas) ---
RED_PATTERNS = [
    r"refresco|soda|gaseosa|cola|sprite|fanta|manzana\s*lift",
    r"jugo\s*(de\s*caja|industrial|embotellado)",
    r"(bebida|energy)\s*(energ[eé]tica)",
    r"malteada|frapp[eé]|frappuccino|jarabe",
    r"fritura|papas?\s*fritas|chips|cheetos|doritos|nachos|totopos\s*de\s*bolsa",
    r"pastel|pan\s*dulce|muffin|donut|dona|galletas?\s*(rellenas)?|croissant",
]
GREEN_PATTERNS = [
    r"agua(\s*natural)?$",
    r"caf[eé]\s*(solo|negro|americano)$",
    r"t[eé]\s*(sin\s*az[uú]car)?$",
    r"ensalada(\s*(verde|mixta))?(\s*(sin|con\s*poco)\s*aderezo)?",
    r"verduras?\s*(al\s*vapor|asadas?)",
]
YELLOW_PATTERNS = [
    r"taco(s)?|torta(s)?|tamales?|quesadillas?",
    r"arroz|pasta|espagueti|las[aá]n?a|fideos",
    r"pan(\s*integral)?|tortillas?",
    r"fruta(s)?\s*(entera)?",
    r"gelatina|cereal",
]

CATEGORY_REASONS = {
    "rojo": "Regla: ultraprocesado/alto en azúcar o frito Ò rojo.",
    "amarillo": "Regla: fuente moderada de carbohidratos/porción Ò amarillo.",
    "verde": "Regla: bebida/comida sin azúcar añadida o verduras Ò verde.",
}

def match_any(patterns: List[str], text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def rules_override(items: List[str]) -> Optional[Tuple[str, str]]:
    """Devuelve (veredicto, razon) si una regla aplica; si no, None."""
    for it in items:
        if match_any(RED_PATTERNS, it):
            return "rojo", CATEGORY_REASONS["rojo"]
    for it in items:
        if match_any(GREEN_PATTERNS, it):
            return "verde", CATEGORY_REASONS["verde"]
    for it in items:
        if match_any(YELLOW_PATTERNS, it):
            return "amarillo", CATEGORY_REASONS["amarillo"]
    return None

# --- OpenAI helpers ---
PROMPT_JSON = (
    "Eres nutricionista para personas con diabetes tipo 2. Analiza la foto y devuelve SOLO un JSON con este esquema exacto: "
    "{"
    "\"items\":[\"string\"],"
    "\"carbs_g\": int | null,"
    "\"protein_g\": int | null,"
    "\"fat_g\": int | null,"
    "\"kcal\": int | null,"
    "\"veredicto\": \"verde\" | \"amarillo\" | \"rojo\" | \"gris\","
    "\"razon\": \"string\","
    "\"sugerencia\": \"string\","
    "\"confidence\": float"
    "}. "
    "Criterio: verde=adecuado; amarillo=moderar por carbohidratos/porción; rojo=alto en azúcares/ultraprocesados; gris=incierto. "
    "Sé concreto y evita el \"gris\" salvo que la imagen no permita clasificar. No incluyas nada fuera del JSON."
)

def call_openai(model: str, url: str, temperature: float = 0.2) -> dict:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_JSON},
                {"type": "image_url", "image_url": {"url": str(url)}}
            ]
        }],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or "{}"
    # Limpieza por si viniera envuelto en ```
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            body = parts[1]
            content = body[4:].strip() if body.strip().startswith("json") else body.strip()
    try:
        return json.loads(content)
    except Exception:
        return {}

def analyze_pipeline(url: str) -> AnalyzeResponse:
    # 1) Primera pasada (rápida)
    data = call_openai(PRIMARY_MODEL, url, temperature=0.2) if USE_OPENAI else {}

    # Fallback si no hay datos
    if not data:
        return AnalyzeResponse(
            items=[], carbs_g=None, protein_g=None, fat_g=None, kcal=None,
            veredicto="gris", razon="No se pudo leer la imagen.",
            sugerencia="Toma la foto de frente, con buena luz.", confidence=0.0,
        )

    # Normaliza campos
    items = [str(x) for x in (data.get("items") or [])]
    veredicto = (data.get("veredicto") or "gris").lower()
    razon = data.get("razon") or ""
    sugerencia = data.get("sugerencia") or ""
    confidence = float(data.get("confidence") or 0.0)

    # 2) Reglas duras (prioridades: platos caseros, snacks salados, bebidas azucaradas)
    override = rules_override(items)
    if override:
        veredicto, rule_reason = override
        razon = f"{razon} {rule_reason}".strip()

    # 3) ¿Escalamos? Si veredicto sigue gris o la confianza es baja
    if veredicto == "gris" or confidence < CONF_THRESHOLD:
        data2 = call_openai(ESCALATE_MODEL, url, temperature=0.1) if USE_OPENAI else {}
        if data2:
            items = [str(x) for x in (data2.get("items") or items)] or items
            veredicto = (data2.get("veredicto") or veredicto).lower()
            razon = data2.get("razon") or razon
            sugerencia = data2.get("sugerencia") or sugerencia
            confidence = float(data2.get("confidence") or confidence)
            # Reaplicar reglas
            override2 = rules_override(items)
            if override2:
                veredicto, rule_reason = override2
                razon = f"{razon} {rule_reason}".strip()

    # Forzar color si aún está gris (tu preferencia)
    if veredicto == "gris":
        veredicto = "amarillo"
        razon = (razon + " Forzado a amarillo por política de clasificación.").strip()

    return AnalyzeResponse(
        items=items,
        carbs_g=data.get("carbs_g"),
        protein_g=data.get("protein_g"),
        fat_g=data.get("fat_g"),
        kcal=data.get("kcal"),
        veredicto=veredicto,
        razon=razon,
        sugerencia=sugerencia,
        confidence=confidence,
    )

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return analyze_pipeline(str(req.image_url))

@app.get("/health")
def health():
    return {"ok": True, "model_primary": PRIMARY_MODEL, "model_escalate": ESCALATE_MODEL}