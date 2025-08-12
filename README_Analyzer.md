# DocBot Image Analyzer (FastAPI)

## Deploy en Railway
1. Crea un repo en GitHub y sube estos archivos (mantén `analyzer_app/main.py`, `requirements.txt` y `Procfile` en la raíz).
2. En Railway: **New → Deploy from GitHub** y selecciona tu repo.
3. En Variables, añade `OPENAI_API_KEY`.
4. Cuando esté *Running*, prueba `GET /health` en el navegador.

## Uso
`POST /analyze` con JSON:
{
  "image_url": "https://i.imgur.com/your_food_photo.jpg"
}