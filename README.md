# TechnicalDoc AI Assistant

Asistente RAG para documentación técnica (`.md`, `.txt`, `.pdf`) con aislamiento multi-Knowledge Base, streaming en tiempo real y validación determinista de citas. Construido con **FastAPI**, **LangChain**, **ChromaDB** y modelos locales.

---

## Características

- **Multi-Knowledge Base:** Bases de conocimiento aisladas con catálogo SQLite y ChromaDB.
- **Citas Verificables:** Respuestas con referencias `[n]` validadas contra las fuentes y prevención de alucinaciones.
- **Streaming en Vivo:** Generación en tiempo real (SSE) con interfaz web integrada.
- **Modo Profesor:** Resolución pedagógica paso a paso con verificación simbólica y cálculo exacto.

---

## Inicio Rápido

### Requisitos
- Python 3.12+ y [`uv`](https://github.com/astral-sh/uv)
- LLM local ([LM Studio](https://lmstudio.ai/) / Ollama) o API key compatible con OpenAI.

### Instalación y Ejecución

```powershell
# 1. Instalar dependencias y preparar entorno
uv sync
copy .env.example .env

# 2. (Opcional) Cargar documentos de demostración iniciales
uv run python scripts/download_demo_docs.py --offline --ingest

# 3. Iniciar la aplicación
uv run uvicorn app.main:app --reload
```

- **Interfaz Web:** [http://localhost:8000/](http://localhost:8000/)
- **API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker

```powershell
# Construir imagen
docker build -t technicaldoc-ai-assistant .

# Ejecutar con persistencia de datos
docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data" --env-file .env technicaldoc-ai-assistant
```
