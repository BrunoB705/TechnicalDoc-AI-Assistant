# TechnicalDoc AI Assistant

Asistente RAG para documentación técnica (Markdown, TXT, PDF) con FastAPI, LangChain, ChromaDB y LLMs (OpenAI o modelos locales mediante LM Studio / Ollama).

---

## 📌 Estado del Proyecto: Fase 7 (Evaluación Cuantitativa RAG + Multi-Knowledge Base + Robustez + Frontend web)

- **Ingesta:** Soporta archivos `.md`, `.txt` y `.pdf`, extrayendo metadatos de secciones jerárquicas y números de página.
- **Embeddings:** `intfloat/multilingual-e5-small` ejecutado localmente con HuggingFace (sin consumo de API).
- **Vectorstore:** ChromaDB persistente en `data/chroma/`.
- **LLM:** Compatible con modelos locales (LM Studio, Ollama) y APIs en la nube (OpenAI, Groq, DeepSeek) mediante abstracción intercambiable.
- **Confiabilidad:** Validación y filtrado automático de citas (`[n]`), eliminación de fuentes alucinadas y rechazo por falta de evidencia (`NO_EVIDENCE` y umbral semántico).
- **Streaming:** Respuestas token por token en tiempo real mediante Server-Sent Events (`POST /query/stream`).
- **Frontend web:** Interfaz en `http://localhost:8000/` (HTML/CSS/JS sin frameworks) con chat en streaming, citas `[n]` clicables, fuentes y panel de documentos (subir/listar/borrar).
- **Evaluación cuantitativa y calibración:** Dataset de 45 casos (`eval/dataset.jsonl`), corpus reproducible multi-KB (`eval/corpus/`) y runner `scripts/evaluate.py` con métricas Recall@k, MRR, Refusal Accuracy, Aislamiento entre KBs y barrido de hiperparámetros (`--calibrate`).
- **Multi-Knowledge Base:** Catálogo persistente en SQLite (`data/catalog.sqlite3`), aislamiento estricto de ámbito por `knowledge_base_id`, migración automática a la base `general` y selector de base en la interfaz.
- **Robustez de ingesta:** Upload leído en bloques de 64 KB con corte temprano (`413`), rollback atómico si la indexación falla, indexado por lotes y contrato SSE unificado (`end` siempre presente).
- **Gestión de documentos:** Alta (upload), listado, detalle y baja de documentos vía API, con detección de duplicados por contenido.

---

## 🚀 Inicio Rápido

### 1. Requisitos
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (gestor de paquetes y entorno virtual)
- [LM Studio](https://lmstudio.ai/) / Ollama con servidor local activo, o una API key compatible (OpenAI, Groq, etc.).

### 2. Instalación y Configuración
```powershell
# Clonar e instalar dependencias
uv sync

# Crear el archivo de entorno
copy .env.example .env
```

Configuración en `.env`:

**Opción A: Modelo local con LM Studio**
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=tu-modelo-local
EMBEDDING_MODEL=intfloat/multilingual-e5-small
CHROMA_DIR=data/chroma
TOP_K=4
MIN_RELEVANCE=0.35
LLM_TIMEOUT=120.0
```

**Opción B: En la nube con OpenAI**
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-tu-api-key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=intfloat/multilingual-e5-small
CHROMA_DIR=data/chroma
TOP_K=4
MIN_RELEVANCE=0.35
LLM_TIMEOUT=120.0
```

### 3. Ingesta de Documentos
Copiá tus documentos a la carpeta `data/docs/` y ejecutá:
```powershell
uv run python scripts/ingest.py data/docs
```
*(Para reiniciar la base de vectores desde cero: `uv run python scripts/ingest.py data/docs --reset`)*.

### 4. Ejecutar el Servidor
```powershell
uv run uvicorn app.main:app --reload
```
- Swagger UI interactivo: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interfaz web: [http://localhost:8000/](http://localhost:8000/)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 Endpoints Principales

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Chequeo de salud del servicio (`{"status": "ok"}`). |
| `POST` | `/query` | Consulta síncrona con respuesta completa, fuentes citadas y latencia. |
| `POST` | `/query/stream` | Consulta en streaming (SSE) con tokens emitidos en tiempo real. |
| `POST` | `/documents` | Sube un documento (`multipart/form-data`, campo `file`) y lo indexa. `201`; `409` duplicado; `413` muy grande; `422` inválido. |
| `GET` | `/documents` | Lista los documentos indexados con cantidad de chunks. |
| `GET` | `/documents/{doc_id}` | Detalle de un documento (`404` si no existe). |
| `DELETE` | `/documents/{doc_id}` | Borra el documento del índice y de `data/docs/` (`204`). |

### Ejemplo de consulta síncrona (`POST /query`)
```json
{
  "question": "¿Cuáles son los requisitos de despliegue del servicio?",
  "top_k": 4
}
```

---

## 🧪 Pruebas Automatizadas

El proyecto cuenta con una suite completa de pruebas unitarias y de integración que no requieren conexión a internet ni consumo de LLMs:
```powershell
uv run pytest -q -W ignore
```
Para verificación de estilo y linter:
```powershell
uv run ruff check .
```

---

## 🗺️ Hoja de Ruta (Roadmap)
- [x] **Fase 0**: Scaffold inicial con `uv` y typing estricto.
- [x] **Fase 1**: MVP Vertical Slice (Ingesta, ChromaDB, FastAPI `POST /query`).
- [x] **Fase 2**: Confiabilidad (Validación de citas, eliminación de citas falsas, manejo de errores `503`/`409`).
- [x] **Fase 2.5**: Streaming en tiempo real vía Server-Sent Events (SSE).
- [x] **Fase 3**: Gestión de documentos vía API (`POST/GET/DELETE /documents`).
- [x] **Fase 4**: Frontend interactivo propio servido en FastAPI.
- [x] **Fase 5**: Robustez de ingesta (upload en streaming, rollback atómico, indexado por lotes, contrato SSE unificado).
- [x] **Fase 6**: Soporte multi-Knowledge Base (aislamiento y catálogo).
- [x] **Fase 7**: Evaluación cuantitativa RAG (Recall@k, MRR, abstención, calibración de umbrales).
- [ ] **Fase 8**: Reproducibilidad, CI/CD y contenerización (Docker, GitHub Actions, corpus demo).
- [ ] **Fase 9**: Hardening para demo pública y portfolio.
