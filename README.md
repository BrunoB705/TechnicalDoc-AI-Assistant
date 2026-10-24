# TechnicalDoc AI Assistant

Asistente RAG para documentación técnica (`.md`, `.txt`, `.pdf`) con aislamiento multi-Knowledge Base, validación determinista de citas, streaming en tiempo real (SSE) y evaluación cuantitativa reproducible. Construido con **FastAPI**, **LangChain**, **ChromaDB**, **SQLite** y modelos de embeddings locales (`intfloat/multilingual-e5-small`).

---

## 🚀 Inicio Rápido

### 1. Requisitos
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (gestor de paquetes y entorno virtual)
- [LM Studio](https://lmstudio.ai/) / Ollama con servidor local activo, o una API key compatible con OpenAI.

### 2. Instalación y Configuración
```powershell
# Instalar dependencias (incluyendo grupo dev)
uv sync --all-groups

# Crear el archivo de entorno
copy .env.example .env
```

Variables principales en `.env` (sincronizadas con los valores por defecto de [`app/config.py`](app/config.py)):
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=qwen/qwen3.5-9b
LLM_TIMEOUT=120.0
MAX_OUTPUT_TOKENS=1024
EMBEDDING_MODEL=intfloat/multilingual-e5-small
CHROMA_DIR=data/chroma
DOCS_DIR=data/docs
CATALOG_PATH=data/catalog.sqlite3
CHUNK_SIZE=500
CHUNK_OVERLAP=75
TOP_K=4
MIN_RELEVANCE=0.80
READ_ONLY_MODE=false
RATE_LIMIT_PER_MINUTE=60
```

### 3. Cargar el Corpus Público de Demostración (Opcional)
Para poblar el sistema rápidamente sin requerir conexión a internet (crea e indexa las bases `python-peps` y `fastapi-guide`, además de las bases técnicas `arquitectura`, `backend` y `operaciones` incluidas en `eval/corpus/`):
```powershell
uv run python scripts/download_demo_docs.py --offline --ingest
```
Si disponés de conexión a internet y querés descargar los textos completos desde los repositorios oficiales de Python PEPs y FastAPI (con fallback automático al corpus embebido si falla la red):
```powershell
uv run python scripts/download_demo_docs.py --ingest
```

También podés ingestar cualquier carpeta local hacia una Knowledge Base específica:
```powershell
uv run python scripts/ingest.py data/docs --kb general --kb-name "General"
```

> **Limitaciones de formato y archivos ignorados por Git:**
> - **Soporte de PDF (`pypdf`):** El extractor procesa únicamente archivos PDF con **capa de texto digital seleccionable**. Los PDFs escaneados como imagen (sin OCR previo) no contienen texto extraíble y son rechazados de forma segura con HTTP `422 Unprocessable Entity` (`invalid_document`).
> - **Persistencia local (`.gitignore`):** Los directorios `data/chroma/`, `data/docs/*`, `data/demo_docs/`, `data/catalog.sqlite3` y `eval/reports/` se generan localmente y están excluidos del control de versiones. Los datasets reproducibles (`eval/dataset.jsonl`, `eval/test_dataset.jsonl` y `eval/corpus/`) sí están versionados en el repositorio.

### 4. Ejecutar el Servidor
```powershell
uv run uvicorn app.main:app --reload
```
- **Interfaz Web Multi-KB:** [http://localhost:8000/](http://localhost:8000/)
- **Swagger UI Interactivo:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Liveness / Readiness:** [http://localhost:8000/health/live](http://localhost:8000/health/live) · [http://localhost:8000/health/ready](http://localhost:8000/health/ready)

---

## 🐳 Despliegue con Docker y Modo Demo Pública (`READ_ONLY_MODE`)

El repositorio incluye un `Dockerfile` multi-stage optimizado con `uv`:

```powershell
# Construir la imagen
docker build -t technicaldoc-ai-assistant:1.0.0 .

# Ejecutar en modo normal con persistencia en volumen local
docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data" --env-file .env technicaldoc-ai-assistant:1.0.0

# Ejecutar como Demo Pública en modo solo lectura (bloquea subidas y borrados con HTTP 403)
docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data" -e READ_ONLY_MODE=true -e RATE_LIMIT_PER_MINUTE=30 --env-file .env technicaldoc-ai-assistant:1.0.0
```

> **Conexión desde Docker hacia un LLM corriendo en el host (LM Studio / Ollama):**
> Dentro de un contenedor Docker, `localhost` apunta al propio contenedor y no a tu máquina anfitriona. En **Docker Desktop (Windows / macOS)** (o en Linux agregando `--add-host=host.docker.internal:host-gateway`), configurá la URL del proveedor usando `host.docker.internal`:
> - **LM Studio:** `-e OPENAI_BASE_URL=http://host.docker.internal:1234/v1` (asegurándote de activar *"Serve on Local Network"* en LM Studio).
> - **Ollama:** `-e OLLAMA_BASE_URL=http://host.docker.internal:11434`.

---

## 📡 Endpoints Principales de la API

### Salud y Observabilidad
| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Chequeo rápido de salud (`{"status": "ok"}`). |
| `GET` | `/health/live` | Liveness probe: versión del servicio y estado de `read_only_mode`. |
| `GET` | `/health/ready` | Readiness probe: verifica conectividad con el catálogo SQLite y ChromaDB (`200` o `503`). |

### Catálogo de Knowledge Bases
| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/knowledge-bases` | Lista todas las Knowledge Bases con conteo de documentos y chunks. |
| `POST` | `/knowledge-bases` | Crea una nueva KB (`201 Created`, `409` duplicada, `403` en modo solo lectura). |
| `GET` | `/knowledge-bases/{kb_id}` | Obtiene el detalle de una Knowledge Base (`404` si no existe). |
| `DELETE` | `/knowledge-bases/{kb_id}` | Elimina una KB, sus archivos y sus vectores (`204`, `422` si es `general`, `403` si solo lectura). |

### Gestión de Documentos (por Knowledge Base y ruta legada `/documents`)
| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/knowledge-bases/{kb_id}/documents` | Sube un archivo (`.md`, `.txt`, `.pdf`) por streaming y lo indexa en `kb_id` (`201`, `403`, `409`, `413`, `422`). |
| `GET` | `/knowledge-bases/{kb_id}/documents` | Lista los documentos indexados en `kb_id`. |
| `GET` | `/knowledge-bases/{kb_id}/documents/{doc_id}` | Detalle de un documento específico en `kb_id`. |
| `DELETE` | `/knowledge-bases/{kb_id}/documents/{doc_id}` | Elimina el documento del disco y sus vectores de ChromaDB (`204`, `403` en solo lectura). |

### Consulta RAG (Contrato Unificado Síncrono y Streaming)
| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/query` | Consulta síncrona en el ámbito de `knowledge_base_id` con validación de citas, reintento guiado y latencia (`200`, `404`, `409`, `429`, `503`). |
| `POST` | `/query/stream` | Consulta en streaming SSE (`retrieved` → `token`* → `end`) con la misma política de validación, reintento de citas y aislamiento por KB. |

Ejemplo de `POST /query`:
```json
{
  "question": "¿Cuántos niveles de memoria caché implementan los procesadores modernos?",
  "knowledge_base_id": "arquitectura",
  "top_k": 4
}
```

---

## 🧪 Pruebas Automatizadas y Calidad de Código

La suite cuenta con **71 pruebas unitarias e integradas** que verifican configuración, reutilización de embeddings, ingesta transaccional, aislamiento entre Knowledge Bases, paridad `/query` vs `/query/stream`, evaluación cuantitativa (dev + test set independiente), reproducibilidad y protecciones (`READ_ONLY_MODE` y rate limiting) sin requerir conexión a internet ni LLMs externos:

```powershell
# Ejecutar toda la suite de tests
uv run pytest -q -W ignore

# Verificar estilo y linting con Ruff
uv run ruff check .
```

---

## 🗺️ Hoja de Ruta (Roadmap Completado)
- [x] **Fase 0**: Scaffold inicial con `uv` y typing estricto.
- [x] **Fase 1**: MVP Vertical Slice (Ingesta, ChromaDB, FastAPI `POST /query`).
- [x] **Fase 2**: Confiabilidad (Validación de citas, eliminación de citas falsas, manejo de errores `503`/`409`).
- [x] **Fase 2.5**: Streaming en tiempo real vía Server-Sent Events (SSE).
- [x] **Fase 3**: Gestión de documentos vía API (`POST/GET/DELETE /documents`).
- [x] **Fase 4**: Frontend interactivo propio servido en FastAPI.
- [x] **Fase 5**: Robustez de ingesta (upload en streaming, rollback atómico, indexado por lotes, contrato SSE unificado).
- [x] **Fase 6**: Soporte multi-Knowledge Base (aislamiento estricto y catálogo SQLite).
- [x] **Fase 7**: Evaluación cuantitativa RAG (Recall@k, MRR, abstención, aislamiento y calibración de umbrales).
- [x] **Fase 8**: Reproducibilidad, CI/CD y contenerización (Docker, GitHub Actions, corpus demo).
- [x] **Fase 9**: Hardening para demo pública (`READ_ONLY_MODE`, rate limiting, `/health/live`, `/health/ready` y documentación).

---

## 📊 Evaluación Cuantitativa y Calibración RAG

El repositorio incluye un framework reproducible de evaluación ([`scripts/evaluate.py`](scripts/evaluate.py) y [`app/evaluation/`](app/evaluation/)) sobre el corpus multi-KB (`eval/corpus/`: bases `arquitectura`, `backend` y `operaciones`), separando dos conjuntos disjuntos:
1. **Conjunto de Desarrollo / Calibración (`eval/dataset.jsonl`, 45 casos):** 30 consultas positivas con evidencia, 8 negativas fuera de dominio y 7 consultas cruzadas entre bases (*cross-KB*).
2. **Conjunto de Prueba Independiente / Held-Out (`eval/test_dataset.jsonl`, 15 casos):** 10 consultas positivas, 3 negativas y 2 *cross-KB* no utilizadas para seleccionar hiperparámetros.

### Resultados de Recuperación y Aislamiento (`intfloat/multilingual-e5-small`)

*(Métricas deterministas de recuperación vectorial y umbral sin LLM, con `chunk_size=500`, `top_k=3..4`, `min_relevance=0.837`)*

| Métrica | Conjunto Dev (`45` casos) | Conjunto Held-Out Test (`15` casos) | Descripción |
|---|---|---|---|
| **Recall@k** | **100.00%** (30/30) | **100.00%** (10/10) | Proporción de consultas positivas donde el documento esperado aparece en el Top-K recuperado. |
| **MRR** (*Mean Reciprocal Rank*) | **1.0000** | **1.0000** | El documento fuente correcto se posiciona en el rango `#1` en todos los casos positivos. |
| **Refusal Accuracy** | **100.00%** (15/15) | **100.00%** (5/5) | Abstención exacta en consultas sin evidencia (negativas puras y consultas *cross-KB*). |
| **Aislamiento entre KBs** | **100.00%** (45/45) | **100.00%** (15/15) | Cero fugas de fragmentos (`0` chunks contaminados) entre bases de conocimiento distintas. |
| **Latencia de Recuperación** | **26.73 ms** (p95: **36.70 ms**) | **25.40 ms** | Búsqueda vectorial local + filtrado obligatorio por `knowledge_base_id` en ChromaDB. |

> **Nota sobre calibración de `multilingual-e5-small` y política de abstención:**
> Debido a la anisotropía del espacio coseno en modelos de la familia E5, las similitudes nunca caen cerca de `0.0`: las consultas fuera de dominio puntúan `0.71–0.79`, las consultas técnicas cruzadas hacia otra KB puntúan `0.81–0.835` y las consultas con evidencia real puntúan `0.839–0.94`.
> - **Modo solo umbral vectorial (sin LLM):** `min_relevance=0.837` separa por sí solo el 100% de las consultas *cross-KB* tanto en el conjunto de desarrollo como en el *held-out test set*.
> - **Modo aplicación (con LLM activo):** `MIN_RELEVANCE=0.80` (valor por defecto en `app/config.py` y `.env.example`) ofrece mayor margen de sensibilidad para consultas parafraseadas de usuario y delega el rechazo fino de casos borde (`0.80–0.837`) a la regla de abstención `NO_EVIDENCE` del LLM.

Para reproducir la evaluación (genera `eval/reports/calibration_report.md`, `eval/reports/latest_report.md` y `eval/reports/latest.json`):
```powershell
# Evaluación determinista de recuperación y aislamiento (Dev + Held-Out Test Set)
uv run python scripts/evaluate.py

# Barrido completo de calibración (chunk_size x top_k x min_relevance)
uv run python scripts/evaluate.py --calibrate

# Evaluación end-to-end incluyendo generación con el LLM configurado y validez de citas [n]
uv run python scripts/evaluate.py --with-llm
```
