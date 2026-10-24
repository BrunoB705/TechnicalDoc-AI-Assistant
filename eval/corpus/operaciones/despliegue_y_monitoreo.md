# Guía de Operación, Despliegue y Monitoreo en Producción

## Requisitos del Entorno de Ejecución
Para desplegar el asistente en producción se requiere:
- **Python 3.12** o superior gestionado mediante **`uv`**.
- Mínimo **2 GB de memoria RAM** disponible para el proceso de inferencia de embeddings (`sentence-transformers`) y el índice HNSW de ChromaDB.
- Almacenamiento persistente montado en el directorio `data/` para conservar `data/chroma/`, `data/docs/` y `data/catalog.sqlite3`.

## Variables de Entorno Críticas
- `LLM_PROVIDER`: proveedor del modelo generativo (`openai` compatible con LM Studio, u `ollama`).
- `OPENAI_BASE_URL`: URL del servidor de inferencia local, habitualmente `http://127.0.0.1:1234/v1` para LM Studio.
- `MIN_RELEVANCE`: umbral mínimo de similitud coseno para aceptar un fragmento recuperado (valor calibrado por defecto: `0.35`).
- `MAX_UPLOAD_MB`: tamaño máximo permitido por archivo subido (por defecto `20` MB).

## Invalidación de Caché y Mantenimiento
- El caché de configuración en memoria (`@lru_cache` en `get_settings`) se invalida automáticamente cada **10 minutos** en el proxy o reiniciando el proceso `uvicorn`.
- Para reconstruir el índice vectorial desde cero sobre una base específica se ejecuta el comando:
  `uv run python scripts/ingest.py <directorio> --kb <kb_id> --reset`.

## Observabilidad y Registro Estructurado
Cada consulta registra un evento JSON estructurado en el logger `"rag"` con las claves `event`, `knowledge_base_id`, `question_len`, `retrieval_ms`, `generation_ms`, `found`, `above_threshold`, `cited` y `outcome`.
