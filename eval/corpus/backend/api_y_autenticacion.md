# Diseño de API y Autenticación del Servicio Backend

## Contratos HTTP y Versionado
El servicio backend expone endpoints REST sobre FastAPI en el puerto por defecto **8000**. Todas las respuestas de error utilizan el esquema JSON canónico `{"error": "<codigo_snake_case>", "detail": "<mensaje>"}`.

## Límites de Consulta y Validación
- La longitud de la pregunta (`question`) debe tener como mínimo **3 caracteres** y como máximo **1000 caracteres**.
- El parámetro `top_k` acepta valores enteros entre **1 y 10**, con un valor por defecto de **4 fragmentos**.
- El tiempo máximo de espera del modelo de lenguaje (`LLM_TIMEOUT`) es de **120 segundos**; si se supera, el backend responde con código HTTP **503 Service Unavailable** (`llm_unavailable`).

## Streaming por Server-Sent Events (SSE)
El endpoint `POST /query/stream` transmite eventos con `Content-Type: text/event-stream`.
La secuencia de eventos garantizada es:
1. `event: retrieved`: contiene los fragmentos recuperados y el tiempo `retrieval_ms`.
2. `event: token`: emite cada fragmento de texto generado por el LLM en tiempo real.
3. `event: end`: evento de cierre obligatorio que incluye `grounded`, `reason`, `message`, `answer` (con citas inválidas saneadas), `sources` y `cited_ids`.

## Reglas de Citación y Abstención
- Toda afirmación generada por el asistente debe incluir citas en formato `[n]` referenciando el índice de la fuente.
- Si el modelo responde sin citas, el servicio realiza **exactamente un reintento** anexando el recordatorio `CITATION_REMINDER`.
- Si el modelo emite el marcador `NO_EVIDENCE`, el sistema se abstiene con `reason="model_refused"`.
