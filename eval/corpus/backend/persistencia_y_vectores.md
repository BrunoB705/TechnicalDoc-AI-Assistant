# Persistencia Vectorial, Catálogo SQLite e Ingesta Transaccional

## Modelo de Embeddings y Normalización
El sistema utiliza el modelo multilingüe **`intfloat/multilingual-e5-small`** con vectores normalizados y métrica de similitud **coseno** (`hnsw:space = cosine`) en ChromaDB.
Los textos indexados llevan el prefijo obligatorio `"passage: "` y las consultas llevan el prefijo `"query: "`.

## Identificadores Deterministas y Deduplicación
- Cada documento subido se identifica mediante `doc_id`, calculado como los primeros **16 caracteres hexadecimales del hash SHA-256** del contenido binario del archivo.
- El tamaño de fragmento por defecto (`CHUNK_SIZE`) es de **1000 caracteres** con un solapamiento (`CHUNK_OVERLAP`) de **150 caracteres**.
- La ingesta en ChromaDB se ejecuta en lotes de **1000 fragmentos** (`INDEX_BATCH_SIZE = 1000`) para respetar el límite interno de lote de ChromaDB (5461 elementos).

## Upload en Streaming y Rollback Atómico
- Los archivos subidos se leen en bloques de **64 KB** (`UPLOAD_CHUNK_SIZE = 65536` bytes), abortando inmediatamente con código **413 Content Too Large** si se supera `MAX_UPLOAD_MB` (por defecto **20 MB**).
- Si ocurre una falla durante la extracción o indexación, el método `_rollback` elimina los vectores asociados al `doc_id` en esa Knowledge Base y restaura la versión previa del archivo en disco.

## Catálogo de Knowledge Bases
El catálogo se almacena en SQLite en la ruta `data/catalog.sqlite3`. La base por defecto tiene identificador `"general"` y no puede ser eliminada (`422 invalid_knowledge_base`).
