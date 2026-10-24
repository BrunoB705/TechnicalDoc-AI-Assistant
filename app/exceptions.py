class RAGError(Exception):
    """Error base del dominio."""


class LLMUnavailableError(RAGError):
    """El modelo de lenguaje no responde (apagado, timeout, conexión rechazada)."""


class EmptyIndexError(RAGError):
    """No hay documentos indexados."""


class DocumentNotFoundError(RAGError):
    """El documento no existe en el índice."""


class DuplicateDocumentError(RAGError):
    """Ya existe un documento indexado con el mismo contenido."""


class InvalidDocumentError(RAGError):
    """El archivo subido no es válido (extensión, nombre o sin texto extraíble)."""


class DocumentTooLargeError(RAGError):
    """El archivo supera el tamaño máximo permitido."""


class KnowledgeBaseNotFoundError(RAGError):
    """La base de conocimiento solicitada no existe."""


class DuplicateKnowledgeBaseError(RAGError):
    """Ya existe una base de conocimiento con ese identificador o nombre."""


class InvalidKnowledgeBaseError(RAGError):
    """Operación o datos inválidos sobre una base de conocimiento."""


class ReadOnlyModeError(RAGError):
    """Operación de escritura bloqueada porque READ_ONLY_MODE está activo."""


class RateLimitExceededError(RAGError):
    """Se superó el límite de consultas por minuto permitido."""
