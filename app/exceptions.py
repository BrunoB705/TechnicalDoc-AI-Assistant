class RAGError(Exception):
    """Error base del dominio."""


class LLMUnavailableError(RAGError):
    """El modelo de lenguaje no responde (apagado, timeout, conexión rechazada)."""


class EmptyIndexError(RAGError):
    """No hay documentos indexados."""


class DocumentNotFoundError(RAGError):
    """El documento no existe en el ?ndice."""


class DuplicateDocumentError(RAGError):
    """Ya existe un documento indexado con el mismo contenido."""


class InvalidDocumentError(RAGError):
    """El archivo subido no es v?lido (extensi?n, nombre o sin texto extra?ble)."""


class DocumentTooLargeError(RAGError):
    """El archivo supera el tama?o m?ximo permitido."""
