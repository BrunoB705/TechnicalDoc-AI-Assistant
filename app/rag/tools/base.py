from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Resultado estructurado de la ejecución de una herramienta determinista."""

    success: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseTool(ABC):
    """Interfaz abstracta para herramientas deterministas del asistente."""

    name: str
    description: str
    domain: str = "universal"

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Ejecuta la herramienta con validación estricta de parámetros."""
        raise NotImplementedError
