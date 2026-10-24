from typing import Any

from app.rag.tools.base import BaseTool, ToolResult
from app.rag.tools.calculator import SafeCalculatorTool
from app.rag.tools.domain.architecture.bases import BaseConverterTool
from app.rag.tools.domain.architecture.ieee754 import IEEE754Tool
from app.rag.tools.domain.architecture.integers import IntegerRepresentationTool


class ToolRegistry:
    """Registro desacoplado y extensible de herramientas deterministas."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "description": t.description, "domain": t.domain}
            for t in self._tools.values()
        ]

    def get_tools_for_domain(self, domain: str) -> list[BaseTool]:
        return [t for t in self._tools.values() if t.domain in (domain, "universal")]

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Herramienta '{name}' no registrada en el sistema.",
            )
        return tool.run(**kwargs)


_DEFAULT_REGISTRY: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Retorna la instancia global del registro con las herramientas predeterminadas."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        reg = ToolRegistry()
        reg.register(SafeCalculatorTool())
        reg.register(BaseConverterTool())
        reg.register(IntegerRepresentationTool())
        reg.register(IEEE754Tool())
        _DEFAULT_REGISTRY = reg
    return _DEFAULT_REGISTRY
