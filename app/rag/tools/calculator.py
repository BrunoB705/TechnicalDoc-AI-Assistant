import ast
import operator
from typing import Any

from app.rag.tools.base import BaseTool, ToolResult

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_pow(base: float | int, exp: float | int) -> float | int:
    """Función de exponenciación protegida contra DoS por superpotencias y resultados complejos."""
    if abs(exp) > 1000 or abs(base) > 1e10:
        raise ValueError("Exponente o base demasiado grandes para cálculo seguro.")
    res = pow(base, exp)
    if isinstance(res, complex):
        raise ValueError("Resultado complejo no soportado en calculadora real.")
    return res


_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "pow": _safe_pow,
    "min": min,
    "max": max,
}


class SafeCalculatorTool(BaseTool):
    """Calculadora matemática segura universal para cualquier curso.

    Evalúa expresiones aritméticas analizando el Árbol de Sintaxis Abstracta (AST)
    sin usar `eval()` ni permitir código ejecutable o acceso a atributos.
    """

    name = "calculator"
    description = (
        "Calculadora matemática universal segura. Evalúa operaciones aritméticas "
        "básicas (+, -, *, /, //, %, **), paréntesis y funciones numéricas seguras."
    )
    domain = "universal"

    def _eval_node(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return node.value
            raise ValueError(f"Tipo constante no permitido: {type(node.value).__name__}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type not in _SAFE_OPERATORS:
                raise ValueError(f"Operador binario no permitido: {op_type.__name__}")
            if op_type is ast.Pow:
                # Prevenir ataques de denegación de servicio por exponentes gigantes
                if abs(right) > 1000 or abs(left) > 1e10:
                    raise ValueError("Exponente o base demasiado grandes para cálculo seguro.")
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("División por cero.")
            res = _SAFE_OPERATORS[op_type](left, right)
            if isinstance(res, complex):
                raise ValueError("Resultado complejo no soportado en calculadora real.")
            return res

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in _SAFE_OPERATORS:
                raise ValueError(f"Operador unario no permitido: {op_type.__name__}")
            res = _SAFE_OPERATORS[op_type](operand)
            if isinstance(res, complex):
                raise ValueError("Resultado complejo no soportado en calculadora real.")
            return res

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Llamadas complejas no permitidas.")
            func_name = node.func.id
            if func_name not in _SAFE_FUNCTIONS:
                raise ValueError(f"Función no permitida: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return _SAFE_FUNCTIONS[func_name](*args)

        raise ValueError(f"Expresión sintáctica no permitida: {type(node).__name__}")

    def run(self, expression: str, **kwargs: Any) -> ToolResult:
        clean = expression.strip()
        if not clean:
            return ToolResult(success=False, output="", error="La expresión aritmética está vacía.")
        if len(clean) > 300:
            return ToolResult(
                success=False, output="", error="La expresión supera el límite de 300 caracteres."
            )

        try:
            parsed = ast.parse(clean, mode="eval")
            result = self._eval_node(parsed.body)
            if isinstance(result, complex):
                raise ValueError("Resultado complejo no soportado en calculadora real.")
            # Redondear si es flotante con muchos decimales espurios
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return ToolResult(
                success=True,
                output=str(result),
                data={"expression": clean, "result": result},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Error al evaluar '{clean}': {exc}",
            )
