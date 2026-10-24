from typing import Any

from app.rag.tools.base import BaseTool, ToolResult


class BaseConverterTool(BaseTool):
    """Herramienta para conversión exacta entre bases numéricas (2, 8, 10, 16)."""

    name = "base_converter"
    description = (
        "Convierte números enteros entre decimal, binario, hexadecimal y octal, "
        "con formateo opcional en nibbles y longitud fija de bits."
    )
    domain = "architecture"

    @staticmethod
    def _format_nibbles(bin_str: str) -> str:
        # Agrupar en bloques de 4 bits de derecha a izquierda
        pad_len = (4 - len(bin_str) % 4) % 4
        padded = ("0" * pad_len) + bin_str
        return " ".join(padded[i : i + 4] for i in range(0, len(padded), 4))

    def run(
        self,
        value: int | str,
        from_base: int = 10,
        bits: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            val_int = int(str(value).strip(), from_base)
        except ValueError:
            return ToolResult(
                success=False,
                output="",
                error=f"No se pudo interpretar el valor '{value}' en base {from_base}.",
            )

        is_neg = val_int < 0
        abs_val = abs(val_int)
        raw_bin = bin(abs_val)[2:]

        if bits is not None:
            if bits < 1 or bits > 128:
                return ToolResult(
                    success=False,
                    output="",
                    error="La cantidad de bits debe estar entre 1 y 128.",
                )
            if abs_val >= (1 << bits):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"El valor {val_int} excede la capacidad de {bits} bits.",
                )
            raw_bin = raw_bin.zfill(bits)

        hex_str = hex(abs_val)[2:].upper()
        if bits is not None:
            hex_str = hex_str.zfill((bits + 3) // 4)

        oct_str = oct(abs_val)[2:]

        sign_str = "-" if is_neg else ""
        bin_formatted = f"{sign_str}{self._format_nibbles(raw_bin)}"

        output_str = (
            f"Decimal: {val_int}\n"
            f"Binario: {bin_formatted}\n"
            f"Hexadecimal: {sign_str}0x{hex_str}\n"
            f"Octal: {sign_str}0o{oct_str}"
        )

        return ToolResult(
            success=True,
            output=output_str,
            data={
                "decimal": val_int,
                "binary": f"{sign_str}{raw_bin}",
                "binary_nibbles": bin_formatted,
                "hexadecimal": f"{sign_str}0x{hex_str}",
                "octal": f"{sign_str}0o{oct_str}",
                "bits": len(raw_bin),
            },
        )
