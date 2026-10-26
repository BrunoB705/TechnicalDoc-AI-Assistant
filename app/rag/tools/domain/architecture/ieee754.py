import math
from fractions import Fraction
from typing import Any

from app.rag.tools.base import BaseTool, ToolResult

FORMAT_PRESETS = {
    "binary16": {"exp_bits": 5, "bias": 15, "frac_bits": 10, "name": "Media Precisión (16 bits)"},
    "media": {"exp_bits": 5, "bias": 15, "frac_bits": 10, "name": "Media Precisión (16 bits)"},
    "binary32": {
        "exp_bits": 8,
        "bias": 127,
        "frac_bits": 23,
        "name": "Precisión Simple (32 bits)",
    },
    "simple": {"exp_bits": 8, "bias": 127, "frac_bits": 23, "name": "Precisión Simple (32 bits)"},
    "binary64": {
        "exp_bits": 11,
        "bias": 1023,
        "frac_bits": 52,
        "name": "Doble Precisión (64 bits)",
    },
    "doble": {
        "exp_bits": 11,
        "bias": 1023,
        "frac_bits": 52,
        "name": "Doble Precisión (64 bits)",
    },
}


def _scale_power_of_two(value: Fraction, exponent: int) -> Fraction:
    return value * (2**exponent) if exponent >= 0 else value / (2 ** (-exponent))


def _round_ties_to_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    comparison = 2 * remainder - value.denominator
    if comparison > 0 or (comparison == 0 and quotient % 2):
        return quotient + 1
    return quotient


class IEEE754Tool(BaseTool):
    """Calculadora y desglosadora determinista para estándar IEEE 754 de coma flotante."""

    name = "ieee754"
    description = (
        "Calcula la representación en punto flotante IEEE 754 (media 16 bits, simple 32 bits, "
        "doble 64 bits o formato parametrizado). Devuelve signo, exponente sesgado, mantisa "
        "y palabra final en binario y hexadecimal con explicación paso a paso."
    )
    domain = "architecture"

    def run(
        self,
        value: float | int | str,
        format_name: str = "binary32",
        exp_bits: int | None = None,
        bias: int | None = None,
        frac_bits: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        fmt_key = format_name.lower().strip()
        if exp_bits is not None and frac_bits is not None:
            e_bits = exp_bits
            f_bits = frac_bits
            b_val = bias if bias is not None else ((1 << (e_bits - 1)) - 1)
            fmt_desc = f"Personalizado ({1 + e_bits + f_bits} bits)"
        elif fmt_key in FORMAT_PRESETS:
            preset = FORMAT_PRESETS[fmt_key]
            e_bits = preset["exp_bits"]
            b_val = preset["bias"]
            f_bits = preset["frac_bits"]
            fmt_desc = preset["name"]
        else:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Formato '{format_name}' desconocido. Formatos disponibles: "
                    f"{', '.join(sorted(FORMAT_PRESETS.keys()))} o proveer exp_bits y frac_bits."
                ),
            )

        total_bits = 1 + e_bits + f_bits
        hex_len = (total_bits + 3) // 4

        try:
            val_float = float(value)
        except (ValueError, TypeError):
            return ToolResult(
                success=False,
                output="",
                error=f"Valor numérico no válido para IEEE 754: '{value}'",
            )

        # 1. Signo
        sign = 1 if math.copysign(1.0, val_float) < 0 else 0
        sign_bin = str(sign)

        # 2. Casos especiales: NaN
        if math.isnan(val_float):
            exp_bin = "1" * e_bits
            frac_bin = "1" + "0" * (f_bits - 1)  # Quiet NaN canónico
            full_bin = f"{sign_bin} {exp_bin} {frac_bin}"
            full_hex = hex(int(full_bin.replace(" ", ""), 2))[2:].upper().zfill(hex_len)
            return ToolResult(
                success=True,
                output=(
                    f"Formato: IEEE 754 {fmt_desc}\n"
                    f"Valor: NaN (Not a Number)\n"
                    f"- Signo: {sign_bin}\n"
                    f"- Exponente: {exp_bin} (todos 1)\n"
                    f"- Mantisa: {frac_bin} (distinta de cero)\n"
                    f"- Binario final: {full_bin}\n"
                    f"- Hexadecimal: 0x{full_hex}"
                ),
                data={
                    "format": fmt_desc,
                    "value": "NaN",
                    "sign": sign_bin,
                    "exponent_bin": exp_bin,
                    "fraction_bin": frac_bin,
                    "full_bin": full_bin,
                    "hex": f"0x{full_hex}",
                },
            )

        # 3. Casos especiales: Infinito (+Inf, -Inf)
        if math.isinf(val_float):
            exp_bin = "1" * e_bits
            frac_bin = "0" * f_bits
            full_bin = f"{sign_bin} {exp_bin} {frac_bin}"
            full_hex = hex(int(full_bin.replace(" ", ""), 2))[2:].upper().zfill(hex_len)
            inf_label = "-Infinito" if sign else "+Infinito"
            return ToolResult(
                success=True,
                output=(
                    f"Formato: IEEE 754 {fmt_desc}\n"
                    f"Valor: {inf_label}\n"
                    f"- Signo: {sign_bin}\n"
                    f"- Exponente: {exp_bin} (todos 1)\n"
                    f"- Mantisa: {frac_bin} (todos 0)\n"
                    f"- Binario final: {full_bin}\n"
                    f"- Hexadecimal: 0x{full_hex}"
                ),
                data={
                    "format": fmt_desc,
                    "value": inf_label,
                    "sign": sign_bin,
                    "exponent_bin": exp_bin,
                    "fraction_bin": frac_bin,
                    "full_bin": full_bin,
                    "hex": f"0x{full_hex}",
                },
            )

        # 4. Caso cero (+0.0, -0.0)
        if val_float == 0.0:
            exp_bin = "0" * e_bits
            frac_bin = "0" * f_bits
            full_bin = f"{sign_bin} {exp_bin} {frac_bin}"
            full_hex = hex(int(full_bin.replace(" ", ""), 2))[2:].upper().zfill(hex_len)
            return ToolResult(
                success=True,
                output=(
                    f"Formato: IEEE 754 {fmt_desc}\n"
                    f"Valor: {val_float}\n"
                    f"- Signo: {sign_bin}\n"
                    f"- Exponente: {exp_bin} (cero)\n"
                    f"- Mantisa: {frac_bin}\n"
                    f"- Binario final: {full_bin}\n"
                    f"- Hexadecimal: 0x{full_hex}"
                ),
                data={
                    "format": fmt_desc,
                    "value": val_float,
                    "sign": sign_bin,
                    "exponent_bin": exp_bin,
                    "exponent_dec": 0,
                    "bias": b_val,
                    "fraction_bin": frac_bin,
                    "full_bin": full_bin,
                    "hex": f"0x{full_hex}",
                },
            )

        abs_val = abs(val_float)
        try:
            exact_abs = abs(Fraction(str(value)))
        except (ValueError, ZeroDivisionError):
            exact_abs = Fraction.from_float(abs_val)

        # Redondeamos directamente al formato destino con roundTiesToEven.
        true_exp = exact_abs.numerator.bit_length() - exact_abs.denominator.bit_length()
        if _scale_power_of_two(exact_abs, -true_exp) < 1:
            true_exp -= 1
        biased_exp = true_exp + b_val
        max_exp = (1 << e_bits) - 1

        if biased_exp >= max_exp:
            # Desbordamiento a infinito
            exp_bin = "1" * e_bits
            combined_frac = "0" * f_bits
            norm_str = "Infinito (overflow de exponente)"
        elif biased_exp <= 0:
            # Número subnormal
            subnormal = _round_ties_to_even(_scale_power_of_two(exact_abs, b_val - 1 + f_bits))
            if subnormal >= (1 << f_bits):
                biased_exp = 1
                exp_bin = bin(biased_exp)[2:].zfill(e_bits)
                combined_frac = "0" * f_bits
                norm_str = f"1.{'0' * f_bits} * 2^{1 - b_val} (redondeado a normal)"
            else:
                biased_exp = 0
                exp_bin = "0" * e_bits
                combined_frac = bin(subnormal)[2:].zfill(f_bits)
                norm_str = f"0.{combined_frac} * 2^{1 - b_val} (Subnormal)"
        else:
            # Número normalizado
            normalized = _scale_power_of_two(exact_abs, -true_exp)
            fraction = _round_ties_to_even((normalized - 1) * (1 << f_bits))
            if fraction >= (1 << f_bits):
                fraction = 0
                true_exp += 1
                biased_exp += 1
            if biased_exp >= max_exp:
                exp_bin = "1" * e_bits
                combined_frac = "0" * f_bits
                norm_str = "Infinito (overflow tras redondeo)"
            else:
                exp_bin = bin(biased_exp)[2:].zfill(e_bits)
                combined_frac = bin(fraction)[2:].zfill(f_bits)
                norm_str = f"1.{combined_frac} * 2^{true_exp}"

        full_raw_bin = f"{sign_bin}{exp_bin}{combined_frac}"
        full_bin = f"{sign_bin} {exp_bin} {combined_frac}"
        full_hex = hex(int(full_raw_bin, 2))[2:].upper().zfill(hex_len)

        steps = [
            f"Paso 1: Signo = {sign_bin} ({'negativo' if sign else 'positivo'}).",
            f"Paso 2: Valor absoluto {abs_val} normalizado: {norm_str}.",
            f"Paso 3: Exponente no sesgado = {true_exp}. Con sesgo (bias={b_val}): "
            f"{true_exp} + {b_val} = {biased_exp} -> {exp_bin}_2 ({e_bits} bits).",
            f"Paso 4: Mantisa fraccionaria (sin bit implícito) -> {combined_frac} ({f_bits} bits).",
            f"Paso 5: Ensamblado final [S: {sign_bin}] [E: {exp_bin}] [M: {combined_frac}].",
        ]

        output = (
            f"Formato: IEEE 754 {fmt_desc}\n"
            f"Valor: {val_float}\n\n"
            + "\n".join(steps)
            + f"\n\nResultado Binario: {full_bin}\nResultado Hexadecimal: 0x{full_hex}"
        )

        return ToolResult(
            success=True,
            output=output,
            data={
                "format": fmt_desc,
                "value": val_float,
                "sign": sign_bin,
                "true_exponent": true_exp,
                "biased_exponent": biased_exp,
                "bias": b_val,
                "exponent_bin": exp_bin,
                "fraction_bin": combined_frac,
                "full_bin": full_bin,
                "hex": f"0x{full_hex}",
            },
        )
