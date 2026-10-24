from typing import Any

from app.rag.tools.base import BaseTool, ToolResult


class IntegerRepresentationTool(BaseTool):
    """Herramienta para representaciones en enteros: Signo y Magnitud, Ca1, Ca2, Exceso y ALU."""

    name = "integer_representation"
    description = (
        "Calcula la representación en Signo y Magnitud, Complemento a Uno (Ca1), "
        "Complemento a Dos (Ca2), Exceso/Sesgo y suma binaria con banderas ALU (Z, N, C, V)."
    )
    domain = "architecture"

    def run(
        self,
        value: int,
        bits: int = 8,
        bias: int | None = None,
        add_with: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if bits < 2 or bits > 64:
            return ToolResult(
                success=False, output="", error="El ancho de palabra debe estar entre 2 y 64 bits."
            )

        min_ca2 = -(1 << (bits - 1))
        max_ca2 = (1 << (bits - 1)) - 1
        min_sm = -((1 << (bits - 1)) - 1)
        max_sm = (1 << (bits - 1)) - 1
        resolved_bias = bias if bias is not None else (1 << (bits - 1))

        if value < min_ca2 or value > max_ca2:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"El valor {value} está fuera del rango representable en Ca2 "
                    f"para {bits} bits: [{min_ca2}, {max_ca2}]."
                ),
            )

        # 1. Signo y Magnitud
        sm_bin = None
        if min_sm <= value <= max_sm:
            sign_bit = "1" if value < 0 else "0"
            mag_bin = bin(abs(value))[2:].zfill(bits - 1)
            sm_bin = sign_bit + mag_bin

        # 2. Complemento a Uno
        ca1_bin = None
        if min_sm <= value <= max_sm:
            if value >= 0:
                ca1_bin = bin(value)[2:].zfill(bits)
            else:
                pos_bin = bin(abs(value))[2:].zfill(bits)
                ca1_bin = "".join("1" if b == "0" else "0" for b in pos_bin)

        # 3. Complemento a Dos
        if value >= 0:
            ca2_bin = bin(value)[2:].zfill(bits)
            ca2_steps = f"Valor no negativo: representación binaria directa en {bits} bits."
        else:
            pos_bin = bin(abs(value))[2:].zfill(bits)
            inv_bin = "".join("1" if b == "0" else "0" for b in pos_bin)
            ca2_int = (int(inv_bin, 2) + 1) & ((1 << bits) - 1)
            ca2_bin = bin(ca2_int)[2:].zfill(bits)
            ca2_steps = (
                f"1. Magnitud positiva (+{abs(value)}): {pos_bin}\n"
                f"2. Invertir bits (Ca1): {inv_bin}\n"
                f"3. Sumar 1 al LSB (Ca2): {ca2_bin}"
            )

        # 4. Desplazamiento / Exceso
        stored_excess = value + resolved_bias
        excess_bin = None
        if 0 <= stored_excess < (1 << bits):
            excess_bin = bin(stored_excess)[2:].zfill(bits)

        # 5. Operación de suma con otro operando (si se provee)
        add_result = None
        if add_with is not None:
            if add_with < min_ca2 or add_with > max_ca2:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"El segundo operando {add_with} excede el rango de {bits} bits.",
                )
            a_val = value
            b_val = add_with
            a_int = a_val if a_val >= 0 else ((1 << bits) + a_val)
            b_int = b_val if b_val >= 0 else ((1 << bits) + b_val)
            raw_sum = a_int + b_int
            sum_bits = raw_sum & ((1 << bits) - 1)
            sum_bin = bin(sum_bits)[2:].zfill(bits)

            # Reconvertir sum_bits a entero con signo en Ca2
            sum_signed = sum_bits if sum_bits <= max_ca2 else sum_bits - (1 << bits)

            # Acarreos y desbordamiento V
            c_out = 1 if raw_sum >= (1 << bits) else 0
            # Acarreo hacia el MSB (bit bits-1)
            a_low = a_int & ((1 << (bits - 1)) - 1)
            b_low = b_int & ((1 << (bits - 1)) - 1)
            c_in_msb = 1 if (a_low + b_low) >= (1 << (bits - 1)) else 0
            overflow = 1 if (c_in_msb ^ c_out) else 0

            # Banderas ALU
            flag_z = 1 if sum_bits == 0 else 0
            flag_n = int(sum_bin[0])
            flag_c = c_out
            flag_v = overflow

            add_result = {
                "a": a_val,
                "b": b_val,
                "sum_signed": sum_signed,
                "sum_bin": sum_bin,
                "flags": {"Z": flag_z, "N": flag_n, "C": flag_c, "V": flag_v},
                "c_out": c_out,
                "c_in_msb": c_in_msb,
            }

        output_lines = [
            f"Representaciones para {value} ({bits} bits, rango Ca2: [{min_ca2}, {max_ca2}]):",
            f"- Complemento a Dos (Ca2): {ca2_bin}",
            f"  Pasos: {ca2_steps}",
        ]
        if sm_bin:
            output_lines.append(f"- Signo y Magnitud: {sm_bin}")
        if ca1_bin:
            output_lines.append(f"- Complemento a Uno (Ca1): {ca1_bin}")
        if excess_bin:
            output_lines.append(
                f"- Exceso con sesgo d={resolved_bias}: {excess_bin} (almacena {stored_excess})"
            )

        if add_result:
            output_lines.append(
                f"\nSuma con {add_with}: {add_result['sum_bin']} (Ca2: {add_result['sum_signed']})"
            )
            output_lines.append(
                f"Banderas ALU: Z={add_result['flags']['Z']}, N={add_result['flags']['N']}, "
                f"C={add_result['flags']['C']}, V={add_result['flags']['V']} "
                f"(Overflow: {'SÍ' if add_result['flags']['V'] else 'NO'})"
            )

        return ToolResult(
            success=True,
            output="\n".join(output_lines),
            data={
                "value": value,
                "bits": bits,
                "range_ca2": [min_ca2, max_ca2],
                "twos_complement": ca2_bin,
                "sign_and_magnitude": sm_bin,
                "ones_complement": ca1_bin,
                "excess": excess_bin,
                "addition": add_result,
            },
        )
