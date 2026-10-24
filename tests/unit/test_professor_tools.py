
from app.rag.tools.calculator import SafeCalculatorTool
from app.rag.tools.domain.architecture.bases import BaseConverterTool
from app.rag.tools.domain.architecture.ieee754 import IEEE754Tool
from app.rag.tools.domain.architecture.integers import IntegerRepresentationTool
from app.rag.tools.registry import get_default_registry


def test_safe_calculator_basic_operations():
    calc = SafeCalculatorTool()
    res = calc.run("2 + 2")
    assert res.success is True
    assert res.output == "4"

    res = calc.run("(10 * 5) - 3")
    assert res.success is True
    assert res.output == "47"

    res = calc.run("2 ** 8")
    assert res.success is True
    assert res.output == "256"

    res = calc.run("73 % 16")
    assert res.success is True
    assert res.output == "9"


def test_safe_calculator_errors_and_security():
    calc = SafeCalculatorTool()
    # División por cero
    res = calc.run("10 / 0")
    assert res.success is False
    assert "cero" in (res.error or "").lower()

    # Intento de código inseguro o llamada no permitida
    res = calc.run("__import__('os').system('ls')")
    assert res.success is False

    # Expresión vacía
    res = calc.run("   ")
    assert res.success is False


def test_base_converter():
    conv = BaseConverterTool()
    # Conversión del número 137
    res = conv.run(137, bits=8)
    assert res.success is True
    assert res.data["decimal"] == 137
    assert res.data["binary"] == "10001001"
    assert "1000 1001" in res.data["binary_nibbles"]
    assert res.data["hexadecimal"] == "0x89"

    # Número negativo
    res_neg = conv.run(-18)
    assert res_neg.success is True
    assert res_neg.data["decimal"] == -18
    assert res_neg.data["hexadecimal"] == "-0x12"


def test_integer_representation_ca2():
    tool = IntegerRepresentationTool()

    # -18 en 8 bits (ejercicio de examen)
    res = tool.run(value=-18, bits=8)
    assert res.success is True
    assert res.data["twos_complement"] == "11101110"
    assert res.data["ones_complement"] == "11101101"
    assert res.data["sign_and_magnitude"] == "10010010"

    # Casos límite de 8 bits
    res_min = tool.run(value=-128, bits=8)
    assert res_min.success is True
    assert res_min.data["twos_complement"] == "10000000"

    res_max = tool.run(value=127, bits=8)
    assert res_max.success is True
    assert res_max.data["twos_complement"] == "01111111"

    # Fuera de rango
    res_overflow = tool.run(value=200, bits=8)
    assert res_overflow.success is False


def test_integer_addition_with_alu_overflow():
    tool = IntegerRepresentationTool()
    # Suma de 4 bits: +5 (0101) + +4 (0100) = +9 (desborda a 1001 = -7 en Ca2)
    res = tool.run(value=5, bits=4, add_with=4)
    assert res.success is True
    addition = res.data["addition"]
    assert addition["sum_bin"] == "1001"
    assert addition["sum_signed"] == -7
    assert addition["flags"]["V"] == 1  # Overflow activo
    assert addition["flags"]["N"] == 1  # Negativo
    assert addition["flags"]["Z"] == 0


def test_ieee754_half_precision_137():
    """Verifica exactamente el caso del brief: 137 en Media Precisión (binary16)."""
    tool = IEEE754Tool()
    res = tool.run(137, format_name="binary16")
    assert res.success is True
    d = res.data

    assert d["sign"] == "0"
    assert d["true_exponent"] == 7
    assert d["biased_exponent"] == 22  # 7 + 15 = 22
    assert d["exponent_bin"] == "10110"
    assert d["fraction_bin"] == "0001001000"
    assert d["full_bin"] == "0 10110 0001001000"
    assert d["hex"] == "0x5848"


def test_ieee754_single_precision_137():
    """Verifica 137 en Precisión Simple (binary32)."""
    tool = IEEE754Tool()
    res = tool.run(137, format_name="binary32")
    assert res.success is True
    d = res.data

    assert d["sign"] == "0"
    assert d["true_exponent"] == 7
    assert d["biased_exponent"] == 134  # 7 + 127 = 134
    assert d["exponent_bin"] == "10000110"
    assert d["fraction_bin"].startswith("0001001")
    assert d["hex"] == "0x43090000"


def test_ieee754_negative_float():
    tool = IEEE754Tool()
    res = tool.run(-18.5, format_name="binary32")
    assert res.success is True
    d = res.data
    assert d["sign"] == "1"
    assert d["exponent_bin"] == "10000011"  # 4 + 127 = 131
    assert d["fraction_bin"].startswith("00101")


def test_tool_registry():
    reg = get_default_registry()
    tools = reg.list_tools()
    names = {t["name"] for t in tools}
    assert {"calculator", "base_converter", "integer_representation", "ieee754"}.issubset(names)

    # Filtrado por dominio
    arch_tools = [t.name for t in reg.get_tools_for_domain("architecture")]
    assert "calculator" in arch_tools  # universal está incluido
    assert "ieee754" in arch_tools

    # Ejecución a través del registro
    exec_res = reg.execute("calculator", expression="15 * 3")
    assert exec_res.success is True
    assert exec_res.output == "45"

    unknown = reg.execute("non_existent_tool")
    assert unknown.success is False
