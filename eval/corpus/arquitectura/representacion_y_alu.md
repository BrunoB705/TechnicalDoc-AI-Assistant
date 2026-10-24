# Representación Numérica y Unidad Aritmético-Lógica (ALU)

## Sistemas de Representación en Enteros
- **Binario sin signo**: Representa naturales en el rango $[0, 2^n - 1]$ para una palabra de $n$ bits.
- **Signo y Magnitud**: El bit más significativo (MSB) indica el signo (0 positivo, 1 negativo) y los $n-1$ bits restantes la magnitud. Presenta doble representación para el cero ($+0$ y $-0$).
- **Complemento a Dos (Ca2)**: Rango asimétrico $[-2^{n-1}, 2^{n-1} - 1]$. Para obtener el opuesto de un número se invierten todos sus bits (complemento a uno) y se suma 1 en el bit menos significativo (LSB). Tiene una única representación del cero.

## Desbordamiento (Overflow) en Complemento a Dos
El desbordamiento aritmético en suma de complemento a dos ocurre únicamente cuando se suman dos operandos del mismo signo y el resultado tiene el signo opuesto. A nivel de hardware se detecta mediante la compuerta XOR entre el acarreo de entrada y el acarreo de salida del bit más significativo: `V = Cn-1 XOR Cn`.

## Punto Flotante IEEE 754
El estándar **IEEE 754 de precisión simple (32 bits)** se compone de:
- **1 bit de signo (S)**.
- **8 bits de exponente sesgado (E)** con exceso (bias) de **127**.
- **23 bits de mantisa fraccionaria (M)** con un bit implícito normalizado `1.M`.

En **doble precisión (64 bits)** se utilizan 1 bit de signo, 11 bits de exponente con exceso **1023** y 52 bits de mantisa.

## Banderas de Estado de la ALU
La ALU actualiza cuatro indicadores principales tras cada operación:
- **Z (Zero)**: Se activa en 1 cuando todos los bits del resultado son cero.
- **N (Negative)**: Copia el bit más significativo (MSB) del resultado.
- **C (Carry)**: Indica acarreo de salida en operaciones sin signo.
- **V (Overflow)**: Indica desbordamiento en aritmética con signo (Ca2).
