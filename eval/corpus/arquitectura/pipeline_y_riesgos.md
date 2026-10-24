# Segmentación de Cauce (Pipeline) y Riesgos (Hazards)

## Etapas del Pipeline RISC Clásico de 5 Etapas
1. **IF (Instruction Fetch)**: Búsqueda de la instrucción en la memoria de instrucciones usando el contador de programa (PC).
2. **ID (Instruction Decode)**: Decodificación de la instrucción y lectura de registros en el banco de registros.
3. **EX (Execute)**: Ejecución de la operación en la ALU o cálculo de dirección efectiva de memoria.
4. **MEM (Memory Access)**: Lectura o escritura en memoria de datos (instrucciones `lw` y `sw`).
5. **WB (Write Back)**: Escritura del resultado de vuelta en el banco de registros.

## Tipos de Riesgos (Hazards)
- **Riesgos Estructurales**: Ocurren cuando dos instrucciones en distintas etapas necesitan el mismo recurso físico en el mismo ciclo (por ejemplo, una única memoria unificada para instrucciones y datos).
- **Riesgos de Datos (Data Hazards)**: Una instrucción depende del resultado de una instrucción previa que aún no completó su etapa WB. El caso más frecuente en RISC es **RAW (Read After Write)**.
- **Riesgos de Control (Branch Hazards)**: Provocados por instrucciones de salto condicional (`beq`, `bne`) donde la dirección destino y la condición recién se conocen en la etapa ID o EX.

## Técnicas de Mitigación
- **Adelantamiento (Forwarding / Bypassing)**: Conecta la salida de los registros intermedios `EX/MEM` y `MEM/WB` directamente a las entradas de la ALU para evitar burbujas en dependencias RAW entre instrucciones aritméticas.
- **Detención por Load-Use (Stall)**: Cuando una instrucción `lw` es seguida inmediatamente por otra que usa el registro cargado, el hardware inserta obligatoriamente **1 ciclo de burbuja (NOP)** incluso con forwarding.
- **Predicción de Saltos (Branch Prediction)**: El predictor dinámico de **2 bits con saturación** requiere fallar dos veces consecutivas antes de cambiar su predicción, mejorando el rendimiento en bucles.
