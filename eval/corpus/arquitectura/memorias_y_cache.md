# Jerarquía de Memoria y Caché

## Niveles de Caché
Los procesadores modernos implementan tres niveles de memoria caché en el chip:
- **L1 (Nivel 1)**: Dividida en L1i (instrucciones) y L1d (datos), típicamente de 32 KB a 64 KB por núcleo, con una latencia de 1 a 4 ciclos de reloj.
- **L2 (Nivel 2)**: Unificada por núcleo, con tamaños entre 256 KB y 1 MB y latencias de 10 a 14 ciclos.
- **L3 (Nivel 3 / LLC)**: Compartida entre todos los núcleos del procesador, de 8 MB a 64 MB y latencia de 30 a 50 ciclos.

## Políticas de Correspondencia (Mapping)
1. **Correspondencia Directa**: Cada bloque de memoria principal solo puede ubicarse en una única línea de caché calculada como `índice = (dirección_bloque) mod (número_de_líneas)`.
2. **Totalmente Asociativa**: Un bloque puede almacenarse en cualquier línea de la caché; requiere comparadores paralelos en todas las etiquetas (tags).
3. **Asociativa por Conjuntos (N-way Set Associative)**: La caché se divide en conjuntos de N vías (por ejemplo, 8 vías). El bloque se asigna a un conjunto mediante módulo y puede ocupar cualquiera de las N vías del conjunto.

## Políticas de Escritura
- **Write-Through (Escritura Inmediata)**: Toda escritura en la caché actualiza simultáneamente la memoria principal. Utiliza un *Write Buffer* para evitar bloquear el procesador.
- **Write-Back (Post-escritura)**: Los datos modificados solo se escriben en la caché marcando el bit de modificación (*dirty bit* = 1). La memoria principal solo se actualiza cuando la línea es reemplazada.

## Algoritmos de Reemplazo
- **LRU (Least Recently Used)**: Reemplaza la línea del conjunto que lleva más tiempo sin ser referenciada.
- **FIFO (First-In, First-Out)**: Reemplaza la línea más antigua cargada en el conjunto.
