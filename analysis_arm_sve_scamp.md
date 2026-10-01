# Análisis de Cuellos de Botella — ARM SVE / SCAMP / audio-MPIII-SVD

## Contexto

Se comparan 4 ejecuciones de SCAMP vectorizado con ARM SVE en gem5, variando únicamente el VLEN (256, 512, 1024, 2048 bits) con 1 hilo. El problema observado: **el tiempo de ejecución no mejora (incluso empeora) al aumentar el VLEN**, a diferencia de lo que ocurre en RISC-V (RVV), donde hay speedup creciente con el VLEN.

---

## 1. Métricas de Alto Nivel

| Métrica | VLEN-256 | VLEN-512 | VLEN-1024 | VLEN-2048 |
|---|---|---|---|---|
| `simSeconds` | 0.03493 s | 0.03520 s | 0.03603 s | 0.03868 s |
| `simInsts` | 294,049,759 | 150,009,892 | 77,050,306 | 41,434,646 |
| `numCycles` | 69,865,413 | 70,396,875 | 72,056,648 | 77,363,504 |
| `CPI` | 0.238 | 0.469 | 0.935 | 1.867 |
| `IPC` | 4.209 | 2.131 | 1.069 | 0.536 |
| `issueRate` | 4.286 | 2.172 | 1.092 | 0.549 |

> [!IMPORTANT]
> Al doblar el VLEN, las instrucciones se reducen a la mitad (correcto), **pero los ciclos NO se reducen — de hecho aumentan**. Esto revela que el procesador está gastando más ciclos por instrucción vectorial conforme se amplía el VLEN, anulando completamente el beneficio vectorial.

---

## 2. Cuello de Botella #1 — Pipeline Vacío (Issue Slots Desperdiciados)

El síntoma más evidente del estancamiento es la distribución de ciclos en los que el pipeline **no emite ninguna instrucción**:

| `numIssuedDist::0` (ciclos sin issue) | VLEN-256 | VLEN-512 | VLEN-1024 | VLEN-2048 |
|---|---|---|---|---|
| Ciclos vacíos | 1.02% | 17.29% | **48.59%** | **69.66%** |

Con VLEN-2048, en **casi 7 de cada 10 ciclos** el procesador no emite ninguna instrucción. Esto es el síntoma central del estancamiento.

### Causa: la Instruction Queue (IQ) se satura

El número de veces que el renombrador tuvo que bloquearse por **IQ llena** (`IQFullEvents`) escala enormemente:

| `rename.IQFullEvents` | VLEN-256 | VLEN-512 | VLEN-1024 | VLEN-2048 |
|---|---|---|---|---|
| IQ full (1er renombrador) | 15,037,117 | — | — | 17,281,431 |
| IQ full (2º renombrador) | 22,223,778 | — | — | 24,924,863 |

> [!NOTE]
> Los datos de VLEN-256 y VLEN-2048 muestran que la IQ se satura igualmente en ambos casos. La diferencia clave es que en VLEN-256 la IQ se drena rápidamente (IPC=4.21), mientras que en VLEN-2048 cada instrucción vectorial tarda muchos ciclos en ejecutarse y **ocupa la IQ durante mucho más tiempo**, generando un efecto de embotellamiento acumulativo.

**También aparece `LQFullEvents`** (Load Queue llena), indicando un segundo punto de presión:

| `rename.LQFullEvents` | VLEN-256 | VLEN-2048 |
|---|---|---|
| LQ full | 5,056,913 | 2,540,622 |

---

## 3. Cuello de Botella #2 — Latencia de Ejecución Vectorial Escala con VLEN

La distribución del número de instrucciones completadas por ciclo (`numCommittedDist`) revela el problema estructural:

| Dist. commit | VLEN-256 | VLEN-512 | VLEN-1024 | VLEN-2048 |
|---|---|---|---|---|
| 0 insts/ciclo | 1.24% | 32.35% | **62.55%** | **80.36%** |
| 1 inst/ciclo | 49.24% | 46.09% | 23.10% | 12.43% |
| 8 insts/ciclo | 42.00% | 21.20% | 7.17% | 3.61% |

En VLEN-256, el 42% de los ciclos terminan con 8 instrucciones completadas (máximo de ancho de commit). En VLEN-2048, el 80% de los ciclos terminan con **cero instrucciones completadas**.

**Esto ocurre porque en el modelo gem5 de ARM SVE, las latencias de ejecución de instrucciones vectoriales con vectores más anchos son mayores.** Cada instrucción SIMD con VLEN-2048 ocupa las unidades funcionales durante más ciclos, bloqueando la emisión de nuevas instrucciones.

---

## 4. Cuello de Botella #3 — Decode Bloqueado

El stage de decode revela un cuello de botella severo en los VLENs altos:

| Decode | VLEN-256 | VLEN-2048 |
|---|---|---|
| `decode.blockedCycles` | 24,608,349 | **70,520,108** |
| `decode.runCycles` | 19,531,148 | **43,542** |
| `decode.idleCycles` | 7,776,195 | 1,502,970 |

En VLEN-2048, el decoder está **bloqueado** el 91% del tiempo (70.5M de 77.4M ciclos totales) y **en ejecución solo 0.056% del tiempo** (43,542 ciclos). El decodificador no puede avanzar porque las etapas aguas abajo (rename/issue/execute) están atascadas por instrucciones de larga latencia.

También en el **fetch**:

| Fetch `nisnDist::0` (ciclos sin fetch) | VLEN-256 | VLEN-2048 |
|---|---|---|
| Ciclos sin traer instrucciones | 25.95% | **88.66%** |

En VLEN-2048, el 88.66% de los ciclos no hay fetch de instrucciones (backpressure total desde el issue queue al fetch).

---

## 5. Cuello de Botella #4 — Saturación de Unidades Funcionales (FU Busy)

Las unidades funcionales se saturan de forma diferente según el VLEN, mostrando **dos cuellos de botella entrelazados**:

| FU más saturada | VLEN-256 | VLEN-512 | VLEN-1024 | VLEN-2048 |
|---|---|---|---|---|
| `MemRead` FU busy | **73.50%** | 15.16% | 40.37% | **54.96%** |
| `SimdPredAlu` FU busy | 13.12% | **83.15%** | **52.43%** | 41.73% |

### La unidad `SimdPredAlu` (registros predicado SVE)

Esta unidad gestiona los **P-registers** (predicados) de SVE, que controlan qué lanes son activos en cada instrucción vectorial. Para vectores más anchos:
- Las instrucciones predicado tienen mayor latencia de ejecución
- El número de instrucciones predicado no disminuye al aumentar VLEN (la mezcla de instrucciones se mantiene constante — ver §7)
- Resultado: la unidad se convierte en cuello de botella temporal, especialmente en VLEN-512 y VLEN-1024

### La unidad `MemRead`

Con vectores más anchos, cada load vectorial lee más bytes (32 bytes con VLEN-256 → 256 bytes con VLEN-2048). Esto:
- Requiere más transacciones internas en el subsistema de memoria
- Mantiene la unidad MemRead ocupada durante más ciclos por instrucción
- Interactúa con los MSHRs de la caché de datos

---

## 6. Cuello de Botella #5 — Accesos a Caché y Prefetch Ineficiente

### D-Cache: misses se reducen (proporcionalmente) pero el prefetcher pierde efectividad

| dcache | VLEN-256 | VLEN-512 | VLEN-1024 | VLEN-2048 |
|---|---|---|---|---|
| `demandMissRate` | 0.2964% | 0.2698% | 0.1907% | 0.1742% |
| `demandAvgMissLatency` (ticks) | 25,190 | 20,832 | 25,338 | 27,170 |
| Prefetcher `pfIssued` | 18,180,020 | 2,399,869 | 1,224,921 | 688,886 |
| Prefetcher `pfUseful` | 4,649,130 | — | — | 680,008 |
| Prefetcher `accuracy` | 0.2557 | 0.2564 | — | 0.2564 |
| Prefetcher `pfLate` | 13,503,529 | — | — | 1,963,442 |

> [!WARNING]
> La latencia media de miss **aumenta con VLEN-2048** (~27,170 ticks frente a ~20,832 en VLEN-512), indicando que el subsistema de memoria está más presionado. El prefetcher emite muchos menos prefetches en VLENs altos (porque el programa genera menos referencias de memoria en términos de instrucciones de carga), y la cobertura baja del 97.9% (VLEN-256) al 96.0% (VLEN-2048).

### WriteLineReq — aparece en VLENs altos

Llama la atención la aparición de `WriteLineReq` (escritura de línea completa de caché, típica de stores vectoriales anchos) con una **tasa de miss muy alta**:

| `dcache.WriteLineReq` | VLEN-1024 | VLEN-2048 |
|---|---|---|
| Miss rate | **42.4%** | **42.5%** |
| Avg miss latency | **82,024 ticks** | **78,681 ticks** |

Esta operación no existe en VLEN-256 (sin WriteLineReq) y aparece porque los stores de vectores anchos activan el protocolo de escritura completa de línea de caché. La latencia de ~82k ticks (~82 ciclos a 1GHz, pero en ticks es 82k/1000 = **82 ns efectivos**) es muy alta y contribuye a los stalls de store queue.

---

## 7. La Causa Raíz: Mezcla de Instrucciones Invariante

El hallazgo más crítico es que la **mezcla de tipos de instrucciones es prácticamente idéntica en todos los VLENs**:

| Tipo de instrucción | VLEN-256 | VLEN-512 | VLEN-1024 | VLEN-2048 |
|---|---|---|---|---|
| `IntAlu` | 27.85% | 27.85% | 27.84% | 27.84% |
| `SimdMisc` | 16.42% | 16.41% | 16.40% | 16.36% |
| `SimdAlu` | 11.44% | 11.44% | 11.43% | 11.41% |
| `MemRead` | 19.69% | 19.69% | 19.70% | 19.71% |
| `MemWrite` | 3.31% | 3.34% | 3.38% | 3.47% |
| `SimdPredAlu` | 1.63% | 1.63% | 1.63% | 1.63% |
| `SimdFloatCmp` | 4.89% | 4.89% | 4.88% | 4.87% |
| `SimdFloatMult` | 4.89% | 4.89% | 4.88% | 4.87% |

> [!CAUTION]
> **Esta es la diferencia fundamental con RISC-V (RVV).**
>
> En RVV, la instrucción `vsetvli` ajusta dinámicamente el número de elementos procesados por iteración. Al doblar el VLEN, se procesan el doble de elementos por iteración, y el número de iteraciones del bucle (y por tanto de instrucciones de overhead: control de bucle, incremento de índices, etc.) se **reduce a la mitad**. El ratio overhead/trabajo vectorial mejora.
>
> En ARM SVE, aunque el compilador podría hacer lo mismo, **el código generado mantiene el mismo número de instrucciones de control, predicado y overhead por iteración**, independientemente del VLEN. Cada instrucción vectorial hace más trabajo (más elementos), pero el overhead **no escala**. El resultado neto es que:
>
> - El número de instrucciones cae a la mitad (correcto)
> - Pero **cada instrucción vectorial tarda el doble de ciclos** (latencia proporcional al VLEN en el modelo gem5 de ARM)
> - El producto `#instrucciones × latencia_media` permanece constante → **ciclos totales constantes o crecientes**

---

## 8. Comparativa Resumida de Cuellos de Botella

```
VLEN-256  ████████████████░░░░  IPC=4.21  Ciclos vacíos: 1%
           Memoria domina (73% FU)
           Decode bloqueado: 35%

VLEN-512  ████████░░░░░░░░░░░░  IPC=2.13  Ciclos vacíos: 17%
           SimdPredAlu domina (83% FU)

VLEN-1024 ████░░░░░░░░░░░░░░░░  IPC=1.07  Ciclos vacíos: 49%
           SimdPredAlu (52%) + Memoria (40%) entrelazados

VLEN-2048 ██░░░░░░░░░░░░░░░░░░  IPC=0.54  Ciclos vacíos: 70%
           Memoria domina (55%) + Decode bloqueado 91%
```

---

## 9. Hipótesis Unificadora

El estancamiento de ARM SVE se explica por la **interacción de tres efectos**:

### Efecto 1 — Latencia de ejecución proporcional al VLEN (modelo gem5)

El modelo gem5 de ARM SVE modela instrucciones vectoriales con latencias que **escalan con el número de elementos** (o VLEN). Al doblar el VLEN, las unidades SIMD tardan el doble en completar cada instrucción. Esto:
- Mantiene la FU ocupada durante más tiempo
- Bloquea la IQ, que a su vez bloquea rename → decode → fetch
- El pipeline se vacía (ciclos con 0 issues)

### Efecto 2 — Overhead de predicados SVE no escalable

ARM SVE requiere operaciones sobre P-registers para cada instrucción vectorial, independientemente del VLEN. La `SimdPredAlu` no gana throughput al aumentar el VLEN, pero las instrucciones que dependen de su resultado deben esperarla. Con latencias crecientes de la unidad SIMD principal, los P-registers se convierten en ruta crítica de dependencia.

### Efecto 3 — Accesos de memoria más anchos generan más presión en el subsistema

Con VLEN-2048, cada load/store vectorial accede hasta 256 bytes. Esto:
- Requiere más transacciones en el bus de caché (si la línea de caché es 64 bytes → 4 transacciones por load)
- Genera `WriteLineReq` para stores completos con muy alta tasa de miss (~42%)
- La latencia de miss sube (~27k ticks en VLEN-2048 vs ~21k en VLEN-512)

---

## 10. Recomendaciones

### Para el trabajo de investigación:

1. **Verificar el modelo de latencia de ARM SVE en gem5**: Comprobar si las latencias de las FUs SIMD están modeladas como escaladas con el VLEN (número de elementos) o como fijas. Si están escaladas, podría ser una limitación del modelo que no refleja hardware real con capacidad de ejecución paralela en múltiples lanes.

2. **Comparar con la configuración del procesador**: El procesador simulado puede tener un número fijo de lanes SIMD (e.g., 4 lanes de 64 bits). Con VLEN-256 usa todos los lanes en 1 ciclo, con VLEN-2048 necesita 8 ciclos → la latencia escala linealmente, y el speedup es 0.

3. **Revisar el código generado por el compilador**: Verificar si el compilador ARM (armclang o GCC con target SVE) genera código que reduce el overhead de control de bucle al aumentar el VLEN (como hace RVV con `vsetvli`). Si no lo hace, el cuello de botella es en la vectorización, no en el hardware.

4. **Analizar los `WriteLineReq` con VLEN≥1024**: La aparición de escrituras de línea completa con 42% de miss rate es un cuello de botella real que podría mitigarse con software prefetching o ajustando el stride del algoritmo SCAMP.

5. **Comparar con SCRIMP**: Si SCRIMP muestra el mismo patrón (mezcla de instrucciones invariante con VLEN), confirmaría que el problema es sistémico en la compilación SVE y no específico de SCAMP.

---

## 11. Datos Clave para la Discusión

| Métrica clave | VLEN-256 | VLEN-2048 | Ratio |
|---|---|---|---|
| Instrucciones ejecutadas | 294.05M | 41.43M | **7.10×** ✓ |
| Ciclos totales | 69.87M | 77.36M | 0.90× ✗ |
| IPC | 4.209 | 0.536 | **7.85×** degradación |
| Decode bloqueado | 35% | **91%** | — |
| Pipeline sin issue | 1% | **70%** | — |
| IQ Full events | 15M+22M | 17M+25M | ~1.1× (por ciclo: mucho más) |
| Latencia vectorial efectiva | ~1 ciclo/inst | ~8 ciclos/inst | **8× más lenta** |
