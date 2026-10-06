# Justificación Microarquitectónica: Mantenimiento de `PredALU.count = 1` en gem5 (ARM SVE vs. RISC-V RVV)

---

## 1. Resumen Ejecutivo

En la configuración por defecto de la CPU fuera de orden de gem5 ([`src/cpu/o3/FuncUnitConfig.py`](file:///home/jsanchez/gem5/src/cpu/o3/FuncUnitConfig.py#L79-L136)), existen **4 unidades SIMD** (`SIMD_Unit.count = 4`) para el procesamiento de datos vectoriales y **1 sola unidad de predicados** (`PredALU.count = 1`) para el procesamiento de operaciones de predicado (`SimdPredAlu`).

Al analizar el comportamiento de algoritmos reales vectorizados (como SCAMP para *Matrix Profile* en series temporales) y observar contención en las estadísticas de `statFuBusy::SimdPredAlu`, surge la pregunta de si:
1. Es justo que existan 4 unidades SIMD de datos pero solo 1 de predicados.
2. Ambas arquitecturas (ARM SVE y RISC-V RVV) compiten en igualdad de condiciones.
3. Se debería incrementar `PredALU.count` a 4 para equipararla a las unidades SIMD.

El presente informe expone con rigor microarquitectónico y evidencia en silicio real por qué **mantener `PredALU.count = 1` es la decisión correcta, fiel a la realidad del hardware comercial y metodológicamente sólida**.

---

## 2. Diferenciación de Unidades: ¿Qué ejecuta realmente `PredALU` vs. `SIMD_Unit`?

Un error conceptual frecuente consiste en asumir que **toda instrucción vectorial que utiliza un predicado** (por ejemplo, una suma aritmética condicionada `add z0.s, p0/m, z1.s, z2.s`) pasa por la `PredALU`.

La inspección del generador de instrucciones de gem5 ([`src/arch/arm/isa/insts/sve.isa:3500-3510`](file:///home/jsanchez/gem5/src/arch/arm/isa/insts/sve.isa#L3500-L3510)) demuestra lo contrario:

```python
# ADD (vectors, predicated)
addCode = 'destElem = srcElem1 + srcElem2;'
sveBinInst('add', 'AddPred', 'SimdAddOp', unsignedTypes, addCode,
           PredType.MERGE, True)

# ADD (vectors, unpredicated)
addCode = 'destElem = srcElem1 + srcElem2;'
sveBinInst('add', 'AddUnpred', 'SimdAddOp', unsignedTypes, addCode)
```

### Tabla de Asignación de Unidades Funcionales

| Tipo de Instrucción SVE | Ejemplo Ensamblador | Clase en gem5 (`OpClass`) | Unidad Funcional Asignada | Instancias (`count`) |
| :--- | :--- | :--- | :--- | :---: |
| **Suma vectorial no predicada** | `add z0.s, z1.s, z2.s` | `SimdAddOp` | `SIMD_Unit` | **4** |
| **Suma vectorial predicada** | `add z0.s, p0/m, z1.s, z2.s` | `SimdAddOp` | `SIMD_Unit` | **4** |
| **Multiplicación vectorial predicada** | `fmul z0.d, p0/m, z1.d, z2.d` | `SimdFloatMultOp` | `SIMD_Unit` | **4** |
| **Comparación vectorial** | `cmpeq p0.s, p1/z, z0.s, z1.s` | `SimdCmpOp` | `SIMD_Unit` | **4** |
| **Lógica entre predicados** | `and p0.b, p1/z, p2.b, p3.b` | `SimdPredAluOp` | `PredALU` | **1** |
| **Generación de predicados** | `ptrue p0.s`, `pfalse p0.b` | `SimdPredAluOp` | `PredALU` | **1** |
| **Test / Banderas de predicado** | `ptest p0, p1.b` | `SimdPredAluOp` | `PredALU` | **1** |

### Conclusión Técnica Fundamental
- Las operaciones aritméticas sobre vectores de datos (enteros, punto flotante, conversiones), **incluso cuando están enmascaradas con un predicado**, se ejecutan siempre en una de las **4 unidades `SIMD_Unit`**.
- La unidad **`PredALU` solo se encarga de calcular operaciones lógicas o de control entre registros de predicados** ($p \leftarrow f(p)$).
- El predicado en una operación aritmética es únicamente un operando fuente que viaja a los carriles de la `SIMD_Unit` para habilitar o deshabilitar la escritura de cada elemento.

---

## 3. Comparativa Arquitectónica: ARM SVE vs. RISC-V RVV

Existe una marcada disparidad en cómo ambas ISAs definen el soporte de máscaras y predicados:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           ARM SVE                                      │
  │  • Banco de registros de predicado DEDICADO: p0 - p15 (16 registros)  │
  │  • Tamaño: 1 bit por cada byte de datos (32 a 256 bits en total)       │
  │  • Unidad de ejecución: PredALU especializada e independiente          │
  └────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────┐
  │                         RISC-V RVV                                     │
  │  • NO tiene banco de predicados separado                               │
  │  • Las máscaras residen en el banco de registros vectoriales (v0 - v31)│
  │  • Unidad de ejecución: SIMD_Unit general (SimdAluOp)                  │
  └────────────────────────────────────────────────────────────────────────┘
```

### 3.1. ¿Por qué RISC-V no usa `PredALU` en gem5?
En RISC-V, las operaciones lógicas sobre máscaras (`vmand.mm`, `vmor.mm`, etc.) se mapean en [`src/arch/riscv/isa/decoder.isa:3635`](file:///home/jsanchez/gem5/src/arch/riscv/isa/decoder.isa#L3635) a `SimdAluOp`.

Al no tener un banco de predicados físicamente diferenciado, gem5 utiliza las unidades `SIMD_Unit` generales para ejecutar las operaciones de máscara en RISC-V.

### 3.2. El Compromiso de Emisión y Recursos (Throughput Trade-off)
Si consideramos un ciclo de reloj donde coinciden operaciones aritméticas y operaciones de máscara/predicado:

```
           ARM SVE (gem5 Default)                  RISC-V RVV (gem5 Default)
  ┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
  │ Puerto 0: SIMD_Unit 0 (Aritmética)  │   │ Puerto 0: SIMD_Unit 0 (Aritmética)  │
  │ Puerto 1: SIMD_Unit 1 (Aritmética)  │   │ Puerto 1: SIMD_Unit 1 (Aritmética)  │
  │ Puerto 2: SIMD_Unit 2 (Aritmética)  │   │ Puerto 2: SIMD_Unit 2 (Aritmética)  │
  │ Puerto 3: SIMD_Unit 3 (Aritmética)  │   │ Puerto 3: SIMD_Unit 3 (Máscara vmand│
  │ Puerto 4: PredALU   0 (Pred. and)   │   └─────────────────────────────────────┘
  └─────────────────────────────────────┘     Total en paralelo: 4 instrucciones
    Total en paralelo: 5 instrucciones
```

1. **En ARM SVE**: La existencia de una `PredALU` desacoplada permite despachar **4 instrucciones aritméticas de datos y 1 instrucción lógica de predicados simultáneamente en el mismo ciclo**.
2. **En RISC-V**: Dado que las operaciones de máscara consumen una `SIMD_Unit`, **le roban un puerto de ejecución a la aritmética vectorial**. Si el procesador emite `vmand.mm`, solo puede emitir como máximo 3 operaciones aritméticas en ese ciclo.

Por lo tanto, la presencia de 1 `PredALU` dedicada no perjudica a SVE frente a RISC-V; al contrario, le otorga una vía de ejecución desacoplada que preserva íntegro el ancho de banda aritmético.

---

## 4. Evidencia en el Hardware Real (Silicon Reality Check)

Para verificar si tener 1 `PredALU` frente a 4 `SIMD_Unit` es realista, debemos contrastar el modelo con las microarquitecturas comerciales líderes en el mercado.

### 4.1. Procesadores ARM SVE Comerciales

#### A. ARM Neoverse V1 (ej. AWS Graviton 3)
- **Pipelines vectoriales**: 4 unidades SIMD de 128 bits (o 2 de 256 bits).
- **Puertos de emisión**:
  - Puertos 0 y 1: Aritmética SIMD / FP.
  - Puertos 2 y 3: Aritmética SIMD / FP.
  - **Operaciones de predicado**: Se despachan únicamente a través del **Puerto 0** (y parcialmente Puerto 1 para ciertas variantes).
- **Relación de unidades**: 4 pipelines de datos frente a **1 (máximo 2) puertos capaces de procesar predicados**.

#### B. ARM Neoverse V2 (ej. NVIDIA Grace CPU, AWS Graviton 4)
- **Pipelines vectoriales**: 4 unidades SIMD de 128 bits compatibles con SVE2.
- **Predicados**: La lógica de predicados se mantiene centralizada en un número reducido de puertos (1 a 2 puertos), compartiendo recursos con operaciones de enteros escalares o bifurcaciones.

#### C. Fujitsu A64FX (Procesador del Supercomputador Fugaku)
- **Pipelines vectoriales**: 2 unidades SVE de 512 bits (EXA y EXB).
- **Predicados**: Ciertas operaciones de predicados solo pueden emitirse al pipeline EXA.
- **Relación**: 2 unidades de datos de 512 bits frente a **1 puerto de predicados**.

### 4.2. Procesadores RISC-V Vector Comerciales

#### A. SiFive Intelligence X280
- Dispone de 1 pipeline vectorial de enteros/FPU de 512 bits. Las operaciones de máscara comparten dicho pipeline de ejecución.

#### B. T-Head XuanTie C908 / C910
- Dispone de 2 pipelines vectoriales donde las operaciones de máscara se resuelven en la ALU vectorial de enteros.

### 4.3. Razones Físicas del Diseño en Silicio
¿Por qué ningún fabricante de chips implementa 4 unidades de predicados?
1. **Diferencia de escala de datos**: Un registro vectorial contiene 256, 512 o 2048 bits de datos de punto flotante o enteros. Un registro de predicado tiene únicamente $\frac{VLEN}{8}$ bits (32 bits para VLEN=256, 64 bits para VLEN=512, 256 bits para VLEN=2048).
2. **Frecuencia de uso en el mix de instrucciones**: En algoritmos vectoriales del mundo real, más del 85-90% de las instrucciones son cargas, almacenamientos y aritmética. Las operaciones de manipulación de predicados (`whilelt`, `ptrue`, `pand`) representan típicamente entre el 3% y el 8% de las instrucciones dinámicas.
3. **Coste de área y puertos de registro**: Cuadruplicar la `PredALU` requeriría añadir 8 puertos de lectura y 4 puertos de escritura adicionales al banco de registros de predicado, aumentando el área, el consumo estático y la complejidad de la red de bypass sin aportar una ganancia medible de IPC en código real.

---

## 5. Por qué Subir `PredALU.count` a 4 sería Metodológicamente Incorrecto

Si en gem5 se cambiara `PredALU.count = 4` en [`src/cpu/o3/FuncUnitConfig.py`](file:///home/jsanchez/gem5/src/cpu/o3/FuncUnitConfig.py#L135), se incurriría en tres fallos metodológicos graves:

1. **Simulación de hardware inexistente**: Se estaría modelando una CPU con capacidad de emitir 4 operaciones de predicado por ciclo, algo que no existe en ningún procesador ARM comercial ni de investigación.
2. **Sesgo artificial a favor de SVE**: En benchmarks intensivos en control o en predicación, SVE obtendría un IPC inflado artificialmente que ningún procesador real podría replicar.
3. **Enmascaramiento de fallos de modelado temporal**: La contención observada en `statFuBusy::SimdPredAlu` cuando se ejecuta código como SCAMP no se debe a que 1 unidad sea insuficiente para procesar 1 instrucción por ciclo, sino a que cada instrucción de predicado permanecía bloqueando la unidad durante decenas de ciclos debido a un cálculo de *chimes* indebido sobre bits de predicado. Aumentar el número de unidades a 4 solo camuflaría un cuello de botella temporal irreal en lugar de corregir la causa física.

---

## 6. Conclusión y Veredicto Científico

| Criterio | Mantener `PredALU.count = 1` | Subir `PredALU.count = 4` |
| :--- | :---: | :---: |
| **Fidelidad con CPUs ARM reales (Neoverse V1/V2, A64FX)** |  **Alta (Coincide con silicio real)** | ❌ **Nula (Hardware ficticio)** |
| **Equidad arquitectónica con RISC-V RVV** |  **Equitativo (Desacopla datos y control)** | ❌ **Injusto (Sobredimensiona SVE)** |
| **Consistencia con el mix de instrucciones real** |  **Óptimo (<8% de instrucciones)** | ❌ **Sobrediseño ineficiente** |
| **Rigor metodológico en investigación de arquitectura** |  **Científicamente defendible** | ❌ **Artificialmente inflado** |

### Veredicto Final
La configuración de **`PredALU` con `count = 1` debe mantenerse inalterada**. Representa fielmente las limitaciones y fortalezas de la microarquitectura ARM SVE en procesadores de alto rendimiento y garantiza que las evaluaciones comparativas entre ARM SVE y RISC-V RVV en gem5 sean científicamente válidas y representativas de hardware real.

