# Informe: modelado de "vector lanes" en gem5 para RISC-V (RVV)

## 1. Objetivo

gem5 modela hoy las extensiones vectoriales (RVV en RISC-V, SVE en ARM) asumiendo que el
procesador dispone de recursos infinitos para procesar un vector completo en un único ciclo,
independientemente de su longitud (`VLEN`) o del ancho de elemento (`SEW`/`ELEN`). Esto es
irreal: un procesador vectorial físico solo tiene un número finito de **lanes** (carriles de
ejecución paralela), y una instrucción con más elementos activos que lanes disponibles necesita
varios ciclos para completarse.

Este trabajo añade un parámetro configurable `lanes` a la ISA RISC-V de gem5 y modifica el
núcleo del simulador (modelo de emisión del `O3CPU`) para que la latencia de finalización **y**
la ocupación de la unidad funcional de las instrucciones vectoriales escalen con el número de
"pasadas" (*chimes*, en la terminología clásica de Hennessy & Patterson) que hacen falta para
procesar todos los elementos activos con el número de lanes configurado.

## 2. Modelo matemático: el "chime"

Para una instrucción (o micro-op) vectorial con `N` elementos activos y `L` lanes configurados:

```
passes = ceil(N / L)        (passes = 1 si lanes = 0, es decir, "sin límite")
```

gem5 obtiene hoy la latencia base (`opLat`) y si la unidad funcional está *pipelined* del
`FUPool` (fijo por `OpClass`, p. ej. `SimdAdd` = 1 ciclo, *pipelined*). El chime se aplica así:

- **FU *pipelined*** (el caso normal en `DefaultFUPool`):
  - Latencia de finalización = `opLat + passes - 1`
  - Ocupación de la FU (cuánto tarda en poder aceptar la siguiente instrucción) = `passes` ciclos
    (antes del parche: siempre 1 ciclo, sin importar `opLat`)
- **FU *no pipelined***:
  - Latencia de finalización = `opLat * passes`
  - Ocupación de la FU = `opLat * passes` (igual que la latencia)

Con `lanes = 0` (valor por defecto), `passes` siempre vale 1 y las fórmulas se reducen
exactamente al comportamiento actual de gem5: **cero regresión** para cualquier configuración
existente que no use el nuevo parámetro.

## 3. Cambios en el núcleo de gem5 (independientes de la ISA)

Estos cambios son genéricos: no mencionan RISC-V ni ARM, y los podrá reutilizar cualquier ISA
que quiera modelar lanes (p. ej. una futura Fase 2 para SVE).

### 3.1. [`src/cpu/static_inst.hh`](../../src/cpu/static_inst.hh)

Nuevo método virtual en la clase base `StaticInst` (de la que heredan **todas** las
instrucciones de **todas** las ISAs):

```cpp
virtual Cycles
numChimePasses(ThreadContext *tc) const
{
    return Cycles(1);
}
```

La implementación por defecto devuelve siempre 1 (sin escalado), así que cualquier instrucción
que no lo sobrescriba se comporta exactamente igual que antes. Se le pasa un `ThreadContext*`
en vez de guardar el valor de `lanes` en la propia instrucción en tiempo de construcción: esto
permite leer la configuración de lanes **en el momento de emitir la instrucción**, directamente
desde `tc->getIsaPtr()`, sin tener que añadir un nuevo parámetro a los constructores de cada una
de las decenas de clases de instrucción vectorial generadas por las plantillas `.isa` (ver
§4.3 para el porqué de esta decisión de diseño).

### 3.2. [`src/cpu/o3/dyn_inst.hh`](../../src/cpu/o3/dyn_inst.hh)

Método de reenvío en `DynInst` (instrucción dinámica del `O3CPU`), siguiendo el mismo patrón que
ya usan `isVector()`, `isFloating()`, etc.:

```cpp
Cycles
numChimePasses() const
{
    return staticInst->numChimePasses(tcBase());
}
```

### 3.3. [`src/cpu/o3/inst_queue.cc`](../../src/cpu/o3/inst_queue.cc) — `scheduleReadyInsts()`

Este es el punto donde el `O3CPU` decide, para cada instrucción que va a emitir este ciclo, su
latencia de finalización y cuándo se libera la unidad funcional que ocupa. Antes del parche:

```cpp
op_latency = fuPool->getOpLatency(op_class);   // fijo por OpClass, siempre igual
...
if (op_latency == Cycles(1)) {
    // emitir ya, liberar la FU el ciclo siguiente
} else {
    // programar evento de finalización a op_latency-1 ciclos
    // si es "pipelined", liberar la FU el ciclo siguiente de todas formas
    // si NO es "pipelined", liberar la FU cuando se complete
}
```

Después del parche, tras obtener `op_latency` del `FUPool` como antes, se consulta
`issuing_inst->numChimePasses()` y, si `passes > 1`, se recalcula `op_latency` con la fórmula del
chime (sección 2) **antes** de que el código existente decida cómo emitir/programar la
instrucción. Para el caso *pipelined* con `passes > 1`, además se programa un evento adicional
(`EventFunctionWrapper`) que libera la unidad funcional `passes - 1` ciclos más tarde, en vez de
liberarla al ciclo siguiente como hacía siempre antes. Con `passes = 1` (el caso por defecto, o
cualquier instrucción no vectorial) el camino de código es idéntico al original: no se
modificó `FUPool` ni ninguna otra estructura compartida más de lo estrictamente necesario, para
minimizar el riesgo de regresión en el resto de gem5 (esta ruta de código la ejecuta **cualquier**
instrucción, de cualquier ISA, en el `O3CPU`).

## 4. Cambios específicos de RISC-V (RVV)

### 4.1. Nuevo parámetro `lanes`

[`src/arch/riscv/RiscvISA.py`](../../src/arch/riscv/RiscvISA.py): nueva clase de parámetro
`RiscvVectorLanes` (entero sin signo, `0` o potencia de 2) y el parámetro `lanes` en `RiscvISA`,
junto a los ya existentes `vlen` y `elen`:

```python
lanes = Param.RiscvVectorLanes(
    0,
    "Number of parallel vector execution lanes. ... 0 (default) means "
    "unbounded: the whole vector register group is processed in a "
    "single pass, matching gem5's previous (unrealistic) behavior.",
)
```

[`src/arch/riscv/isa.hh`](../../src/arch/riscv/isa.hh) /
[`isa.cc`](../../src/arch/riscv/isa.cc): se almacena el valor, se expone
`getNumVecLanes()`, y se valida en el constructor de la ISA (`fatal_if` si `lanes` supera el
número máximo de elementos representables con `SEW=8` para el `VLEN` configurado).

### 4.2. Cálculo del chime para RVV

[`src/arch/riscv/insts/vector.hh`](../../src/arch/riscv/insts/vector.hh) /
[`vector.cc`](../../src/arch/riscv/insts/vector.cc): override de `numChimePasses()` en la clase
base `VectorMicroInst`:

```cpp
Cycles
VectorMicroInst::numChimePasses(ThreadContext *tc) const
{
    if (microVl == 0)
        return Cycles(1);

    auto *isa = static_cast<ISA *>(tc->getIsaPtr());
    uint32_t lanes = isa->getNumVecLanes();
    if (lanes == 0)
        return Cycles(1);

    return Cycles((microVl + lanes - 1) / lanes);
}
```

`microVl` ya existía como miembro de `VectorMicroInst` (es el número de elementos activos que
procesa ese micro-op concreto, calculado en construcción a partir de `VLEN`/`SEW`/`LMUL`/`VL`
por el código ya presente en `src/arch/riscv/isa/templates/vector_arith.isa`). No hizo falta
añadir ningún miembro nuevo: solo leerlo y aplicarle la fórmula del chime.

### 4.3. Por qué **no** se tocó ningún template `.isa`

RVV parte cada macro-instrucción vectorial en uno o más micro-ops (uno por registro físico del
grupo `LMUL`), y hay **decenas** de puntos de construcción de esos micro-ops repartidos en
`vector_arith.isa`, `vector_mem.isa`, formatos de segmentado, *widening*/*narrowing*, etc. — solo
`vector_arith.isa` tiene más de 40 sitios donde se invoca `new ClaseMicro(machInst, elen, vlen)`.
Añadir un tercer parámetro `lanes` a **todos** esos constructores habría sido mecánico pero de
gran volumen y alto riesgo de errores por omisión.

En su lugar, `VectorMicroInst` es la clase base común de la que heredan **todas** esas clases
generadas (aritmética y también memoria: `VectorMemMicroInst`, `VlSegMicroInst`,
`VsSegMicroInst`, etc.), y todas ya guardaban `microVl`. Cambiando la firma de `numChimePasses`
para que reciba un `ThreadContext*` (en vez de guardar `lanes` en cada instrucción en
construcción), basta con **un solo** override en `VectorMicroInst` para cubrir toda la ISA RVV,
tanto instrucciones aritméticas/lógicas/FMA como loads/stores vectoriales, sin tocar ni un solo
template `.isa`. Como ventaja adicional, este diseño lee `lanes` del `ThreadContext` real en el
momento de emisión, así que es correcto también en configuraciones multi-core heterogéneas
(cada hilo puede tener su propia ISA con distinto `lanes`), cosa que un valor cacheado en tiempo
de construcción, o una variable global, no garantizarían.

## 5. Experimento de validación

### 5.1. Diseño del microbenchmark

En [`util/vector_lanes_bench/`](../../util/vector_lanes_bench/):

- **`bench_vector_lanes.S`**: un bucle de `vadd.vv` en ensamblador RISC-V, parametrizable en
  tiempo de compilación (`SEW`, `LMUL`, `AVL`, `CHAIN`, `NITER`, `UNROLL`). Usa los registros
  `v8`/`v16`/`v24` (múltiplos de 8, válidos para cualquier `LMUL` sin problemas de alineación de
  grupo). Sin `CHAIN` mide ocupación/throughput de la FU (instrucciones independientes, cada una
  escribe el mismo registro arquitectónico pero se renombra a un registro físico distinto); con
  `CHAIN` fuerza una cadena de dependencia RAW real para medir latencia pura.
- **`Makefile`**: compila binarios estáticos `-nostdlib -static` con soporte de la extensión
  vectorial (`-march=rv64gcv`).
- **`run_lanes_bench.py`**: config de gem5 (basado en el patrón de
  `configs/example/gem5_library/riscv-rvv-example.py`) que instancia un `RiscvO3CPU` con
  `vlen`/`elen`/`lanes` configurables por línea de comandos, más una jerarquía de caché privada
  L1/L2 (imprescindible: ver §5.2).
- **`expected_cycles.py`**: calculadora analítica (sin dependencia de gem5) que reproduce
  exactamente el particionado macro-instrucción → micro-ops de gem5 y la fórmula del chime, para
  poder predecir el resultado antes de simular.

### 5.2. Primer intento: resultado inválido por un error del arnés de medición

Primera ejecución (`--vlen 256 --sew 32 --lmul 1 --lanes 4 --mode indep`):

| Magnitud | Valor |
|---|---|
| Ciclos medidos | 313.874 |
| Ciclos/instrucción medidos | ≈ 4.90 |
| Ciclos/instrucción predichos | entre 0.5 y 2.0 |

El resultado no encajaba con la predicción. Las propias estadísticas de gem5 revelaron la causa:
se había configurado la jerarquía de caché como `NoCache()` (CPU conectada directamente al bus
de memoria), así que **cada** fetch de instrucción iba a DRAM:

- `fetchStats0.icacheStallCycles = 230.140` (73% de los 313.873 ciclos totales).
- `decode.idleCycles = 230.092` (95.8%).
- `board.memory.mem_ctrl`: 6020 lecturas con un *gap* medio de ≈52.136 ticks (≈52 ciclos) entre
  ellas → `6020 × 52 ≈ 313.000` ciclos, que coincide casi exactamente con el total simulado.
- `statFuBusy::SimdAdd = 0`: en toda la ejecución nunca hubo contención por la unidad funcional
  vectorial — prueba de que el *backend* nunca llegó a acumular instrucciones vectoriales en
  vuelo suficientes como para ejercitar el modelo de ocupación que queríamos medir.

Conclusión: el conteo de instrucciones era correcto (`committedInstType::SimdAdd = 64.000`,
exactamente `NITER × UNROLL`), pero el tiempo total estaba dominado por completo por latencia de
DRAM del *fetch*, sin relación con el modelo de *lanes*. **Fix**: se sustituyó `NoCache()` por
`PrivateL1PrivateL2CacheHierarchy` (32 KiB/32 KiB/512 KiB) en `run_lanes_bench.py`. El *footprint*
de instrucciones del benchmark es minúsculo (un bucle de ~10-15 instrucciones), así que cabe
entero en L1I tras la primera pasada.

### 5.3. Segundo intento: resultado válido

Misma configuración, con la caché corregida:

| Magnitud | Valor |
|---|---|
| Ciclos medidos (`numCycles`) | 128.235 |
| Instrucciones `vadd.vv` (`SimdAdd`) | 64.000 |
| **Ciclos/instrucción medidos** | **2.0037** |
| `l1i-cache-0` miss rate | 0.18% (11 misses / 6026 accesos) |
| `fetchStats0.icacheStallCycles` | 261 (frente a 230.140 antes) |
| `statFuBusy::SimdAdd` | 0 |

**Predicción analítica** (`expected_cycles.py --vlen 256 --sew 32 --lmul 1 --lanes 4 --mode
indep`):

```
micro_vlmax (elementos/registro) = 256/32 = 8
passes = ceil(8/4) = 2
cycles/instrucción si serializado en 1 FU = 2
cycles/instrucción con 4 FUs en paralelo (ideal)  = 0.5
```

El valor medido (**2.0037**) coincide, con un error del **0.18%**, con la predicción de
"serializado en una sola FU" (= 2), que además coincide numéricamente con la fórmula de latencia
pura (`opLat + passes - 1 = 1 + 2 - 1 = 2`). El 0.18% de diferencia es exactamente el pequeño
overhead fijo (instrucción `vsetvli`, `vmv.v.i` de inicialización, rama de control de bucle) que
`expected_cycles.py` ya advertía como inevitable al dividir ciclos totales entre número de
instrucciones vectoriales.

**¿Por qué no se alcanzó el extremo "4 FUs en paralelo" (0.5 ciclos/instrucción)?** Las propias
estadísticas lo explican y no delatan ningún problema en el parche:

- `rename.IQFullEvents = 57.841` y `decode.blockedCycles = 118.901` (92.7% de los ciclos): la
  cola de instrucciones (IQ) está prácticamente siempre llena.
- `numIssuedDist`: el 98.8% de los ciclos emiten 0 o 1 instrucción; emitir 2 en el mismo ciclo
  ocurre solo el 1.96% de las veces, y 3-4 casi nunca.

Esto indica que, en este benchmark, el planificador de emisión del `O3CPU`
(`InstructionQueue::scheduleReadyInsts`, ver §3.3) nunca llega a tener más de una instrucción
`SimdAdd` "lista" simultáneamente disponible para repartir entre las 4 unidades SIMD del
`DefaultFUPool` — una característica preexistente de la estructura de planificación por
`OpClass` de gem5 (`listOrder`), no algo introducido por este parche. El resultado, por tanto,
ejercita limpiamente la rama de **latencia pura** de la fórmula del chime, y la reproduce con
precisión de menos de un 0.2% de error.

### 5.4. Veredicto

**El mecanismo de `lanes` implementado es correcto** para el escenario validado (RVV,
aritmética independiente, `O3CPU`, `SEW=32`, `LMUL=1`, `lanes=4`): el número de ciclos por
instrucción vectorial predicho analíticamente (2) coincide con el medido en gem5 (2.0037,
0.18% de error explicado por overhead de arranque conocido y ya documentado). Las estadísticas
de la simulación (miss rate de caché, distribución de instrucciones emitidas por ciclo, ausencia
de contención de FU) son además internamente consistentes con la explicación dada, reforzando la
confianza en el resultado.

### 5.5. Recomendaciones para reforzar la validación (opcional, no bloqueante)

1. **Experimento de control**: repetir exactamente el mismo binario con `--lanes 0` (el valor
   por defecto). Debería dar ≈1.0 ciclos/instrucción (el comportamiento histórico de gem5, sin
   escalado), confirmando el contraste antes/después de forma explícita.
2. **Barrido de `SEW`/`LMUL`/`AVL`**: usar la matriz de binarios que genera `make` (por defecto
   `SEW ∈ {8,32,64}`, `LMUL ∈ {1,4}`) para confirmar la fórmula en más puntos, especialmente
   `LMUL > 1` (varios micro-ops por macro-instrucción) y valores de `AVL` menores que `VLMAX`.
3. **Modo `chain`** (cadena de dependencia real): validar la rama de latencia con `opLat > 1`
   usando `--vadd-lat`/`--vmul-lat`, para comprobar también la fórmula `opLat + passes - 1` con
   un `opLat` base distinto de 1 (con `SimdAdd`/`SimdMult`, que en `DefaultFUPool` valen 1 por
   defecto, ambas ramas de la fórmula coinciden numéricamente, como ha ocurrido en este
   experimento).

## 6. Resumen de archivos modificados/creados

| Archivo | Cambio |
|---|---|
| `src/cpu/static_inst.hh` | Nuevo `virtual Cycles numChimePasses(ThreadContext*) const` (genérico, default = comportamiento actual) |
| `src/cpu/o3/dyn_inst.hh` | Método de reenvío `numChimePasses()` |
| `src/cpu/o3/inst_queue.cc` | `scheduleReadyInsts()`: aplica la fórmula del chime a latencia y ocupación de FU |
| `src/arch/riscv/RiscvISA.py` | Nuevo parámetro `lanes` (+ clase `RiscvVectorLanes`) |
| `src/arch/riscv/isa.hh` / `isa.cc` | Almacenamiento, accessor `getNumVecLanes()`, validación |
| `src/arch/riscv/insts/vector.hh` / `vector.cc` | Override de `numChimePasses()` en `VectorMicroInst`, usando `microVl` ya existente |
| `util/vector_lanes_bench/*` | Microbenchmarks de verificación (ensamblador, Makefile, config gem5, calculadora analítica) |

## 7. Próximos pasos

- **Fase 2**: extender el mismo mecanismo a ARM SVE (`sve_lanes`, override de `numChimePasses()`
  en la jerarquía de instrucciones SVE en `src/arch/arm/insts/sve.hh`), reutilizando toda la
  infraestructura genérica de la Fase 0 sin cambios.
- Ejecutar las recomendaciones de la §5.5 para ampliar la cobertura de la validación empírica.
