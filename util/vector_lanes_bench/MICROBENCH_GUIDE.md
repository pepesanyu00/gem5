# Guía de los Microbenchmarks de Vector Lanes

## Índice

1. [¿Qué estamos midiendo?](#1-qué-estamos-midiendo)
2. [Dos modos de ejecución: `indep` y `chain`](#2-dos-modos-de-ejecución-indep-y-chain)
3. [Cómo funciona cada benchmark por dentro](#3-cómo-funciona-cada-benchmark-por-dentro)
4. [El predictor analítico: `expected_cycles`](#4-el-predictor-analítico-expected_cycles)
5. [Configuración de parámetros](#5-configuración-de-parámetros)
6. [Cómo compilar y ejecutar paso a paso](#6-cómo-compilar-y-ejecutar-paso-a-paso)
7. [Cómo comprobar los resultados](#7-cómo-comprobar-los-resultados)
8. [Referencia rápida de archivos](#8-referencia-rápida-de-archivos)

---

## 1. ¿Qué estamos midiendo?

Un procesador vectorial tiene un registro vectorial de `VLEN` bits de ancho. Si el
ancho de elemento es `SEW` bits (por ejemplo, 32 para enteros de 32 bits), entonces
cada registro contiene `VLEN / SEW` elementos. En nuestro caso con VLEN=256 y
SEW=32, eso son **8 elementos**.

El parámetro `lanes` indica cuántos de esos elementos puede procesar el hardware
**en paralelo en un solo ciclo**. Si `lanes = 4`, el procesador necesita
`ceil(8/4) = 2` ciclos para procesar los 8 elementos de una instrucción.
Esos ciclos se llaman **pasadas** o **chimes** (terminología de Hennessy & Patterson).

```
passes = ceil(elementos_activos / lanes)
```

El propósito de estos microbenchmarks es **verificar empíricamente** que gem5
aplica este cálculo correctamente, midiendo dos cosas:

1. **Latencia de completación**: cuánto tarda una instrucción vectorial en producir
   su resultado (importante cuando hay cadenas de dependencias).
2. **Ocupación de la FU (throughput)**: cuánto tiempo ocupa la unidad funcional
   una instrucción (importante cuando las instrucciones son independientes).

---

## 2. Dos modos de ejecución: `indep` y `chain`

Cada benchmark se compila en dos variantes que aíslan cada uno de esos dos
aspectos. La diferencia es **una sola línea de ensamblador**:

### Modo `indep` (independiente) — mide throughput

```asm
.rept 32
    vadd.vv v24, v8, v16      @ z0 = z1 + z2 en SVE
.endr
```

Cada instrucción lee `v8` y `v16` (que nunca cambian) y escribe en `v24`. No hay
dependencia de datos real entre instrucciones consecutivas: cada una sobreescribe
`v24` pero la siguiente no lee el valor que acaba de escribir. El procesador O3
de gem5 usa **register renaming** para eliminar la falsa dependencia WAW
(Write-After-Write), así que todas las instrucciones pueden **emitirse en paralelo**
a las unidades funcionales disponibles.

**¿Qué limita el rendimiento?** Únicamente la **ocupación de la FU**. Si cada
instrucción ocupa la FU durante `passes` ciclos, y hay `FU_COUNT` unidades
funcionales idénticas, el máximo throughput es:

```
CPI_indep = passes / FU_COUNT
```

### Modo `chain` (cadena de dependencias) — mide latencia

```asm
.rept 32
    vadd.vv v24, v24, v16     @ z0 = z0 + z2 en SVE
.endr
```

Aquí el destino (`v24`/`z0`) es también uno de los operandos fuente. Cada
instrucción **necesita el resultado de la anterior** para empezar — es una
cadena RAW (Read-After-Write) perfecta. Esto serializa completamente la
ejecución: da igual cuántas FUs haya, solo se puede ejecutar una instrucción
a la vez.

**¿Qué limita el rendimiento?** Únicamente la **latencia de completación**.
Para una FU pipelined con latencia base `opLat`:

```
CPI_chain = opLat + passes - 1
```

Con `opLat = 1` (valor por defecto de `SimdAdd` en gem5), esto se simplifica a:

```
CPI_chain = passes
```

### Resumen visual

```
              INDEP (independiente)                 CHAIN (cadena)
         ┌──────────────────────────┐       ┌──────────────────────────┐
         │ vadd v24, v8,  v16  ─┐  │       │ vadd v24, v24, v16  ─┐  │
         │ vadd v24, v8,  v16  ─┤  │       │                      │  │
         │ vadd v24, v8,  v16  ─┤  │       │ vadd v24, v24, v16  ─┤  │
         │ vadd v24, v8,  v16  ─┘  │       │                      │  │
         │ (todas van en paralelo) │       │ vadd v24, v24, v16  ─┘  │
         │ Bottleneck: FU occup.   │       │ (una tras otra)         │
         └──────────────────────────┘       │ Bottleneck: latencia    │
                                            └──────────────────────────┘
```

### ¿Por qué necesitamos ambos modos?

El modelo de chime afecta a dos cosas distintas en el simulador:

| Aspecto | Dónde se aplica en gem5 | Cómo lo verificamos |
|---|---|---|
| Latencia de completación | `op_latency = opLat + passes - 1` | Modo **chain** |
| Ocupación de la FU | `EventFunctionWrapper` que retrasa `freeUnitNextCycle` | Modo **indep** |

Si solo usáramos `chain`, un bug en la ocupación pasaría desapercibido (y viceversa).

---

## 3. Cómo funciona cada benchmark por dentro

### 3.1. RISC-V (`bench_vector_lanes.S`)

```asm
_start:
    vsetvli t0, x0, e32, m1, ta, ma   @ Configura: SEW=32, LMUL=1, vl=VLMAX
    vmv.v.i v8, 1                      @ v8  = {1, 1, 1, ...}  (fuente A)
    vmv.v.i v16, 2                     @ v16 = {2, 2, 2, ...}  (fuente B)

    li s0, 2000                        @ Contador de iteraciones
loop:
    .rept 32                           @ 32 instrucciones desenrolladas
        vadd.vv v24, v8, v16           @ (o v24, v24, v16 en modo chain)
    .endr
    addi s0, s0, -1
    bnez s0, loop

    li a7, 93                          @ syscall exit(0)
    li a0, 0
    ecall
```

**Particularidades RVV:**

- `vsetvli` configura la longitud vectorial. Los parámetros `SEW` y `LMUL` se
  pasan como `-D` al compilador (`-DSEW=32 -DLMUL=1`).
- Con `LMUL > 1`, cada macro-instrucción `vadd.vv` se descompone internamente
  en **LMUL micro-operaciones**, cada una procesando un registro del grupo.
  Por ejemplo, con LMUL=4, un `vadd.vv v8, v16, v24` genera 4 micro-ops:
  una para v8, otra para v9, v10 y v11.
- Los registros v8-v15, v16-v23, v24-v31 se usan como grupos para soportar
  hasta LMUL=8 sin conflictos de alineamiento.

### 3.2. ARM SVE (`bench_sve_lanes.S`)

```asm
_start:
    ptrue p0.s                         @ Predicado todo-activo (elem 32-bit)
    dup   z1.s, #1                     @ z1 = {1, 1, 1, ...}  (fuente A)
    dup   z2.s, #2                     @ z2 = {2, 2, 2, ...}  (fuente B)

    mov   x0, #2000                    @ Contador de iteraciones
.Lloop:
    .rept 32                           @ 32 instrucciones desenrolladas
        add z0.s, z1.s, z2.s           @ (o z0.s, z0.s, z2.s en modo chain)
    .endr
    subs x0, x0, #1
    b.ne .Lloop

    mov x8, #93                        @ syscall exit(0)
    mov x0, #0
    svc #0
```

**Particularidades SVE:**

- SVE **no tiene LMUL**. Cada instrucción opera sobre un único registro
  vectorial completo. El número de elementos depende solo de `VL` y del
  ancho de elemento.
- La longitud vectorial se configura en gem5 con el parámetro `sve_vl`
  (en cuadwords de 128 bits). `sve_vl=2` → VL = 256 bits.
- El sufijo `.s` (32 bits), `.b` (8 bits), `.h` (16 bits) o `.d` (64 bits)
  determina el ancho de elemento, controlado por `-DELEM_WIDTH=32` en
  compilación.

### 3.3. Estructura del loop

Ambos benchmarks comparten la misma estructura:

```
Total instrucciones vectoriales = NITER × UNROLL = 2000 × 32 = 64000
```

El loop se desenrolla (`UNROLL=32`) para minimizar el impacto del overhead
del control de flujo (`addi/subs` + `bnez/b.ne`). De las 34 instrucciones
por iteración, 32 son vectoriales y 2 son escalares de control, así que el
overhead escalar es solo un ~6% del total, y se reduce aún más aumentando
`UNROLL` o `NITER`.

---

## 4. El predictor analítico: `expected_cycles`

Antes de ejecutar gem5, calculamos el CPI esperado analíticamente con dos
scripts Python independientes (no necesitan gem5):

### 4.1. RISC-V: `expected_cycles.py`

```bash
python3 expected_cycles.py --vlen 256 --sew 32 --lmul 1 --lanes 4 --mode chain
```

Calcula:
1. `micro_vlmax = VLEN / SEW = 256/32 = 8` elementos por micro-op
2. `passes = ceil(8 / 4) = 2`
3. Modo chain: `CPI = opLat + passes - 1 = 1 + 2 - 1 = 2`
4. Total ciclos ≈ 64000 × 2 = 128000

### 4.2. SVE: `expected_cycles_sve.py`

```bash
python3 expected_cycles_sve.py --sve-vl 2 --elem-width 32 --lanes 4 --mode chain
```

Calcula:
1. `eCount = (2 × 128) / 32 = 8` elementos
2. `passes = ceil(8 / 4) = 2`
3. Modo chain: `CPI = 1 + 2 - 1 = 2`
4. Total ciclos ≈ 64000 × 2 = 128000

### 4.3. ¿Por qué coinciden?

RISC-V con `VLEN=256, SEW=32, LMUL=1` y SVE con `sve_vl=2, elem_width=32`
son configuraciones **equivalentes**: ambos procesan 8 elementos de 32 bits
en un registro de 256 bits. Con `lanes=4`, ambos necesitan 2 pasadas. El
modelo matemático es el mismo; lo que cambia es cómo cada ISA llega al
conteo de elementos activos internamente.

---

## 5. Configuración de parámetros

### 5.1. Parámetros de compilación del benchmark (Makefile)

Se pasan como `-D` al ensamblador:

| Parámetro | Valores | Default | Descripción |
|---|---|---|---|
| `SEW` (RISC-V) | 8, 16, 32, 64 | 32 | Ancho de elemento en bits |
| `LMUL` (RISC-V) | 1, 2, 4, 8 | 1 | Multiplicador de grupo de registros |
| `ELEM_WIDTH` (SVE) | 8, 16, 32, 64 | 32 | Ancho de elemento en bits |
| `CHAIN` | definido / no definido | no definido | Si se define → modo chain; si no → indep |
| `AVL` (RISC-V) | entero | no definido | Longitud vectorial solicitada (si no se define, usa VLMAX) |
| `NITER` | entero | 2000 | Iteraciones del loop externo |
| `UNROLL` | entero | 32 | Instrucciones vectoriales por iteración |

### 5.2. Parámetros de simulación gem5

Se pasan por línea de comandos al script de configuración:

**RISC-V** (`run_lanes_bench.py`):

| Parámetro | Default | Descripción |
|---|---|---|
| `--vlen` | 256 | VLEN en bits |
| `--elen` | 64 | ELEN en bits |
| `--lanes` | 0 | Número de lanes (0 = sin límite, comportamiento original gem5) |
| `--vadd-lat` | 1 (stock) | Override de latencia de `SimdAdd` |
| `--binary` | (requerido) | Ruta al binario compilado |

**ARM SVE** (`run_sve_lanes_bench.py`):

| Parámetro | Default | Descripción |
|---|---|---|
| `--sve-vl` | 2 | Longitud vectorial SVE en quadwords (1=128b, 2=256b, 4=512b) |
| `--sve-lanes` | 0 | Número de lanes (0 = sin límite) |
| `--vadd-lat` | 1 (stock) | Override de latencia de `SimdAdd` |
| `--binary` | (requerido) | Ruta al binario compilado |

### 5.3. Parámetros del script maestro (`run_all_benchmarks.sh`)

| Parámetro | Default | Descripción |
|---|---|---|
| `--lanes` | 4 | Lanes para ambas ISAs |
| `--vlen` | 256 | VLEN para RISC-V |
| `--sve-vl` | 2 | SVE VL en quadwords |
| `--fu-count` | 4 | Número de FUs SIMD (para predicción multi-FU) |
| `--niter` | 2000 | Iteraciones (debe coincidir con el Makefile) |
| `--unroll` | 32 | Unroll (debe coincidir con el Makefile) |
| `--skip-build` | false | Omitir compilación de benchmarks y gem5 |
| `--riscv-only` | false | Solo ejecutar benchmarks RISC-V |
| `--arm-only` | false | Solo ejecutar benchmarks ARM SVE |
| `--dry-run` | false | Solo imprimir comandos, no ejecutar |

### 5.4. Relación entre parámetros del benchmark y de gem5

Es **fundamental** que los parámetros de compilación y de simulación sean
coherentes. El benchmark se compila con un `SEW`/`ELEM_WIDTH` fijo, y gem5 se
configura con un `VLEN`/`sve_vl` y `lanes`. La relación es:

```
RISC-V:  eCount = VLEN / SEW               passes = ceil(eCount / lanes)
SVE:     eCount = (sve_vl × 128) / EW       passes = ceil(eCount / lanes)
```

Si cambias `VLEN` en gem5 pero no recompilas el benchmark, no pasa nada: el
benchmark no sabe nada sobre VLEN. El `vsetvli` del benchmark pide VLMAX, y
gem5 responde con el VLMAX correspondiente al VLEN configurado. Pero si cambias
`SEW` o `LMUL`, sí necesitas recompilar.

---

## 6. Cómo compilar y ejecutar paso a paso

### 6.1. Prerequisitos

```bash
# 1. Compilar gem5 para RISC-V
scons build/RISCV/gem5.opt -j$(nproc)

# 2. Compilar gem5 para ARM
scons build/ARM/gem5.opt -j$(nproc)

# 3. Verificar cross-compilers
riscv64-unknown-linux-gnu-gcc --version   # RISC-V
aarch64-linux-gnu-gcc --version           # ARM
```

### 6.2. Compilar los benchmarks

```bash
cd util/vector_lanes_bench

# RISC-V: genera build/bench_sew{8,32,64}_lmul{1,4}_{indep,chain}
make

# SVE: genera build_sve/bench_sve_elem{8,32,64}_{indep,chain}
make -f Makefile_sve
```

### 6.3. Ejecución individual (para entender cada paso)

```bash
# Paso 1: Predicción analítica
python3 expected_cycles.py --vlen 256 --sew 32 --lmul 1 --lanes 4 --mode chain
# → predicted cycles/instruction = 2

# Paso 2: Simulación gem5
./build/RISCV/gem5.opt -d results/riscv_sew32_lmul1_chain \
    util/vector_lanes_bench/run_lanes_bench.py \
    --binary util/vector_lanes_bench/build/bench_sew32_lmul1_chain \
    --vlen 256 --elen 64 --lanes 4

# Paso 3: Extraer numCycles
grep numCycles results/riscv_sew32_lmul1_chain/stats.txt
# → system.cpu.numCycles    128235

# Paso 4: Calcular CPI medido
python3 -c "print(f'CPI = {128235/64000:.4f}')"
# → CPI = 2.0037
```

### 6.4. Ejecución automática (todos los benchmarks)

```bash
# Ejecutar todo (compila + simula + compara)
./util/vector_lanes_bench/run_all_benchmarks.sh

# Solo RISC-V, sin recompilar
./util/vector_lanes_bench/run_all_benchmarks.sh --riscv-only --skip-build

# Solo ARM con 8 lanes
./util/vector_lanes_bench/run_all_benchmarks.sh --arm-only --lanes 8

# Ver qué haría sin ejecutar nada
./util/vector_lanes_bench/run_all_benchmarks.sh --dry-run
```

---

## 7. Cómo comprobar los resultados

### 7.1. La métrica: Cycles Per Instruction (CPI)

```
CPI_medido = numCycles / N_VEC_INSTS
```

Donde `N_VEC_INSTS = NITER × UNROLL = 2000 × 32 = 64000`.

Este valor **sobreestima ligeramente** el CPI real porque `numCycles` incluye
un overhead fijo de inicialización (setup de vectores, `vsetvli`/`ptrue`) y
de control del loop (`addi`/`bnez`). Con 64000 instrucciones vectoriales, este
overhead es despreciable (<0.5%).

### 7.2. Interpretación de los resultados por modo

**Modo `chain`** (el más limpio para validación):

| Resultado | Significado |
|---|---|
| CPI ≈ `passes` | ✅ El modelo de chime funciona correctamente |
| CPI ≈ 1 (independiente de `passes`) | ❌ `numChimePasses` no se está llamando o devuelve 1 |
| CPI = `passes` pero distinto del predicho | ❌ El cálculo de `eCount` o `lanes` es incorrecto |

**Modo `indep`** (más complejo, depende del número de FUs):

| Resultado | Significado |
|---|---|
| CPI ≈ `passes / FU_COUNT` | ✅ Tanto latencia como ocupación FU funcionan |
| CPI ≈ `1 / FU_COUNT` (independiente de `passes`) | ⚠️ La latencia se ajusta pero la ocupación FU no |
| CPI ≈ `passes` | ⚠️ Solo se usa 1 FU (posible bottleneck de issue width) |

### 7.3. ¿Qué configura cuántos FUs SIMD hay?

El `DefaultFUPool` de gem5 tiene **4 unidades SIMD** (`SIMD_Unit.count = 4`).
Cada una puede ejecutar `SimdAdd`, `SimdMult`, etc. Por eso las instrucciones
independientes pueden repartirse entre hasta 4 FUs.

### 7.4. Tabla de la matriz completa de pruebas

Con la configuración por defecto (`VLEN=256, lanes=4, sve_vl=2`):

| ISA | Elem | LMUL | eCount | passes | chain CPI | indep CPI (4 FUs) |
|---|---|---|---|---|---|---|
| RVV | 8 | 1 | 32 | 8 | 8 | 2.0 |
| RVV | 8 | 4 | 32/uop | 8 | 8 | 8.0 (4×8/4) |
| RVV | 32 | 1 | 8 | 2 | 2 | 0.5 |
| RVV | 32 | 4 | 8/uop | 2 | 2 | 2.0 (4×2/4) |
| RVV | 64 | 1 | 4 | 1 | 1 | 0.25 |
| RVV | 64 | 4 | 4/uop | 1 | 1 | 1.0 (4×1/4) |
| SVE | 8 | — | 32 | 8 | 8 | 2.0 |
| SVE | 32 | — | 8 | 2 | 2 | 0.5 |
| SVE | 64 | — | 4 | 1 | 1 | 0.25 |

### 7.5. El script genera una tabla resumen automática

Al final de `run_all_benchmarks.sh`, se imprime una tabla con:

```
  Benchmark                     Pred.CPI  numCycles   Meas.CPI   Error    Status
  ────────────────────────────  ────────  ──────────  ────────  ────────  ──────
  riscv_sew32_lmul1_chain            2      128235    2.0037     0.18%      OK
  sve_elem32_chain                   2      128180    2.0028     0.14%      OK
  ...
```

Los umbrales de status son:
- **OK** (verde): error < 1%
- **OK** (amarillo): error 1-5%
- **CHECK** (rojo): error ≥ 5% — requiere investigación

---

## 8. Referencia rápida de archivos

| Archivo | Tipo | Descripción |
|---|---|---|
| `bench_vector_lanes.S` | Ensamblador RISC-V | Benchmark RVV (fuente) |
| `bench_sve_lanes.S` | Ensamblador AArch64 | Benchmark SVE (fuente) |
| `Makefile` | Build | Compila variantes RVV → `build/` |
| `Makefile_sve` | Build | Compila variantes SVE → `build_sve/` |
| `run_lanes_bench.py` | Config gem5 | Script de configuración para simular RVV |
| `run_sve_lanes_bench.py` | Config gem5 | Script de configuración para simular SVE |
| `expected_cycles.py` | Predictor | Calcula CPI analítico para RVV |
| `expected_cycles_sve.py` | Predictor | Calcula CPI analítico para SVE |
| `run_all_benchmarks.sh` | Orquestador | Compila, ejecuta y compara todo automáticamente |
| `results/` | Resultados | Carpetas de salida gem5, una por benchmark |
| `REPORT.md` | Documentación | Informe detallado de la implementación RVV |
| `plan.md` | Documentación | Plan de implementación (Fases 0, 1, 2) |
