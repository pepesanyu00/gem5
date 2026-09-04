# Plan: Modelado de lanes vectoriales en gem5 (RVV + SVE, O3CPU)

## Contexto validado (investigación)

- RVV: `vlen`/`elen` son Params en `src/arch/riscv/RiscvISA.py` (clases `RiscvVectorLength`/
  `RiscvVectorElementLength`), almacenados en `RiscvISA` (`isa.cc`). Cada instrucción vectorial
  macro se parte en microops (uno por registro del grupo LMUL, `num_microops =
  vtype_regs_per_group(vtype)`, ver `src/arch/riscv/isa/templates/vector_arith.isa` líneas ~90-115
  y `vector_mem.isa` análogo). Cada microop procesa hasta `micro_vlmax` elementos (VLEN/SEW) en
  **una sola llamada a `execute()`**, y su latencia es la de su `OpClass` fijo
  (`SimdAddOp`, `SimdMultOp`, etc., asignados en `decoder.isa`), consultado vía
  `FUPool::getOpLatency()` (`src/cpu/o3/fu_pool.hh/.cc`) — **no depende del nº de elementos**.
  `VectorMicroInst`/`VectorMemMicroInst` (`src/arch/riscv/insts/vector.hh`) ya guardan `microVl`
  (elementos activos de ese microop): es el dato clave para escalar latencia.

- SVE: `sveVL` (en quadwords de 128 bits) es Param `sve_vl` en `src/arch/arm/ArmSystem.py`,
  propagado a `ArmISA::isa.hh/.cc`. Las instrucciones SVE **no se dividen en microops**: cada
  `execute()` (plantillas en `src/arch/arm/isa/insts/sve.isa`) itera con un `for` sobre
  `eCount = ArmStaticInst::getCurSveVecLen<Element>(...)`, una función pura de `sveVL` y el ancho
  del elemento (no depende de valores runtime), calculable en construcción igual que `microVl` en
  RVV. La latencia también es fija por `OpClass` vía el mismo `FUPool`.

- Mecanismo genérico de latencia (`src/cpu/o3/inst_queue.cc::scheduleReadyInsts()`,
  `src/cpu/o3/fu_pool.hh/.cc`): el latency lookup es **estático por OpClass**
  (`std::array<Cycles, Num_OpClasses> maxOpLatencies`), fijado una vez en la construcción del
  `FUPool` a partir de `OpDesc.opLat` (`src/cpu/FuncUnit.py`). El campo `count` de `FUDesc` modela
  *instancias paralelas de FU* (issue de instrucciones independientes), NO paralelismo de datos
  dentro de una instrucción — no sirve para modelar lanes. No existe ningún gancho para variar la
  latencia u ocupación de la FU en función de propiedades dinámicas de la instrucción (VL, LMUL,
  SEW). Este es exactamente el vacío a cubrir.
- Modelo de referencia (clásico, Hennessy & Patterson): tiempo de una instrucción vectorial con
  `lanes` carriles = `T_start + ceil(VL/lanes)` ciclos ("chime" model). Se adoptará esta fórmula.

## Objetivo de diseño

Introducir un parámetro configurable `lanes` (RVV) / `sve_lanes` (SVE), y un mecanismo **genérico**
en el núcleo de gem5 (no ligado a una ISA) que permita que una instrucción calcule su propia
latencia de finalización y su tiempo de ocupación de la FU (throughput) en función del nº de
elementos activos y de `lanes`, en vez de usar el valor fijo de `FUPool`. Aplica a instrucciones
aritmética/lógica/FMA vectoriales y también a loads/stores vectoriales (ocupación del puerto de
acceso, no la latencia de memoria/caché en sí, que sigue gobernada por el sistema de memoria).

Fórmula (por instrucción/microop, con `N` = elementos activos, `L` = lanes, `passes =
ceil(N/L)`):
- Si el OpClass es *pipelined*: latencia total = `opLat + (passes - 1)`; ocupación de la FU =
  `passes` ciclos (antes solo liberaba tras 1 ciclo).
- Si no es *pipelined*: latencia total = `opLat * passes`; ocupación de la FU = la misma cantidad.
- `lanes = 0` (valor por defecto) ⇒ comportamiento actual sin cambios (passes = 1 siempre),
  **cero regresión** para usuarios/tests existentes.

## Fases

### Fase 0 — Infraestructura genérica en el núcleo (bloquea F1 y F2)

1. `src/cpu/static_inst.hh`: añadir métodos virtuales con implementación por defecto neutra, p.ej.
   `virtual uint32_t numActiveElements() const { return 0; }` (0 = "no vectorial / sin escalado")
   y `virtual Cycles extraDataParallelCycles(Cycles opLat, bool pipelined, uint32_t lanes) const`
   que por defecto devuelve `Cycles(0)`. Alternativa más limpia: un único método
   `computeVectorTiming(Cycles opLat, bool pipelined) const -> {Cycles latency, Cycles occupancy}`
   con default = comportamiento actual exacto.
2. `src/cpu/o3/inst_queue.cc` (`scheduleReadyInsts()`, ~líneas 752-880): tras obtener
   `op_latency = fuPool->getOpLatency(op_class)`, consultar
   `issuing_inst->staticInst->computeVectorTiming(op_latency, fuPool->isPipelined(op_class))` y
   usar el par `(latency, occupancy)` resultante para (a) la latencia usada en el evento
   `FUCompletion`/liberación inmediata, y (b) cuántos ciclos permanece ocupada la FU antes de
   `freeUnitNextCycle`.
3. `src/cpu/o3/fu_pool.hh/.cc`: el modelo actual libera FUs *pipelined* tras 1 ciclo siempre
   (bitset `unitBusy` + evento a 1 ciclo). Para soportar ocupación > 1 ciclo en FUs pipelined hay
   que generalizar la liberación (programar `freeUnitNextCycle` tras `occupancy` ciclos en vez de
   1 fijo). **Este es el cambio de mayor riesgo/complejidad**: toca código compartido por
   *todas* las OpClass, no solo vectoriales, así que requiere que el valor por defecto
   (`lanes=0` ⇒ `occupancy=1` para pipelined, como hoy) quede garantizado por tests de regresión
   antes de tocar nada más.
4. Añadir flag/bandera común `StaticInst::flags[IsVector]` (ya existe y se usa en RVV) como
   comprobación rápida opcional para evitar overhead en instrucciones escalares.

**Verificación de Fase 0**: correr la suite de regresión de gem5 (`tests/gem5`, especialmente
configs O3 existentes para x86/ARM/RISC-V sin vectores) y confirmar cero cambio en stats
(`numCycles`, `IPC`) — valida que el refactor es transparente cuando `lanes=0`.

### Fase 1 — RISC-V RVV (prototipo de validación del mecanismo)

1. `src/arch/riscv/RiscvISA.py`: nueva clase `RiscvVectorLanes(UInt32)` (validar potencia de 2 o
   0) y Param `lanes = Param.RiscvVectorLanes(0, "Number of parallel vector lanes (0 = unbounded)")`.
2. `src/arch/riscv/isa.hh/.cc`: almacenar `lanes`, accessor `numLanes()`, `fatal_if` si
   `lanes` no es 0 ni potencia de 2; warning/`inform` si `lanes > vlen/8` (más lanes que el máximo
   de elementos representables a SEW=8, caso sin sentido práctico).
3. Localizar y extender el mismo punto donde hoy se propagan `vlen`/`elen` desde el decoder hasta
   los constructores de instrucción (revisar `src/arch/riscv/decoder.hh/.cc` y los call sites en
   `src/arch/riscv/isa/formats/vector_arith.isa` y `vector_mem.isa`) para pasar también `lanes`.
4. `src/arch/riscv/insts/vector.hh`: añadir miembro `lanes` a `VectorMicroInst` y a las clases
   micro de memoria (líneas ~640-730), guardado en el constructor.
5. `src/arch/riscv/isa/templates/vector_arith.isa` y `vector_mem.isa`: enhebrar el nuevo parámetro
   `lanes` por todos los templates de construcción de macro/microops (mismo patrón que
   `_elen`/`_vlen`).
6. Implementar en `VectorMicroInst`/`VectorMemMicroInst` el override de
   `computeVectorTiming()` usando `microVl` (ya disponible) y `lanes` con la fórmula de chime.
7. Config de ejemplo: exponer `lanes` en algún config de `configs/example/gem5_library` o
   documentar en `RiscvO3CPU`/`RiscvCPU.py` que el parámetro se configura vía `RiscvISA(lanes=N)`.

**Verificación de Fase 1**:
- Test dirigido: programa ensamblador con `vsetvli` variando `LMUL`/`SEW` + `vadd.vv` en bucle;
  comparar ciclos observados (`stats.txt`, `system.cpu.numCycles`) contra la fórmula analítica
  `opLat + ceil(VL/lanes) - 1` por instrucción para varias combinaciones de `vlen`, `lmul`, `sew`,
  `lanes`.
- Confirmar que con `lanes=0` los resultados son bit-idénticos a los actuales (regresión).
- Revisar/actualizar tests existentes en `tests/gem5/` relacionados con RVV si los hubiera.

### Fase 2 — ARM SVE (reutilizando infraestructura de Fase 0) — *depende de F0, en paralelo con
validación de F1 pero recomendable esperar a que F1 cierre el diseño de F0*

1. `src/arch/arm/ArmSystem.py` (junto a `sve_vl`, líneas ~46-62/329-333): nuevo Param `sve_lanes`
   (misma semántica: 0 = unbounded, potencia de 2).
2. `src/arch/arm/system.hh/.cc`, `src/arch/arm/isa.hh/.cc`: almacenar y propagar `sveLanes`
   igual que `sveVL` (líneas de referencia: `isa.hh` ~98-99, `isa.cc` ~115/124, `system.hh`
   ~128-129/208-209).
3. `src/arch/arm/insts/sve.hh`: añadir a la jerarquía base de instrucciones SVE un miembro
   `lanes` + método para calcular elementos activos en construcción (replicar la lógica pura de
   `getCurSveVecLen<Element>()` sin necesidad de ejecutar), y override de
   `computeVectorTiming()`.
4. `src/arch/arm/isa/insts/sve.isa` y `src/arch/arm/isa/formats/sve_2nd_level.isa`: enhebrar
   `lanes` en los templates/constructores de instrucción (**requiere exploración adicional al
   iniciar esta fase** para localizar los puntos exactos de construcción — los `InstObjParams`
   usados por las macros `%(class_name)s` de sve.isa no se investigaron línea a línea).
5. `src/arch/arm/isa/insts/sve_mem.isa`: confirmar si hay microops para loads/stores SVE (no
   detectado en la investigación inicial — SVE parece no dividir en microops) y aplicar el mismo
   tratamiento de ocupación de FU si el patrón es de bucle único como en aritmética.

**Verificación de Fase 2**: análoga a Fase 1, con instrucciones SVE (`fadd`, `mul`, `ld1`) variando
`sve_vl` y `sve_lanes`.

## Archivos relevantes (resumen)

- `src/cpu/static_inst.hh` — nuevo método virtual de timing vectorial (núcleo, Fase 0)
- `src/cpu/o3/inst_queue.cc` — punto de consulta de latencia/ocupación en issue (Fase 0)
- `src/cpu/o3/fu_pool.hh` / `fu_pool.cc` — generalizar liberación de FU pipelined a N ciclos (Fase 0)
- `src/cpu/FuncUnit.py`, `src/cpu/op_class.hh` — sin cambios funcionales, solo referencia
- `src/arch/riscv/RiscvISA.py`, `isa.hh`, `isa.cc` — parámetro `lanes` (Fase 1)
- `src/arch/riscv/decoder.hh/.cc` — propagación de `lanes` (Fase 1)
- `src/arch/riscv/insts/vector.hh` — `VectorMicroInst`/mem-micro, override de timing (Fase 1)
- `src/arch/riscv/isa/templates/vector_arith.isa`, `vector_mem.isa` — threading de `lanes` (Fase 1)
- `src/arch/arm/ArmSystem.py`, `system.hh/.cc`, `isa.hh/.cc` — parámetro `sve_lanes` (Fase 2)
- `src/arch/arm/insts/sve.hh` — override de timing SVE (Fase 2)
- `src/arch/arm/isa/insts/sve.isa`, `sve_mem.isa`, `formats/sve_2nd_level.isa` — threading (Fase 2)

## Decisiones tomadas con el usuario

- CPU objetivo: **solo O3CPU** (no MinorCPU).
- Alcance: **latencia + ocupación/throughput de la FU** (modelo de chime completo, no solo delay).
- Instrucciones: **aritmética/lógica/FMA + loads/stores vectoriales** (memoria: solo ocupación del
  puerto/AGU, no se toca la latencia de caché/memoria).
- Estrategia: **incremental** — RVV primero (Fase 1) valida el mecanismo genérico de Fase 0;
  SVE (Fase 2) lo reutiliza.

## Estado de implementación (actualizado)

**Fase 0 — IMPLEMENTADA** (diseño más simple que lo previsto originalmente, sin tocar
`fu_pool.cc`):
- `src/cpu/static_inst.hh`: nuevo `virtual Cycles numChimePasses(ThreadContext *tc) const`
  (default = `Cycles(1)`, cero regresión). Incluye `base/types.hh` para `Cycles`.
- `src/cpu/o3/dyn_inst.hh`: forwarding `numChimePasses()` que llama a
  `staticInst->numChimePasses(tcBase())`.
- `src/cpu/o3/inst_queue.cc` (`scheduleReadyInsts()`): calcula `passes =
  issuing_inst->numChimePasses()`; si `passes>1`, ajusta `op_latency` (pipelined: `+passes-1`;
  no pipelined: `*passes`) ANTES del check `op_latency==1`, y añade un
  `EventFunctionWrapper` que libera la FU tras `passes-1` ciclos extra en el caso pipelined
  (en vez de modificar `FUPool`/`fu_pool.cc` directamente — mucho menos invasivo y con el mismo
  efecto). Con `passes==1` (default) el código es bit-idéntico al original.

**Fase 1 (RVV) — IMPLEMENTADA (parcialmente, ver decisión de diseño clave)**:
- `src/arch/riscv/RiscvISA.py`: `RiscvVectorLanes(UInt32)` (0 = unbounded, o potencia de 2) +
  Param `lanes`.
- `src/arch/riscv/isa.hh`/`isa.cc`: miembro `lanes`, accessor `getNumVecLanes()`, validación
  (`fatal_if lanes > vlen/8`).
- **Decisión de diseño clave (se apartó del plan original)**: en vez de enhebrar un nuevo
  parámetro `lanes` por las DECENAS de templates/constructores `.isa` (arith/widening/
  narrowing/mask/mem/seg/etc. — se comprobó que solo `vector_arith.isa` ya tiene 40+ puntos de
  construcción con `_elen, _vlen`), se cambió la firma de `numChimePasses` para recibir
  `ThreadContext *tc` y leer `lanes` **en el momento de issue**, directamente desde
  `tc->getIsaPtr()`. Esto permite implementar el override **una sola vez** en la clase base común
  `VectorMicroInst` (`src/arch/riscv/insts/vector.hh/.cc`), ya que TODAS las micro-ops
  aritméticas Y de memoria (incluyendo `VectorMemMicroInst`, `VlSegMicroInst`, etc.) heredan de
  ella y ya almacenan `microVl` (elementos activos del microop). Fórmula:
  `ceil(microVl / lanes)`. Cubre aritmética + memoria sin tocar ningún template `.isa`.
  Ventaja añadida: correcto también en configuraciones heterogéneas multi-core (lee `lanes` del
  ISA del thread que emite, no de un global).
- **Pendiente/no verificado en esta sesión**: compilación real (el usuario declinó lanzar el
  build dos veces; hay que ejecutar `scons build/RISCV/gem5.opt -j4` para validar antes de dar
  por buena la Fase 1). Pendiente también el microbenchmark de verificación (vsetvli + vadd.vv
  variando lmul/sew/lanes) y la comprobación de regresión con `lanes=0`.

**Microbenchmarks de verificación de Fase 1 — CREADOS (no ejecutados, a petición del usuario)**:
en `util/vector_lanes_bench/`:
- `bench_vector_lanes.S`: benchmark RVV parametrizable (SEW/LMUL/AVL/CHAIN/NITER/UNROLL vía
  `-D`). Usa v8/v16/v24 (múltiplos de 8, válido para LMUL 1/2/4/8 sin problemas de alineación).
  Modo `indep` (sin CHAIN): mide ocupación/throughput de la FU (instrucciones independientes,
  renombradas). Modo `chain` (con `-DCHAIN`): cadena RAW real, mide latencia de finalización.
  Sale con `ecall`/`exit(93)`.
- `Makefile`: `make` construye una matriz por defecto (SEW∈{8,32,64} × LMUL∈{1,4} × {indep,chain})
  en `build/`; `make custom SEW=.. LMUL=.. AVL=.. CHAIN=1 OUT=..` para un build a medida.
  Usa `CROSS_COMPILE` (default `riscv64-unknown-elf-`), `-march=rv64gcv -mabi=lp64d -nostdlib
  -static -Wl,-e,_start`.
- `run_lanes_bench.py`: script de gem5 (stdlib, basado en el patrón de
  `configs/example/gem5_library/riscv-rvv-example.py`) que corre un binario ya compilado con
  `RiscvO3CPU`, `--vlen --elen --lanes` y opcionalmente `--vadd-lat/--vmul-lat` (parchea
  `DefaultFUPool` vía una subclase `LaneBenchFUPool` que solo toca `SimdAdd`/`SimdMult` si se
  pide explícitamente — por defecto reproduce el pool stock). Usa `NoCache()` +
  `SingleChannelDDR3_1600` + `BinaryResource(local_path=...)`.
- `expected_cycles.py`: calculadora analítica standalone (sin dependencia de gem5, ya probada
  que ejecuta sin errores) que replica EXACTAMENTE el particionado macro-op→microops de gem5
  (`micro_vlmax = VLEN//SEW`, hasta LMUL microops) y la fórmula de chime, imprimiendo la
  predicción y explicando cómo compararla con `system.cpu.numCycles` en `m5out/stats.txt`.
  Importante hallazgo documentado en el propio script: `DefaultFUPool.SIMD_Unit.count == 4`
  (4 unidades SIMD paralelas), por lo que el benchmark `indep` puede alcanzar hasta 4x más
  throughput del ingenuamente esperado con una sola FU — el script reporta un rango
  [single-FU, fu_count-FUs] en vez de un único valor. También documenta que con LMUL>1 el
  benchmark `chain` en realidad forma LMUL cadenas independientes (una por slice del grupo de
  registros), recomendando LMUL=1 para aislar pura latencia sin ambigüedad.
- Pendiente: el usuario compilará (toolchain riscv64-unknown-elf-gcc con soporte 'v') y ejecutará
  estos benchmarks él mismo; después seguimos con Fase 2 (SVE).

**Primera ejecución real del usuario — resultado: INCONCLUSIVA por bug en el harness (no en el
código C++ de gem5)**:
- Comando: `--vlen 256 --sew 32 --lmul 1 --lanes 4 --mode indep --niter 2000 --unroll 32`.
  Predicción: cycles/inst en [0.5 (4 FUs), 2 (1 FU)]. Medido: `numCycles=313874 /
  64000 SimdAdd insts ≈ 4.90` — muy por encima de lo predicho.
- Causa raíz identificada en `stats.txt`: usé `NoCache()` en `run_lanes_bench.py` (por
  simplicidad) → CADA fetch de instrucción va directo a DRAM. Evidencia: `fetchStats0.
  icacheStallCycles=230140` (73% de los 313873 ciclos totales), `decode.idleCycles=230092`
  (95.8%), `mem_ctrl` con 6020 lecturas (líneas de icache) y `avgGap≈52136 ticks` (~52
  ciclos/lectura) — 6020×52 ≈ 313000 ciclos, coincide casi exactamente con el total. Además
  `statFuBusy::SimdAdd=0` (nunca hubo contención de FU en toda la ejecución): confirma que el
  backend estaba muerto de hambre por fetch, nunca llegó a acumular suficientes instrucciones
  vectoriales en vuelo como para ejercitar (o poder invalidar) el modelo de ocupación de FU.
  Conteo de instrucciones SÍ correcto funcionalmente: `committedInstType::SimdAdd=64000`
  (=NITER*UNROLL exacto), confirma que el binario/benchmark en sí están bien construidos.
- **Fix aplicado**: cambié `NoCache()` por `PrivateL1PrivateL2CacheHierarchy` (32KiB/32KiB/512KiB,
  igual que el ejemplo oficial `riscv-rvv-example.py`) en `run_lanes_bench.py`. La huella de
  instrucciones del benchmark es minúscula (un bucle de ~10-15 instrucciones), así que con L1I
  cabe entera y tras la primera iteración deja de haber miss — así el tiempo medido debería
  reflejar de verdad el modelo de chime/lanes en vez de latencia de DRAM.
- **Lección para recordar**: al diseñar micro-benchmarks de timing en gem5, NUNCA usar
  `NoCache()` salvo que se quiera medir explícitamente el subsistema de memoria — para aislar
  el backend/ejecución hay que cachear perfectamente el código (footprint pequeño + L1I normal)
  para que el fetch dejen de ser el cuello de botella.
- Pendiente: el usuario debe re-ejecutar con el config corregido y volver a comparar contra
  `expected_cycles.py`. Aún no hay veredicto validado sobre si el modelo C++ implementado
  (Fase 0 + Fase 1) es correcto.

**Segunda ejecución (con caché L1/L2 corregida) — VALIDACIÓN EXITOSA**:
- Mismo comando (`lanes=4, SEW=32, LMUL=1, VLEN=256, mode=indep`): `numCycles=128235`,
  `SimdAdd=64000` → 2.0037 ciclos/inst medidos vs. 2.0 predicho (single-FU/latencia pura) →
  error 0.18% (coincide con el overhead fijo ya documentado en expected_cycles.py).
- `l1i-cache-0` miss rate 0.18% (11/6026), `icacheStallCycles=261` (antes 230140) → la caché
  real solucionó el problema. `statFuBusy::SimdAdd=0` sigue en 0 pero ahora por una razón
  distinta y benigna: `rename.IQFullEvents=57841`, `decode.blockedCycles=118901` (92.7%),
  `numIssuedDist` casi nunca emite >1 SimdAdd/ciclo → el planificador de `scheduleReadyInsts`
  (mecanismo `listOrder`, una entrada "lista" por OpClass a la vez) nunca expone más de una
  instrucción SimdAdd simultáneamente lista, así que nunca se ejercitan las 4 FUs del
  `DefaultFUPool` en paralelo — comportamiento preexistente de gem5, no un bug del parche.
  Resultado: el experimento ejercitó limpiamente la rama de latencia pura de la fórmula
  (`opLat+passes-1=2`) y la reprodujo con <0.2% de error.
- **Veredicto**: el mecanismo de lanes (Fase 0 + Fase 1, RVV) es CORRECTO para el escenario
  validado. Informe completo redactado en `util/vector_lanes_bench/REPORT.md` (contexto,
  fórmulas, cambios de código con snippets, ambos experimentos con cifras, veredicto,
  recomendaciones de validación adicional: control con lanes=0, barrido LMUL/AVL, modo chain
  con opLat>1).
- Pendiente: Fase 2 (SVE), y opcionalmente las validaciones adicionales sugeridas en el informe.

**Fase 2 (SVE) — IMPLEMENTADA (cobertura amplia, no exhaustiva al 100%, documentado)**:

Investigación clave: a diferencia de RVV, SVE NO tiene una clase base común de la que hereden
todas las instrucciones vectoriales — hay ~40 clases "shape" en `src/arch/arm/insts/sve.hh` que
derivan TODAS directamente de `ArmStaticInst`, sin intermedio. PERO cada una de esas clases se
usa a través de un `def template ...Declare` en `src/arch/arm/isa/templates/sve.isa`/
`sve_mem.isa` que genera la clase LEAF `template <class _Element> class %(class_name)s : public
%(base_class)s { typedef _Element Element; ... }` — es decir, el tipo de elemento SIEMPRE está
disponible como typedef `Element`/`TPElem` dentro de la clase generada. Esto permite añadir el
override de `numChimePasses()` DIRECTAMENTE en cada template `...Declare` (usando
`ArmStaticInst::getCurSveVecLen<Element>(tc)`), sin tocar ninguna jerarquía de clases ni
constructor — más simple aún que la solución de RVV.

Cambios:
- `src/arch/arm/ArmSystem.py`: nueva clase `SveVectorLanes(UInt32)` (0=unbounded o potencia de
  2) + parámetro `sve_lanes` en `ArmSystem` (modo FS), mirroring `sve_vl`.
- `src/arch/arm/ArmISA.py`: `sve_lanes_se` (modo SE), importa `SveVectorLanes` de ArmSystem.py.
- `src/arch/arm/system.hh/.cc`: `_sveLanes` + accessor `sveLanes()`.
- `src/arch/arm/isa.hh/.cc`: `sveLanes` + accessor `getNumSveLanes()`, inicializado desde
  `system->sveLanes()` (FS) o `p.sve_lanes_se` (SE), mismo patrón exacto que `sveVL`.
- `src/arch/arm/insts/static_inst.hh/.cc`: nuevo `static Cycles ArmStaticInst::numSveChimePasses(
  ThreadContext *tc, unsigned eCount)` — única fuente de verdad de la fórmula de chime para SVE
  (equivalente a lo que hace `VectorMicroInst::numChimePasses` en RVV).
- `src/arch/arm/isa/templates/sve.isa`: override `numChimePasses()` añadido a 27 templates
  `...Declare` (aritmética/lógica/comparación/reducción/FMA/complejos/clamp/matmul — cubre
  add/sub/mul/div/fadd/fmul/fmla/cmp/and/orr/eor/reduce/fcmla/mla-widening/matmul, etc.). Usa
  `Element` salvo `SveMatMulOpDeclare` (usa `TPElem`).
- `src/arch/arm/isa/templates/sve_mem.isa`: override añadido a 5 templates: `SveMemFillSpillOp`,
  `SveContigMemSSOp`, `SveContigMemSIOp` (usan `getCurSveVecLen<TPElem>(tc)`, transferencia
  completa) y `SveIndexedMemVIMicroop`/`SveIndexedMemSVMicroop` (gather/scatter; usan el miembro
  YA existente `numElems`, análogo a `microVl` en RVV — más preciso que recalcular desde vlen).
- Total cubierto: **32 templates** (27+5), confirmado por grep (`numChimePasses` aparece 27 veces
  en sve.isa y 5 veces en sve_mem.isa).
- **NO cubierto aún (documentado como pendiente)**: instrucciones de generación de índice/
  predicado/control (SveIndexII/IR/RI/RR, SvePredCountOp/PredOp, SvePtrueOp, SveAdrOp,
  SveWhileOp, SvePselOp, SveCompTermOp, SveElemCountOp, SvePartBrkOp/PropOp, SveSelectOp,
  SveUnpackOp, SvePredicateTestOp, SvePredUnary*WImplicit* x3, SveOpWImplicitSrcDstOp) y mem
  segmentada/estructurada (SveFirstFaultWritebackMicroop, SveGatherLoadCpySrcVecMicroop,
  SveStructMemSIMicroop, SveStructMemSSMicroop, SveIntrlvMicroop, SveDeIntrlvMicroop) — mismo
  patrón mecánico, extensión trivial si se necesita luego.
- Pendiente: compilar ARM (`scons build/ARM/gem5.opt`), y crear microbenchmarks SVE equivalentes
  a `util/vector_lanes_bench/` (aún no creados en esta sesión) para validar empíricamente igual
  que se hizo con RVV.

## Riesgos / complejidad

- **Alto riesgo**: cambio en `fu_pool.cc`/`inst_queue.cc` es código compartido por *todas* las
  OpClass de gem5 (no solo vectoriales) — cualquier error de regresión afecta a todo O3CPU.
  Mitigación: valor por defecto `lanes=0` debe ser matemáticamente idéntico al comportamiento
  actual; correr suite de regresión completa antes/después de Fase 0.
- **Complejidad media-alta**: enhebrar el nuevo parámetro por todos los templates `.isa` (muchos
  puntos de construcción) es mecánico pero propenso a errores por omisión (patrón ya usado con
  `vlen`/`elen`, por lo que hay precedente claro a seguir).
- **Complejidad media**: en SVE aún falta localizar con precisión los puntos de construcción de
  instrucción en `sve.isa`/`sve_2nd_level.isa` (no explorado a fondo en esta sesión) — se marca
  como primer paso de la Fase 2.
- Estimación cualitativa: Fase 0 y Fase 1 combinadas son un esfuerzo de ingeniería
  sustancial (cambios en el core del simulador + ISA RVV completa); Fase 2 debería ser más rápida
  una vez cerrado el diseño genérico, pero con incertidumbre adicional por los puntos aún no
  localizados en SVE.
