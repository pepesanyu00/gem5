/*
 * Microbenchmark to validate gem5's RVV "vector lanes" chime timing model
 * (RiscvISA::lanes, src/arch/riscv/insts/vector.cc:VectorMicroInst::
 * numChimePasses, src/cpu/o3/inst_queue.cc:scheduleReadyInsts).
 *
 * The program executes a tight loop of `vadd.vv` instructions, configured
 * once via `vsetvli` for a given element width (SEW) and register-group
 * multiplier (LMUL), and then exits via the exit(2) syscall. It is meant
 * to be compiled into several variants (see Makefile) that isolate either:
 *
 *   - FU OCCUPANCY / THROUGHPUT: instructions are independent (no true
 *     data dependency between them), so back-to-back issue is limited
 *     only by how long each vadd occupies the vector ALU.
 *
 *   - COMPLETION LATENCY: instructions form a true RAW dependency chain
 *     (each vadd consumes the previous one's result), so back-to-back
 *     issue is limited by how long it takes for a single vadd's result
 *     to become available.
 *
 * Both variants report (indirectly, via gem5's system.cpu.numCycles stat)
 * the steady-state number of cycles per `vadd.vv`, which should match the
 * analytical prediction computed by expected_cycles.py:
 *
 *   passes  = ceil(active_elements_per_microop / lanes)     (lanes=0 -> 1)
 *   independent (this file built without CHAIN): cycles/vadd ~= passes
 *   dependent   (this file built with    CHAIN): cycles/vadd ~= opLat + passes - 1
 *
 * where active_elements_per_microop = min(vl, VLEN / SEW) -- RVV splits a
 * vadd.vv into one microop per physical vector register in the LMUL group,
 * and each microop can hold at most VLEN/SEW elements (see
 * src/arch/riscv/isa/templates/vector_arith.isa and
 * src/arch/riscv/utility.hh:vtype_VLMAX(..., per_reg=true)).
 * With LMUL > 1, a single `vadd.vv` becomes LMUL microops that share the
 * same functional unit sequentially, so it contributes LMUL * (cycles
 * predicted above) to the total, not just one microop's worth -- see
 * expected_cycles.py, which accounts for this.
 *
 * Build-time parameters (see Makefile, passed via -D to the assembler):
 *   SEW    - element width in bits: 8, 16, 32 or 64 (default 32)
 *   LMUL   - register-group multiplier: 1, 2, 4 or 8 (default 1)
 *            (fractional LMUL, mf2/mf4/mf8, is not supported by this
 *            simple benchmark)
 *   AVL    - if defined, requested vector length in elements; vl is set
 *            to min(AVL, VLMAX). If undefined, vl = VLMAX (the whole
 *            register group is active every instruction).
 *   CHAIN  - if defined, instructions form the RAW dependency chain
 *            described above (measures completion LATENCY). If
 *            undefined, instructions are independent (measures FU
 *            OCCUPANCY / throughput).
 *   NITER  - number of loop iterations (default 2000)
 *   UNROLL - number of vadd.vv per loop iteration (default 32)
 *
 * Registers v8..v15 / v16..v23 / v24..v31 are used as the two source
 * groups and the destination group respectively so that the benchmark
 * works unmodified for LMUL in {1, 2, 4, 8} without violating RVV's
 * register-group alignment rules (all three base indices are multiples
 * of 8).
 */

#ifndef SEW
#define SEW 32
#endif
#ifndef LMUL
#define LMUL 1
#endif
#ifndef NITER
#define NITER 2000
#endif
#ifndef UNROLL
#define UNROLL 32
#endif

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define VSEW CAT(e, SEW)
#define VLMULTAG CAT(m, LMUL)

    .text
    .global _start
_start:
#ifdef AVL
    li      a2, AVL
    vsetvli t0, a2, VSEW, VLMULTAG, ta, ma
#else
    vsetvli t0, x0, VSEW, VLMULTAG, ta, ma
#endif

    /* Source operands: constant, functionally irrelevant for timing. */
    vmv.v.i v8, 1
    vmv.v.i v16, 2
#ifdef CHAIN
    vmv.v.i v24, 0
#endif

    li      s0, NITER
loop:
    .rept UNROLL
#ifdef CHAIN
    vadd.vv v24, v24, v16
#else
    vadd.vv v24, v8, v16
#endif
    .endr

    addi    s0, s0, -1
    bnez    s0, loop

    /* exit(0) -- riscv64 Linux syscall ABI, supported by gem5 SE mode. */
    li      a7, 93
    li      a0, 0
    ecall
