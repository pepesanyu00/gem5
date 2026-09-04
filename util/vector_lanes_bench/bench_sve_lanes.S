/*
 * Microbenchmark to validate gem5's ARM SVE "vector lanes" chime timing model
 * (ArmISA::sve_lanes, ArmStaticInst::numSveChimePasses,
 * src/cpu/o3/inst_queue.cc:scheduleReadyInsts).
 *
 * Analogous to bench_vector_lanes.S (RISC-V RVV), but using AArch64 SVE
 * instructions. The program executes a tight loop of SVE integer add
 * instructions, then exits via the exit(93) syscall.
 *
 * SVE differs from RVV in that:
 *   - There is no LMUL concept: each instruction operates on a full
 *     SVE vector register (VL bits wide).
 *   - The active element count is purely a function of VL and element
 *     width: eCount = VL_bits / ELEM_WIDTH.
 *   - SVE instructions are NOT split into micro-ops (unlike RVV which
 *     creates one micro-op per LMUL register); each SVE instruction is
 *     a single StaticInst that the chime model scales as a whole.
 *
 * Build-time parameters (see Makefile_sve, passed via -D to the assembler):
 *   ELEM_WIDTH - element width in bits: 8, 16, 32 or 64 (default 32)
 *   CHAIN      - if defined, instructions form a RAW dependency chain
 *                (measures completion LATENCY). If undefined, instructions
 *                are independent (measures FU OCCUPANCY / throughput).
 *   NITER      - number of loop iterations (default 2000)
 *   UNROLL     - number of add instructions per loop iteration (default 32)
 *
 * Registers used:
 *   z0 - destination (chain mode: also a source for RAW dependency)
 *   z1, z2 - source operands (initialized to constants)
 *   x0 - loop counter
 *   p0 - all-true predicate (ptrue, used implicitly by unpredicated add)
 */

#ifndef ELEM_WIDTH
#define ELEM_WIDTH 32
#endif
#ifndef NITER
#define NITER 2000
#endif
#ifndef UNROLL
#define UNROLL 32
#endif

/* Map ELEM_WIDTH to SVE element size suffix for the assembler */
#if ELEM_WIDTH == 8
#define SUFFIX b
#elif ELEM_WIDTH == 16
#define SUFFIX h
#elif ELEM_WIDTH == 32
#define SUFFIX s
#elif ELEM_WIDTH == 64
#define SUFFIX d
#else
#error "Unsupported ELEM_WIDTH: must be 8, 16, 32 or 64"
#endif

    .text
    .global _start

_start:
    /* Set p0 to all-true for the configured element width.
     * ptrue sets all predicate lanes to 1 for the given element size. */
    ptrue   p0.SUFFIX

    /* Initialize source vectors with constant values.
     * Functionally irrelevant for timing measurement. */
    dup     z1.SUFFIX, #1
    dup     z2.SUFFIX, #2
#ifdef CHAIN
    dup     z0.SUFFIX, #0
#endif

    /* Loop counter */
    mov     x0, #NITER

.Lloop:
    .rept UNROLL
#ifdef CHAIN
    /* Dependent chain: z0 = z0 + z2 (RAW on z0 each iteration) */
    add     z0.SUFFIX, z0.SUFFIX, z2.SUFFIX
#else
    /* Independent: z0 = z1 + z2 (no true dependency between iterations,
     * z0 is overwritten each time but register renaming eliminates the
     * WAW hazard — only FU occupancy limits throughput) */
    add     z0.SUFFIX, z1.SUFFIX, z2.SUFFIX
#endif
    .endr

    subs    x0, x0, #1
    b.ne    .Lloop

    /* exit(0) — AArch64 Linux syscall ABI, supported by gem5 SE mode.
     * syscall number 93 = exit */
    mov     x8, #93
    mov     x0, #0
    svc     #0
