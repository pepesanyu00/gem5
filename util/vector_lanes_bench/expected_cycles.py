#!/usr/bin/env python3
"""
Analytical predictor for the RVV "vector lanes" chime-timing microbenchmark
(util/vector_lanes_bench/bench_vector_lanes.S), matching the model
implemented in:

  - src/arch/riscv/insts/vector.cc   (VectorMicroInst::numChimePasses)
  - src/cpu/o3/inst_queue.cc         (InstructionQueue::scheduleReadyInsts)

This is a standalone script (no gem5 Python environment required) meant to
be run *before* comparing against the actual `system.cpu.numCycles` in
m5out/stats.txt after simulating the corresponding binary with
run_lanes_bench.py. It does not touch or invoke gem5 in any way.

Model summary
-------------
RVV splits a `vadd.vv` into one microop per physical vector register in
the LMUL group (up to LMUL microops), each processing at most
`micro_vlmax = VLEN // SEW` elements. For a microop with `n` active
elements and a configured `lanes` count:

    passes = 1                          if lanes == 0 (unbounded, default)
    passes = ceil(n / lanes)            otherwise

gem5's issue logic (src/cpu/o3/inst_queue.cc) then uses `passes` to scale
both the completion latency and the functional-unit occupancy of that
microop:

    pipelined FU:     latency  = op_lat + passes - 1
                       occupancy = passes                    (cycles)
    non-pipelined FU: latency  = op_lat * passes
                       occupancy = op_lat * passes            (cycles)

(SimdAdd/SimdMult are pipelined in gem5's DefaultFUPool, so this script
assumes the pipelined case unless --non-pipelined is given.)

Two independent complications on top of the per-microop formula:

1. LMUL > 1 means a single `vadd.vv` decomposes into up to LMUL
   independent microops (each handling a different register-group slice).
   In the INDEPENDENT benchmark build, all of them (and all loop
   iterations) are mutually data-independent, so they can be spread
   across every available SimdAdd functional-unit instance
   (`--fu-count`, 4 by default in gem5's DefaultFUPool). In the CHAIN
   (dependency) benchmark build, each of the LMUL microops forms *its own*
   independent serial chain across loop iterations (slice i only depends
   on slice i of the previous iteration) -- so with LMUL > 1 the "chain"
   benchmark is actually LMUL independent chains, which can *also* be
   spread across multiple functional units. Only LMUL == 1 gives a
   perfectly clean, single global dependency chain limited purely by
   completion latency.

2. Real O3 throughput additionally depends on issue width, ROB/IQ size
   and the number of physical vector registers -- this script reports an
   idealized bound assuming none of those are the bottleneck (true for
   gem5's default O3 configuration at the small UNROLL/LMUL values this
   benchmark uses, but always sanity-check against the measured stats).

Usage
-----
    ./expected_cycles.py --vlen 256 --sew 32 --lmul 4 --lanes 4 \\
        --mode indep --niter 2000 --unroll 32

    ./expected_cycles.py --vlen 256 --sew 32 --lmul 1 --lanes 4 \\
        --op-lat 4 --mode chain --niter 2000 --unroll 32
"""

import argparse
import math


def microop_chunks(vlen, sew, lmul, avl):
    """Returns the list of per-microop active-element counts, replicating
    gem5's macro-op -> microop split (see
    src/arch/riscv/isa/templates/vector_arith.isa and
    src/arch/riscv/utility.hh:vtype_VLMAX(..., per_reg=True))."""
    micro_vlmax = vlen // sew
    vlmax_full = lmul * micro_vlmax
    vl = vlmax_full if avl is None else min(avl, vlmax_full)

    chunks = []
    remaining = vl
    for _ in range(lmul):
        if remaining <= 0:
            break
        chunks.append(min(remaining, micro_vlmax))
        remaining -= micro_vlmax
    return micro_vlmax, vlmax_full, vl, chunks


def passes_for(n, lanes):
    if lanes == 0:
        return 1
    return math.ceil(n / lanes)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vlen", type=int, default=256)
    p.add_argument("--sew", type=int, default=32, choices=[8, 16, 32, 64])
    p.add_argument("--lmul", type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--avl", type=int, default=None,
                    help="Requested vector length; default = VLMAX")
    p.add_argument("--lanes", type=int, default=0,
                    help="Configured RiscvISA.lanes (0 = unbounded)")
    p.add_argument("--op-lat", type=int, default=1,
                    help="Base FUPool opLat for the op (SimdAdd/SimdMult "
                    "default to 1 in gem5's DefaultFUPool)")
    p.add_argument("--non-pipelined", action="store_true",
                    help="Model a non-pipelined FU instead of the default "
                    "pipelined one")
    p.add_argument("--fu-count", type=int, default=4,
                    help="Number of parallel FU instances for this OpClass "
                    "(DefaultFUPool.SIMD_Unit.count == 4)")
    p.add_argument("--mode", choices=["indep", "chain"], required=True)
    p.add_argument("--niter", type=int, default=2000)
    p.add_argument("--unroll", type=int, default=32)
    args = p.parse_args()

    micro_vlmax, vlmax_full, vl, chunks = microop_chunks(
        args.vlen, args.sew, args.lmul, args.avl)
    passes = [passes_for(c, args.lanes) for c in chunks]

    print("== Vector configuration ==")
    print(f"  VLEN={args.vlen} SEW={args.sew} LMUL={args.lmul} "
          f"lanes={args.lanes or 'unbounded'}")
    print(f"  micro_vlmax (elements/register) = {micro_vlmax}")
    print(f"  VLMAX (this vtype)              = {vlmax_full}")
    print(f"  vl actually used                = {vl}"
          + ("" if args.avl is None else f" (requested AVL={args.avl})"))
    print(f"  microops for this macro vadd.vv = {len(chunks)} "
          f"(active elements per microop: {chunks})")
    print(f"  chime passes per microop        = {passes}")

    if args.mode == "chain":
        if args.non_pipelined:
            per_step = [args.op_lat * n for n in passes]
        else:
            per_step = [args.op_lat + n - 1 for n in passes]

        print("\n== Latency (dependency-chain) prediction ==")
        print(f"  cycles per macro vadd.vv, per independent chain slice: "
              f"{per_step}")
        if args.lmul == 1:
            cycles_per_inst = per_step[0]
            total = args.niter * args.unroll * cycles_per_inst
            print(f"  LMUL=1 => single global chain, no ambiguity.")
            print(f"  predicted cycles/instruction = {cycles_per_inst}")
            print(f"  predicted TOTAL cycles       ~= {total} "
                  f"(+ small, mostly-fixed loop/setup overhead)")
        else:
            print(f"  LMUL={args.lmul} > 1: this is actually {args.lmul} "
                  "independent chains (one per register-group slice), "
                  "which CAN be spread across up to --fu-count "
                  f"({args.fu_count}) functional units.")
            lower = max(per_step)
            upper = sum(per_step)
            print(f"  predicted cycles/instruction in "
                  f"[{lower} (all {args.lmul} slices parallel across "
                  f"FUs), {upper} (fully serialized on 1 FU)]")
            print(f"  predicted TOTAL cycles in "
                  f"[{args.niter*args.unroll*lower}, "
                  f"{args.niter*args.unroll*upper}]")

    else:  # indep
        occ = passes  # occupancy cycles per microop, pipelined case
        if args.non_pipelined:
            occ = [args.op_lat * n for n in passes]

        total_occupancy = sum(occ) * args.niter * args.unroll
        single_fu_cpi = sum(occ)
        multi_fu_cpi = sum(occ) / args.fu_count

        print("\n== Throughput / FU-occupancy prediction ==")
        print(f"  occupancy cycles per microop = {occ}")
        print(f"  cycles/instruction if serialized on a single FU "
              f"= {single_fu_cpi}")
        print(f"  cycles/instruction with {args.fu_count} parallel FUs "
              f"(idealized, no other bottleneck) = {multi_fu_cpi:.3f}")
        print(f"  predicted TOTAL cycles in "
              f"[{total_occupancy/args.fu_count:.0f}, {total_occupancy}] "
              "depending on how many FU instances gem5 actually manages "
              "to keep busy (bounded above by issue width, ROB/IQ size "
              "and the number of physical vector registers)")

    print("\n== How to compare against gem5 ==")
    print("  1. Run the matching binary with run_lanes_bench.py.")
    print("  2. In m5out/stats.txt, take `system.cpu.numCycles`.")
    n_vec_insts = args.niter * args.unroll
    print(f"  3. cycles_per_inst_measured ~= numCycles / {n_vec_insts} "
          "(this slightly OVER-estimates cycles/inst because it also "
          "includes a small, mostly-fixed setup + loop-control overhead; "
          "increase --niter/--unroll in the benchmark to shrink that "
          "error further).")


if __name__ == "__main__":
    main()
