#!/usr/bin/env python3
"""
Analytical predictor for the ARM SVE "vector lanes" chime-timing microbenchmark
(util/vector_lanes_bench/bench_sve_lanes.S), matching the model implemented in:

  - src/arch/arm/insts/static_inst.cc  (ArmStaticInst::numSveChimePasses)
  - src/cpu/o3/inst_queue.cc           (InstructionQueue::scheduleReadyInsts)

This is the SVE counterpart of expected_cycles.py (RVV). Standalone script
(no gem5 Python environment required), run *before* comparing against the
actual `system.cpu.numCycles` in m5out/stats.txt.

Model summary
-------------
SVE differs from RVV in a key simplification: there is no LMUL. Each SVE
instruction operates on a single full vector register of VL bits, where
VL = sve_vl * 128 bits (sve_vl is in quadwords, range 1-16).

The active element count for a given element width is:

    eCount = VL_bits / elem_width

For a configured `lanes` count:

    passes = 1                          if lanes == 0 (unbounded, default)
    passes = ceil(eCount / lanes)       otherwise

gem5's issue logic then scales both latency and FU occupancy:

    pipelined FU:     latency  = op_lat + passes - 1
                       occupancy = passes                    (cycles)
    non-pipelined FU: latency  = op_lat * passes
                       occupancy = op_lat * passes            (cycles)

(SimdAdd is pipelined in gem5's DefaultFUPool, which is the default used
by the accompanying run_sve_lanes_bench.py.)

Since SVE does NOT have LMUL, each macro-instruction is also 1 micro-op,
so there is no LMUL-induced multiplication of microops — much simpler
than the RVV case.

Usage
-----
    ./expected_cycles_sve.py --sve-vl 2 --elem-width 32 --lanes 4 \\
        --mode indep --niter 2000 --unroll 32

    ./expected_cycles_sve.py --sve-vl 2 --elem-width 32 --lanes 4 \\
        --op-lat 4 --mode chain --niter 2000 --unroll 32
"""

import argparse
import math


def passes_for(eCount, lanes):
    if lanes == 0:
        return 1
    return math.ceil(eCount / lanes)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sve-vl",
        type=int,
        default=2,
        help="SVE vector length in quadwords (128-bit units). "
        "E.g., 2 = 256 bits. Range: 1-16.",
    )
    p.add_argument(
        "--elem-width",
        type=int,
        default=32,
        choices=[8, 16, 32, 64],
        help="Element width in bits",
    )
    p.add_argument(
        "--lanes",
        type=int,
        default=0,
        help="Configured ArmISA.sve_lanes (0 = unbounded)",
    )
    p.add_argument(
        "--op-lat",
        type=int,
        default=1,
        help="Base FUPool opLat for the op (SimdAdd default = 1)",
    )
    p.add_argument(
        "--non-pipelined",
        action="store_true",
        help="Model a non-pipelined FU instead of the default pipelined one",
    )
    p.add_argument(
        "--fu-count",
        type=int,
        default=4,
        help="Number of parallel FU instances for this OpClass "
        "(DefaultFUPool.SIMD_Unit.count == 4)",
    )
    p.add_argument(
        "--mode", choices=["indep", "chain"], required=True
    )
    p.add_argument("--niter", type=int, default=2000)
    p.add_argument("--unroll", type=int, default=32)
    args = p.parse_args()

    vl_bits = args.sve_vl * 128
    eCount = vl_bits // args.elem_width
    passes = passes_for(eCount, args.lanes)

    print("== SVE Vector configuration ==")
    print(
        f"  sve_vl={args.sve_vl} ({vl_bits} bits)  "
        f"elem_width={args.elem_width}  "
        f"lanes={args.lanes or 'unbounded'}"
    )
    print(f"  eCount (active elements) = {vl_bits} / {args.elem_width} = {eCount}")
    print(f"  chime passes             = ceil({eCount} / {args.lanes or '∞'}) = {passes}")

    if args.mode == "chain":
        if args.non_pipelined:
            latency = args.op_lat * passes
        else:
            latency = args.op_lat + passes - 1

        print(f"\n== Latency (dependency-chain) prediction ==")
        print(f"  cycles per instruction = opLat + passes - 1 = "
              f"{args.op_lat} + {passes} - 1 = {latency}")

        total = args.niter * args.unroll * latency
        print(f"  predicted cycles/instruction = {latency}")
        print(
            f"  predicted TOTAL cycles       ~= {total} "
            "(+ small, mostly-fixed loop/setup overhead)"
        )

    else:  # indep
        if args.non_pipelined:
            occ = args.op_lat * passes
        else:
            occ = passes  # occupancy cycles per instruction

        n_vec_insts = args.niter * args.unroll
        total_occupancy = occ * n_vec_insts
        single_fu_cpi = occ
        multi_fu_cpi = occ / args.fu_count

        print(f"\n== Throughput / FU-occupancy prediction ==")
        print(f"  occupancy cycles per instruction = {occ}")
        print(
            f"  cycles/instruction if serialized on a single FU "
            f"= {single_fu_cpi}"
        )
        print(
            f"  cycles/instruction with {args.fu_count} parallel FUs "
            f"(idealized) = {multi_fu_cpi:.3f}"
        )
        print(
            f"  predicted TOTAL cycles in "
            f"[{total_occupancy // args.fu_count}, {total_occupancy}] "
            "depending on how many FU instances gem5 actually keeps busy"
        )

    print(f"\n== How to compare against gem5 ==")
    print("  1. Run the matching binary with run_sve_lanes_bench.py.")
    print("  2. In m5out/stats.txt, take `system.cpu.numCycles`.")
    n_vec_insts = args.niter * args.unroll
    print(
        f"  3. cycles_per_inst_measured ~= numCycles / {n_vec_insts} "
        "(this slightly OVER-estimates cycles/inst because it also "
        "includes a small, mostly-fixed setup + loop-control overhead; "
        "increase --niter/--unroll in the benchmark to shrink that "
        "error further)."
    )


if __name__ == "__main__":
    main()
