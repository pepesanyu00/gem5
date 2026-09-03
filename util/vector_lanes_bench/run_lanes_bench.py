"""
gem5 SE-mode config script to run the RVV "vector lanes" chime-timing
microbenchmarks (see util/vector_lanes_bench/bench_vector_lanes.S) on
RiscvO3CPU, with VLEN, ELEN and the new `lanes` parameter (introduced to
model a finite number of vector execution lanes) all configurable from
the command line.

This script deliberately does *not* run anything by itself beyond what
you ask it to: build the benchmark binaries first with the accompanying
Makefile, then invoke this script once per binary/configuration you want
to measure, and compare `system.cpu.numCycles` (or the derived
cycles-per-instruction figures) in the resulting `m5out/stats.txt`
against the analytical prediction from `expected_cycles.py`.

Usage
-----

    scons build/RISCV/gem5.opt

    # (build the benchmark binaries -- see util/vector_lanes_bench/Makefile)
    make -C util/vector_lanes_bench

    ./build/RISCV/gem5.opt util/vector_lanes_bench/run_lanes_bench.py \\
        --binary util/vector_lanes_bench/build/bench_sew32_lmul1_indep \\
        --vlen 256 --elen 64 --lanes 4

    # Optionally override the default (opLat=1) latency gem5 assigns to
    # SimdAdd/SimdMult, to also exercise the opLat>1 branch of the chime
    # formula:
    ./build/RISCV/gem5.opt util/vector_lanes_bench/run_lanes_bench.py \\
        --binary util/vector_lanes_bench/build/bench_sew32_lmul1_chain \\
        --vlen 256 --elen 64 --lanes 4 --vadd-lat 4

Then inspect m5out/stats.txt for `system.cpu.numCycles` and
`system.cpu.commitStats0.numInsts` (exact stat names can vary slightly
across gem5 versions/CPU configurations) and compare
(numCycles / (NITER * UNROLL)) against expected_cycles.py's prediction.
"""

import argparse

from m5.objects import DefaultFUPool, RiscvO3CPU

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires


class LaneBenchFUPool(DefaultFUPool):
    """DefaultFUPool with optional latency overrides for SimdAdd/SimdMult.

    Left at their gem5-stock defaults (opLat=1, pipelined=True) unless
    explicitly overridden, so that not passing --vadd-lat/--vmul-lat
    reproduces the exact same timing as an unmodified DefaultFUPool.
    """

    def __init__(self, vadd_lat=None, vmul_lat=None):
        super().__init__()
        for fu in self.FUList:
            for op in fu.opList:
                if vadd_lat is not None and op.opClass == "SimdAdd":
                    op.opLat = vadd_lat
                if vmul_lat is not None and op.opClass == "SimdMult":
                    op.opLat = vmul_lat


class LaneBenchCore(BaseCPUCore):
    def __init__(self, vlen, elen, lanes, vadd_lat, vmul_lat, cpu_id=0):
        core = RiscvO3CPU(cpu_id=cpu_id)
        core.fuPool = LaneBenchFUPool(vadd_lat=vadd_lat, vmul_lat=vmul_lat)
        super().__init__(core=core, isa=ISA.RISCV)
        self.core.isa[0].vlen = vlen
        self.core.isa[0].elen = elen
        self.core.isa[0].lanes = lanes


requires(isa_required=ISA.RISCV)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--binary", required=True, type=str, help="Path to a benchmark binary "
    "built by util/vector_lanes_bench/Makefile"
)
parser.add_argument("--vlen", type=int, default=256, help="VLEN in bits")
parser.add_argument("--elen", type=int, default=64, help="ELEN in bits")
parser.add_argument(
    "--lanes", type=int, default=0,
    help="Number of vector execution lanes (0 = unbounded/unrealistic "
    "default, matching gem5's historical behavior)"
)
parser.add_argument(
    "--vadd-lat", type=int, default=None,
    help="Override the base (lanes=0-equivalent) latency of SimdAdd, "
    "in cycles. Default: gem5 stock value (1)."
)
parser.add_argument(
    "--vmul-lat", type=int, default=None,
    help="Override the base latency of SimdMult, in cycles. Default: "
    "gem5 stock value (1)."
)
args = parser.parse_args()

cache_hierarchy = NoCache()
memory = SingleChannelDDR3_1600()
processor = BaseCPUProcessor(
    cores=[
        LaneBenchCore(
            vlen=args.vlen,
            elen=args.elen,
            lanes=args.lanes,
            vadd_lat=args.vadd_lat,
            vmul_lat=args.vmul_lat,
        )
    ]
)

board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

board.set_se_binary_workload(BinaryResource(local_path=args.binary))

simulator = Simulator(board=board, full_system=False)
print(
    f"Beginning simulation! binary={args.binary} vlen={args.vlen} "
    f"elen={args.elen} lanes={args.lanes} vadd_lat={args.vadd_lat} "
    f"vmul_lat={args.vmul_lat}"
)
simulator.run()
