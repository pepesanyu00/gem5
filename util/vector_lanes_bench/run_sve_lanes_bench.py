"""
gem5 SE-mode config script to run the ARM SVE "vector lanes" chime-timing
microbenchmarks (see util/vector_lanes_bench/bench_sve_lanes.S) on an ARM
O3CPU, with SVE vector length and the new `sve_lanes` parameter all
configurable from the command line.

This is the SVE counterpart of run_lanes_bench.py (RVV). Build the
benchmark binaries first with the accompanying Makefile_sve, then invoke
this script once per binary/configuration and compare
`system.cpu.numCycles` in `m5out/stats.txt` against the analytical
prediction from `expected_cycles_sve.py`.

Usage
-----

    scons build/ARM/gem5.opt -j$(nproc)

    # Build SVE benchmark binaries
    make -f Makefile_sve -C util/vector_lanes_bench

    # Run: sve_vl=2 means 2 quadwords = 256 bits, with 4 lanes
    ./build/ARM/gem5.opt util/vector_lanes_bench/run_sve_lanes_bench.py \\
        --binary util/vector_lanes_bench/build_sve/bench_sve_elem32_indep \\
        --sve-vl 2 --sve-lanes 4

    # Optionally override SimdAdd latency for the opLat>1 chime formula test:
    ./build/ARM/gem5.opt util/vector_lanes_bench/run_sve_lanes_bench.py \\
        --binary util/vector_lanes_bench/build_sve/bench_sve_elem32_chain \\
        --sve-vl 2 --sve-lanes 4 --vadd-lat 4

Then compare (numCycles / (NITER * UNROLL)) against expected_cycles_sve.py.
"""

import argparse

from m5.objects import (
    ArmDefaultSERelease,
    ArmO3CPU,
    DefaultFUPool,
)

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires


class SveLaneBenchFUPool(DefaultFUPool):
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


class SveLaneBenchCore(BaseCPUCore):
    def __init__(self, sve_vl, sve_lanes, vadd_lat, vmul_lat, cpu_id=0):
        core = ArmO3CPU(cpu_id=cpu_id)
        core.fuPool = SveLaneBenchFUPool(vadd_lat=vadd_lat, vmul_lat=vmul_lat)
        super().__init__(core=core, isa=ISA.ARM)

        # In SE mode, SVE VL and lanes are configured on ArmISA, not ArmSystem.
        # ArmISA exposes `sve_vl_se` and `sve_lanes_se` for this purpose.
        self.core.isa[0].sve_vl_se = sve_vl
        self.core.isa[0].sve_lanes_se = sve_lanes

        # Make sure SVE is enabled in the SE release (it is by default in
        # ArmDefaultSERelease, but be explicit):
        self.core.isa[0].release_se = ArmDefaultSERelease()


requires(isa_required=ISA.ARM)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--binary",
    required=True,
    type=str,
    help="Path to a benchmark binary "
    "built by util/vector_lanes_bench/Makefile_sve",
)
parser.add_argument(
    "--sve-vl",
    type=int,
    default=2,
    help="SVE vector length in quadwords (128-bit units). "
    "E.g., 2 = 256 bits, 4 = 512 bits. Range: 1-16.",
)
parser.add_argument(
    "--sve-lanes",
    type=int,
    default=0,
    help="Number of SVE execution lanes (0 = unbounded/unrealistic "
    "default, matching gem5's historical behavior). Must be 0 or "
    "a power of 2.",
)
parser.add_argument(
    "--vadd-lat",
    type=int,
    default=None,
    help="Override the base (lanes=0-equivalent) latency of SimdAdd, "
    "in cycles. Default: gem5 stock value (1).",
)
parser.add_argument(
    "--vmul-lat",
    type=int,
    default=None,
    help="Override the base latency of SimdMult, in cycles. Default: "
    "gem5 stock value (1).",
)
args = parser.parse_args()

## IMPORTANT: a real (even minimal) cache hierarchy is required here.
## See run_lanes_bench.py comments for the full rationale (the NoCache()
## mistake that swamped the RVV measurement with DRAM fetch latency).
cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32KiB", l1i_size="32KiB", l2_size="512KiB"
)
memory = SingleChannelDDR3_1600()
processor = BaseCPUProcessor(
    cores=[
        SveLaneBenchCore(
            sve_vl=args.sve_vl,
            sve_lanes=args.sve_lanes,
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
    f"Beginning simulation! binary={args.binary} sve_vl={args.sve_vl} "
    f"sve_lanes={args.sve_lanes} vadd_lat={args.vadd_lat} "
    f"vmul_lat={args.vmul_lat}"
)
simulator.run()

print("Simulation finished!")
