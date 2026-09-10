"""
This script demonstrates how to run RISC-V vector-enabled binaries in SE mode
with gem5. It accepts the number of CORES, VLEN, and ELEN as optional
parameters, as well as the resource name to run. If no resource name is
provided, a list of available resources will be displayed. If one is given the
simulation will then execute the specified resource binary with the selected
parameters until completion.


Usage
-----

# Compile gem5 for RISC-V
scons build/RISCV/gem5.opt

# Run the simulation
./build/RISCV/gem5.opt configs/example/gem5_library/riscv-rvv-example.py \
    [-c CORES] [-v VLEN] [-e ELEN] <resource>

"""

import argparse
import m5
from m5.objects import RiscvO3CPU, RiscvMinorCPU, RiscvTimingSimpleCPU

#from gem5.components.boards.mem_mode import MemMode
from gem5.components.boards.riscv_board import RiscvBoard
#from gem5.components.cachehierarchies.ruby.mesi_three_level_cache_hierarchy import MESIThreeLevelCacheHierarchy

from gem5.components.memory.memory import ChanneledMemory
from gem5.components.memory.dram_interfaces.ddr4 import DDR4_2400_16x4

from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
#from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent

from custom_cache.three_level_cache import ThreeLevelCacheHierarchy
requires(isa_required=ISA.RISCV)



class RVVCore(BaseCPUCore):
    def __init__(self, elen, vlen, lanes,cpu_id):
        super().__init__(core=RiscvO3CPU(cpu_id=cpu_id), isa=ISA.RISCV)
        self.core.isa[0].elen = elen
        self.core.isa[0].vlen = vlen


parser = argparse.ArgumentParser()
parser.add_argument("binary", nargs="+", type=str)
parser.add_argument("-c", "--cores", required=False, type=int, default=64)
parser.add_argument("-v", "--vlen", required=False, type=int, default=512)
parser.add_argument("-e", "--elen", required=False, type=int, default=64)
parser.add_argument("--lanes", required=False, type=int, default=0, help="Number of vector lanes")
parser.add_argument("--l1i_size", help="Size of the l1i cache.", default="32KiB")
parser.add_argument("--l1d_size", help="Size of the l1d cache.", default="64KiB")
parser.add_argument("--l2_size", help="Size of the l2 cache.", default="256KiB")
parser.add_argument("--l3_size", help="Size of the l3 cache.", default="16MiB")

args = parser.parse_args()

cache_hierarchy = ThreeLevelCacheHierarchy(
    l1d_size=args.l1d_size,
    l1i_size=args.l1i_size,
    l2_size=args.l2_size,
    l3_size=args.l3_size,
)

memory = ChanneledMemory(DDR4_2400_16x4, 1, 64, size='32GiB')

# Processor configuration with CPU change
processor = BaseCPUProcessor(
    cores = [RVVCore(args.elen, args.vlen, args.lanes, i) for i in range(args.cores)]
)


board = RiscvBoard(
    clk_freq="2GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)
# To change timing mode
#board.set_mem_mode(MemMode.TIMING)

path, *arguments = args.binary
binary = BinaryResource(local_path=path)
board.set_se_binary_workload(binary, arguments= arguments)

# Function to switch CPU in regions of interest
def roi_begin_handler():
    print("Taking stats from SCAMP")
    m5.stats.reset()  # Reset ROI statistics
    print("stats have been reset in tick {}!".format(
        simulator.get_current_tick(),
    )
)
    #simulator.save_checkpoint("checkpoints")

# Function to switch back to Timing at the end of the ROI
def roi_end_handler():
    m5.stats.dump()  # Save ROI statistics
    print("stats have been dumped in tick {}!".format(
        simulator.get_current_tick(),
    )
)


simulator = Simulator(
    board=board, 
    full_system=False,
    on_exit_event={
        ExitEvent.WORKBEGIN : roi_begin_handler,
        ExitEvent.WORKEND : roi_end_handler,
    }                
)

print("Beginning simulation!")
simulator.run()

print(
    "Exiting @ tick {} because {}.".format(
        simulator.get_current_tick(),
        simulator.get_last_exit_event_cause(),
    )
)

#m5.drain()
#print("drained")
#simulator.save_checkpoint("checkpoints")
