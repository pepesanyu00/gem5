#!/usr/bin/env python3
"""
Example of configuring an ARM system with SVE extension using
gem5's m5 libraries. This script creates a three-level cache
hierarchy and configures the vector length (vl) through the
'sve_vl_se' parameter of the ArmISA object.
"""

import m5
from m5.objects import Root, System, SystemXBar, AddrRange, ArmO3CPU, ArmISA, Cache, SnoopFilter, L2XBar

from custom_cache.caches.l1dcache import L1DCache
from custom_cache.caches.l1icache import L1ICache
from custom_cache.caches.l2cache import L2Cache
from custom_cache.caches.l3cache import L3Cache

from m5.objects import MemCtrl, DDR4_2400_16x4  # Update imports
from m5.objects import SEWorkload
from m5.objects import RubySystem
# Try to import ARM-specific processes
try:
    # New path in recent versions
    from m5.objects.arm.ArmProcess import ArmProcess, ArmSEProcess
except ImportError:
    try:
        # Alternative path
        from m5.objects.Process import Process, SEProcess
        # Use generic process if ARM-specific is not available
        ArmProcess = SEProcess
    except ImportError:
        # Use base process as a last resort
        from m5.objects import Process
        ArmProcess = Process
import argparse
import sys



# Argument parsing
parser = argparse.ArgumentParser(description="ARM simulation with SVE using m5")
parser.add_argument("binary", nargs="+", type=str, help="Path to the binary to simulate and its arguments")
parser.add_argument("-c", "--cores", type=int, default=64, help="Number of cores (default: 64)")
parser.add_argument("-v", "--vlen", type=int, default=512, help="Vector length (VL) in bits (default: 512)")
parser.add_argument("-e", "--elen", type=int, default=64, help="Element length (ELEN) (default: 64)")
parser.add_argument("--lanes", type=int, default=0, help="Number of vector lanes (0 = unbounded)")
parser.add_argument("--l1i_size", default="32KiB", help="L1 instruction cache size (default: 32KiB)")
parser.add_argument("--l1d_size", default="64KiB", help="L1 data cache size (default: 64KiB)")
parser.add_argument("--l2_size", default="256KiB", help="L2 cache size (default: 256KiB)")
parser.add_argument("--l3_size", default="16MiB", help="L3 cache size (default: 16MiB)")
args = parser.parse_args()

# Create the system
system = System()
system.clk_domain = m5.objects.SrcClockDomain(clock="2GHz", voltage_domain=m5.objects.VoltageDomain()) 
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange("32GiB")]
system.exit_on_work_items = True  # Habilita salida en m5_work_begin y m5_work_end

# Create CPUs with SVE extension
# Replace direct assignment with Vector
system.cpu = [ArmO3CPU(cpu_id=i) for i in range(args.cores)]
# Create the main connection (memory bus)
system.membus = SystemXBar(width=64)
system.membus.snoop_filter.max_capacity = "16MiB"

# Configure each CPU
for i in range(args.cores):
    # Vector register configuration
    system.cpu[i].numPhysVecRegs = 256
    system.cpu[i].numPhysVecPredRegs = 64
    system.cpu[i].createInterruptController()
    # Configure ISA with SVE parameter: 'sve_vl_se' (vl expressed in 128-bit words)
    system.cpu[i].isa = ArmISA(sve_vl_se=args.vlen/128, sve_lanes_se=args.lanes)

    #system.cpu[i].icache_port = system.membus.cpu_side_ports
    #system.cpu[i].dcache_port = system.membus.cpu_side_ports


# --- Cache hierarchy configuration ---

system.cache_line_size = 64 


# A shared L3 bus and L3 cache are created
system.l3bus = L2XBar()
system.l3bus.snoop_filter.max_capacity = "16MiB"
system.l3cache = L3Cache(size = args.l3_size)


system.l3cache.cpu_side = system.l3bus.mem_side_ports

# Private L1 and L2
for i in range(args.cores):
    cpu = system.cpu[i]
    # Configure L1 caches
    cpu.icache = L1ICache(size = args.l1i_size)
    cpu.dcache = L1DCache(size = args.l1d_size)
    
    # Connect L1 caches to CPU
    cpu.icache.cpu_side = cpu.icache_port
    cpu.dcache.cpu_side = cpu.dcache_port

    cpu.l2bus = L2XBar(snoop_response_latency=1)
    cpu.l2bus.snoop_filter.max_capacity = "16MiB"

    # Connect L1 caches to private bus
    cpu.icache.mem_side = cpu.l2bus.cpu_side_ports
    cpu.dcache.mem_side = cpu.l2bus.cpu_side_ports
    
    cpu.l2cache = L2Cache(size = args.l2_size)

    cpu.l2cache.cpu_side = cpu.l2bus.mem_side_ports

    cpu.l2cache.mem_side = system.l3bus.cpu_side_ports
    
# Connect L3 cache to main memory bus
system.l3cache.mem_side = system.membus.cpu_side_ports

# Create DDR4 memory controller and connect it to the main bus
system.mem_ctrl = MemCtrl(dram=DDR4_2400_16x4(range=system.mem_ranges[0]))
system.mem_ctrl.port = system.membus.mem_side_ports

# --- End of cache hierarchy ---



# Create a process for ARM
binary = args.binary[0]
process = ArmProcess()
process.executable = binary
if len(args.binary) > 1:
    process.cmd = [binary] + args.binary[1:]
else:
    process.cmd = [binary]

# Configure workload at system level first
system.workload = SEWorkload.init_compatible(binary)

# Assign the process to each CPU and create threads
for cpu in system.cpu:
    cpu.workload = [process]  # Needs to be a list in new versions
    cpu.createThreads()

# Create system root and perform instantiation
root = Root(full_system=False,system=system)
m5.instantiate()

# Function to switch CPU in regions of interest
def roi_begin_handler():
    print("Taking stats from SCAMP")
    m5.stats.reset()  # Reset ROI statistics
    print("stats have been reset in tick {}!".format(m5.curTick()))
    #simulator.save_checkpoint("checkpoints")

# Function to switch back to Timing at the end of the ROI
def roi_end_handler():
    m5.stats.dump()  # Save ROI statistics
    print("stats have been dumped in tick {}!".format(m5.curTick()))

print("Starting simulation...")
exit_event = m5.simulate()

print("Simulation finished at tick {} due to: {}".format(m5.curTick(), exit_event.getCause()))

# Continue simulating until a termination event is reached
checks = 0
while exit_event.getCause() != "exiting with last active thread context":
    if exit_event.getCause() == "workbegin":
        roi_begin_handler()
    elif exit_event.getCause() == "workend":
        roi_end_handler()
    elif exit_event.getCause() == "user interrupt received":
        break
    exit_event = m5.simulate()

print("Simulation finished at tick {} due to: {}".format(m5.curTick(), exit_event.getCause()))

#m5.checkpoint("checkpoints")
#print("Checkpoint saved in 'checkpoints'")