# Makefile for the RVV "vector lanes" chime-timing microbenchmark
# (bench_vector_lanes.S). See that file's header comment for background.
#
# Usage
# -----
#
# Build the default matrix of benchmark variants into build/:
#
#   make
#
# Build a single, custom-parametrized variant:
#
#   make custom SEW=64 LMUL=4 AVL=17 CHAIN=1 NITER=5000 UNROLL=16 \
#       OUT=build/custom_sew64_lmul4_avl17_chain
#
# Cross-compiler
# --------------
# Override CROSS_COMPILE if your toolchain uses a different prefix, e.g.:
#
#   make CROSS_COMPILE=riscv64-linux-gnu-
#
# The toolchain must support the RVV 1.0 vector extension (binutils >=
# 2.38 / gcc >= 13, or an equivalent riscv-gnu-toolchain build with 'v'
# support).

CROSS_COMPILE ?= riscv64-unknown-linux-gnu-
CC             = $(CROSS_COMPILE)g++
M5FLAGS_RISCV = -I/home/jsanchez/gem5/include -L/home/jsanchez/gem5/util/m5/build/riscv/out -lm5
ARCH           = rv64gcv
ABI            = lp64d

CFLAGS  = -march=$(ARCH) -mabi=$(ABI) -nostdlib -static -Wl,-e,_start $(M5FLAGS_RISCV)
BUILD   = build

# SEW/LMUL combinations used by the default `all` target. Edit freely.
SEWS      = 8 32 64
LMULS     = 1 4
NITER     ?= 2000
UNROLL    ?= 32

SRC = bench_vector_lanes.S

.PHONY: all clean custom
all: $(BUILD)
	@for sew in $(SEWS); do \
	  for lmul in $(LMULS); do \
	    for mode in indep chain; do \
	      name=$(BUILD)/bench_sew$${sew}_lmul$${lmul}_$${mode}; \
	      if [ "$$mode" = "chain" ]; then chain_flag="-DCHAIN"; else chain_flag=""; fi; \
	      echo "Building $$name (SEW=$$sew LMUL=$$lmul mode=$$mode)"; \
	      $(CC) $(CFLAGS) -DSEW=$$sew -DLMUL=$$lmul -DNITER=$(NITER) \
	        -DUNROLL=$(UNROLL) $$chain_flag -o $$name $(SRC); \
	    done; \
	  done; \
	done
	@echo "Binaries written to $(BUILD)/"

# Single custom build, e.g.:
#   make custom SEW=32 LMUL=1 AVL=5 CHAIN=1 OUT=build/my_bench
SEW    ?= 32
LMUL   ?= 1
OUT    ?= $(BUILD)/bench_custom
DEFS    = -DSEW=$(SEW) -DLMUL=$(LMUL) -DNITER=$(NITER) -DUNROLL=$(UNROLL)
ifdef AVL
DEFS   += -DAVL=$(AVL)
endif
ifdef CHAIN
DEFS   += -DCHAIN
endif

custom: $(BUILD)
	$(CC) $(CFLAGS) $(DEFS) -o $(OUT) $(SRC)
	@echo "Built $(OUT) (SEW=$(SEW) LMUL=$(LMUL) AVL=$(AVL) CHAIN=$(CHAIN))"

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
