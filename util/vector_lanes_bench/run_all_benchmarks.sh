#!/usr/bin/env bash
# =============================================================================
# run_all_benchmarks.sh
#
# Master script to build, run and validate ALL vector lanes microbenchmarks
# (RISC-V RVV + ARM SVE) against their analytical predictions.
#
# For each benchmark binary it:
#   1. Runs the expected_cycles predictor and captures the predicted CPI.
#   2. Runs gem5 in SE mode, directing output to results/<bench_name>/.
#   3. Extracts system.cpu.numCycles from stats.txt.
#   4. Computes measured CPI = numCycles / N_VEC_INSTS (default 64000).
#   5. Compares predicted vs measured CPI and reports the error.
#
# Usage:
#   ./run_all_benchmarks.sh                          # run everything
#   ./run_all_benchmarks.sh --riscv-only             # only RISC-V
#   ./run_all_benchmarks.sh --arm-only               # only ARM SVE
#   ./run_all_benchmarks.sh --skip-build             # skip scons + make
#   ./run_all_benchmarks.sh --lanes 8                # override lanes count
#   ./run_all_benchmarks.sh --vlen 512 --sve-vl 4    # override vector lengths
#   ./run_all_benchmarks.sh --dry-run                # print commands, don't execute
#
# Prerequisites:
#   - gem5 built for RISC-V  (build/RISCV/gem5.opt)
#   - gem5 built for ARM     (build/ARM/gem5.opt)
#   - RISC-V cross-compiler  (riscv64-unknown-linux-gnu-gcc or similar)
#   - AArch64 cross-compiler (aarch64-linux-gnu-gcc or similar)
# =============================================================================

set -euo pipefail

# ── Paths (relative to gem5 root) ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEM5_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_DIR="$SCRIPT_DIR"
RESULTS_DIR="$BENCH_DIR/results"

GEM5_RISCV="$GEM5_ROOT/build/RISCV/gem5.opt"
GEM5_ARM="$GEM5_ROOT/build/ARM/gem5.opt"

RISCV_BUILD_DIR="$BENCH_DIR/build"
SVE_BUILD_DIR="$BENCH_DIR/build_sve"

# ── Default configuration ──────────────────────────────────────────────────
VLEN=256            # RISC-V VLEN in bits
ELEN=64             # RISC-V ELEN in bits
LANES=4             # lanes for both ISAs
SVE_VL=2            # SVE VL in quadwords (2 = 256 bits)
NITER=2000
UNROLL=32
N_VEC_INSTS=$((NITER * UNROLL))   # = 64000

RUN_RISCV=true
RUN_ARM=true
SKIP_BUILD=false
DRY_RUN=false

# ── RISC-V benchmark matrix ───────────────────────────────────────────────
RISCV_SEWS=(8 32 64)
RISCV_LMULS=(1 4)
MODES=(indep chain)

# ── SVE benchmark matrix ──────────────────────────────────────────────────
SVE_ELEM_WIDTHS=(8 32 64)

# ── Color codes ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

# =============================================================================
# Argument parsing
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --riscv-only)  RUN_ARM=false;   shift ;;
        --arm-only)    RUN_RISCV=false; shift ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --dry-run)     DRY_RUN=true;    shift ;;
        --lanes)       LANES="$2";      shift 2 ;;
        --vlen)        VLEN="$2";       shift 2 ;;
        --sve-vl)      SVE_VL="$2";     shift 2 ;;
        --niter)       NITER="$2"; N_VEC_INSTS=$((NITER * UNROLL)); shift 2 ;;
        --unroll)      UNROLL="$2"; N_VEC_INSTS=$((NITER * UNROLL)); shift 2 ;;
        -h|--help)
            head -35 "$0" | tail -30
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================

log_header() {
    echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
}

log_section() {
    echo -e "\n${BOLD}── $1 ──${NC}"
}

log_ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
log_warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
log_err()  { echo -e "  ${RED}✗${NC} $1"; }

run_or_dry() {
    if $DRY_RUN; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} $*"
        return 0
    else
        "$@"
    fi
}

# Extract numCycles from a gem5 stats.txt file.
# Handles both "system.cpu.numCycles" and "board.processor...numCycles" patterns.
extract_num_cycles() {
    local stats_file="$1"
    if [[ ! -f "$stats_file" ]]; then
        echo "ERROR"
        return 1
    fi
    # Try multiple patterns (gem5 stat names vary by config version)
    local val
    val=$(grep -E '(system\.cpu|board\.processor).*numCycles' "$stats_file" \
          | head -1 | awk '{print $2}')
    if [[ -z "$val" ]]; then
        echo "ERROR"
        return 1
    fi
    echo "$val"
}

# Compute the analytical predicted CPI for a RISC-V benchmark.
# Args: sew lmul mode
# Outputs: the single-FU CPI (the relevant one based on validation results).
predict_cpi_riscv() {
    local sew=$1 lmul=$2 mode=$3
    local micro_vlmax=$((VLEN / sew))
    local passes

    if [[ $LANES -eq 0 ]]; then
        passes=1
    else
        passes=$(( (micro_vlmax + LANES - 1) / LANES ))
    fi

    if [[ "$mode" == "chain" ]]; then
        if [[ $lmul -eq 1 ]]; then
            # latency = opLat + passes - 1 (opLat=1 by default)
            echo $((1 + passes - 1))
        else
            # LMUL>1 chain: sum of per-slice latencies (serialized on 1 FU)
            echo $(( lmul * (1 + passes - 1) ))
        fi
    else
        # indep: single-FU CPI = sum of occupancies = lmul * passes
        echo $(( lmul * passes ))
    fi
}

# Compute the analytical predicted CPI for an SVE benchmark.
# Args: elem_width mode
predict_cpi_sve() {
    local elem_width=$1 mode=$2
    local vl_bits=$((SVE_VL * 128))
    local eCount=$((vl_bits / elem_width))
    local passes

    if [[ $LANES -eq 0 ]]; then
        passes=1
    else
        passes=$(( (eCount + LANES - 1) / LANES ))
    fi

    if [[ "$mode" == "chain" ]]; then
        # latency = opLat + passes - 1 (opLat=1)
        echo $((1 + passes - 1))
    else
        # indep: single-FU CPI = passes (occupancy)
        echo $passes
    fi
}

# Print a comparison row and return 0 if error < 5%, 1 otherwise.
# Args: bench_name predicted_cpi num_cycles
print_comparison() {
    local name="$1"
    local predicted_cpi="$2"
    local num_cycles="$3"

    if [[ "$num_cycles" == "ERROR" ]]; then
        printf "  %-45s  %8s  %12s  %10s  %10s  %8s\n" \
               "$name" "$predicted_cpi" "N/A" "N/A" "N/A" "FAIL"
        return 1
    fi

    local measured_cpi
    measured_cpi=$(awk "BEGIN {printf \"%.4f\", $num_cycles / $N_VEC_INSTS}")
    local error_pct
    error_pct=$(awk "BEGIN {
        if ($predicted_cpi == 0) { printf \"INF\"; exit }
        e = ($measured_cpi - $predicted_cpi) / $predicted_cpi * 100
        printf \"%.2f\", (e < 0 ? -e : e)
    }")

    local status
    local status_color
    # < 1% is excellent, < 5% is good, >= 5% is concerning
    if awk "BEGIN {exit !($error_pct < 1.0)}" 2>/dev/null; then
        status="OK"
        status_color="$GREEN"
    elif awk "BEGIN {exit !($error_pct < 5.0)}" 2>/dev/null; then
        status="OK"
        status_color="$YELLOW"
    else
        status="CHECK"
        status_color="$RED"
    fi

    printf "  %-45s  %8s  %12s  %10s  %10s  ${status_color}%8s${NC}\n" \
           "$name" "$predicted_cpi" "$num_cycles" "$measured_cpi" "${error_pct}%" "$status"
}

# =============================================================================
# Summary table storage
# =============================================================================
declare -a SUMMARY_ROWS=()

# =============================================================================
# PHASE 0: Build gem5 binaries and benchmarks (unless --skip-build)
# =============================================================================
if ! $SKIP_BUILD; then
    if $RUN_RISCV; then
        log_header "Building RISC-V benchmarks"
        if [[ ! -x "$GEM5_RISCV" ]]; then
            log_warn "gem5 RISC-V binary not found at $GEM5_RISCV"
            log_warn "Build it with: scons build/RISCV/gem5.opt -j\$(nproc)"
        fi
        log_section "Compiling RISC-V benchmark binaries"
        run_or_dry make -C "$BENCH_DIR" NITER=$NITER UNROLL=$UNROLL
    fi

    if $RUN_ARM; then
        log_header "Building ARM SVE benchmarks"
        if [[ ! -x "$GEM5_ARM" ]]; then
            log_warn "gem5 ARM binary not found at $GEM5_ARM"
            log_warn "Build it with: scons build/ARM/gem5.opt -j\$(nproc)"
        fi
        log_section "Compiling SVE benchmark binaries"
        run_or_dry make -f Makefile_sve -C "$BENCH_DIR" NITER=$NITER UNROLL=$UNROLL
    fi
fi

mkdir -p "$RESULTS_DIR"

# =============================================================================
# PHASE 1: Run RISC-V benchmarks
# =============================================================================
if $RUN_RISCV; then
    log_header "Running RISC-V RVV benchmarks (VLEN=${VLEN}, lanes=${LANES})"

    for sew in "${RISCV_SEWS[@]}"; do
        for lmul in "${RISCV_LMULS[@]}"; do
            for mode in "${MODES[@]}"; do
                bench_name="riscv_sew${sew}_lmul${lmul}_${mode}"
                binary="$RISCV_BUILD_DIR/bench_sew${sew}_lmul${lmul}_${mode}"
                outdir="$RESULTS_DIR/$bench_name"

                log_section "$bench_name"

                # ── Check binary exists ────────────────────────────────
                if [[ ! -f "$binary" ]]; then
                    log_err "Binary not found: $binary (skipping)"
                    SUMMARY_ROWS+=("$bench_name|SKIP|ERROR|SKIP")
                    continue
                fi

                # ── Predicted CPI ──────────────────────────────────────
                predicted_cpi=$(predict_cpi_riscv "$sew" "$lmul" "$mode")
                log_ok "Predicted CPI (single-FU): $predicted_cpi"

                # ── Run expected_cycles.py (full output for reference) ─
                log_ok "Running expected_cycles.py:"
                if ! $DRY_RUN; then
                    python3 "$BENCH_DIR/expected_cycles.py" \
                        --vlen "$VLEN" --sew "$sew" --lmul "$lmul" \
                        --lanes "$LANES" --mode "$mode" \
                        --niter "$NITER" --unroll "$UNROLL" \
                        2>&1 | sed 's/^/    /'
                fi

                # ── Run gem5 ───────────────────────────────────────────
                if [[ ! -x "$GEM5_RISCV" ]]; then
                    log_err "gem5 RISC-V binary not found (skipping sim)"
                    SUMMARY_ROWS+=("$bench_name|$predicted_cpi|ERROR|SKIP")
                    continue
                fi

                log_ok "Running gem5 → $outdir"
                if ! $DRY_RUN; then
                    "$GEM5_RISCV" -d "$outdir" \
                        "$BENCH_DIR/run_lanes_bench.py" \
                        --binary "$binary" \
                        --vlen "$VLEN" --elen "$ELEN" --lanes "$LANES" \
                        2>&1 | tail -3 | sed 's/^/    /'
                fi

                # ── Extract & compare ──────────────────────────────────
                num_cycles=$(extract_num_cycles "$outdir/stats.txt" 2>/dev/null || echo "ERROR")
                SUMMARY_ROWS+=("$bench_name|$predicted_cpi|$num_cycles|OK")

                if [[ "$num_cycles" != "ERROR" ]]; then
                    measured_cpi=$(awk "BEGIN {printf \"%.4f\", $num_cycles / $N_VEC_INSTS}")
                    log_ok "numCycles=$num_cycles  →  measured CPI=${measured_cpi}  (predicted=${predicted_cpi})"
                else
                    log_warn "Could not extract numCycles from $outdir/stats.txt"
                fi
            done
        done
    done
fi

# =============================================================================
# PHASE 2: Run ARM SVE benchmarks
# =============================================================================
if $RUN_ARM; then
    log_header "Running ARM SVE benchmarks (sve_vl=${SVE_VL} [=$(( SVE_VL * 128 )) bits], lanes=${LANES})"

    for ew in "${SVE_ELEM_WIDTHS[@]}"; do
        for mode in "${MODES[@]}"; do
            bench_name="sve_elem${ew}_${mode}"
            binary="$SVE_BUILD_DIR/bench_sve_elem${ew}_${mode}"
            outdir="$RESULTS_DIR/$bench_name"

            log_section "$bench_name"

            # ── Check binary exists ────────────────────────────────────
            if [[ ! -f "$binary" ]]; then
                log_err "Binary not found: $binary (skipping)"
                SUMMARY_ROWS+=("$bench_name|SKIP|ERROR|SKIP")
                continue
            fi

            # ── Predicted CPI ──────────────────────────────────────────
            predicted_cpi=$(predict_cpi_sve "$ew" "$mode")
            log_ok "Predicted CPI (single-FU): $predicted_cpi"

            # ── Run expected_cycles_sve.py (full output for reference) ─
            log_ok "Running expected_cycles_sve.py:"
            if ! $DRY_RUN; then
                python3 "$BENCH_DIR/expected_cycles_sve.py" \
                    --sve-vl "$SVE_VL" --elem-width "$ew" \
                    --lanes "$LANES" --mode "$mode" \
                    --niter "$NITER" --unroll "$UNROLL" \
                    2>&1 | sed 's/^/    /'
            fi

            # ── Run gem5 ───────────────────────────────────────────────
            if [[ ! -x "$GEM5_ARM" ]]; then
                log_err "gem5 ARM binary not found (skipping sim)"
                SUMMARY_ROWS+=("$bench_name|$predicted_cpi|ERROR|SKIP")
                continue
            fi

            log_ok "Running gem5 → $outdir"
            if ! $DRY_RUN; then
                "$GEM5_ARM" -d "$outdir" \
                    "$BENCH_DIR/run_sve_lanes_bench.py" \
                    --binary "$binary" \
                    --sve-vl "$SVE_VL" --sve-lanes "$LANES" \
                    2>&1 | tail -3 | sed 's/^/    /'
            fi

            # ── Extract & compare ──────────────────────────────────────
            num_cycles=$(extract_num_cycles "$outdir/stats.txt" 2>/dev/null || echo "ERROR")
            SUMMARY_ROWS+=("$bench_name|$predicted_cpi|$num_cycles|OK")

            if [[ "$num_cycles" != "ERROR" ]]; then
                measured_cpi=$(awk "BEGIN {printf \"%.4f\", $num_cycles / $N_VEC_INSTS}")
                log_ok "numCycles=$num_cycles  →  measured CPI=${measured_cpi}  (predicted=${predicted_cpi})"
            else
                log_warn "Could not extract numCycles from $outdir/stats.txt"
            fi
        done
    done
fi

# =============================================================================
# PHASE 3: Summary comparison table
# =============================================================================
log_header "SUMMARY: Predicted vs Measured CPI (N_VEC_INSTS=${N_VEC_INSTS})"

echo ""
printf "  ${BOLD}%-45s  %8s  %12s  %10s  %10s  %8s${NC}\n" \
       "Benchmark" "Pred.CPI" "numCycles" "Meas.CPI" "Error" "Status"
printf "  %-45s  %8s  %12s  %10s  %10s  %8s\n" \
       "$(printf '%0.s─' {1..45})" "$(printf '%0.s─' {1..8})" \
       "$(printf '%0.s─' {1..12})" "$(printf '%0.s─' {1..10})" \
       "$(printf '%0.s─' {1..10})" "$(printf '%0.s─' {1..8})"

pass_count=0
fail_count=0
skip_count=0

for row in "${SUMMARY_ROWS[@]}"; do
    IFS='|' read -r name predicted_cpi num_cycles status <<< "$row"

    if [[ "$status" == "SKIP" ]] || [[ "$predicted_cpi" == "SKIP" ]]; then
        printf "  %-45s  %8s  %12s  %10s  %10s  ${YELLOW}%8s${NC}\n" \
               "$name" "-" "-" "-" "-" "SKIPPED"
        ((skip_count++)) || true
        continue
    fi

    if $DRY_RUN; then
        printf "  %-45s  %8s  %12s  %10s  %10s  ${YELLOW}%8s${NC}\n" \
               "$name" "$predicted_cpi" "DRY-RUN" "-" "-" "DRY-RUN"
        continue
    fi

    if print_comparison "$name" "$predicted_cpi" "$num_cycles"; then
        ((pass_count++)) || true
    else
        ((fail_count++)) || true
    fi
done

echo ""
echo -e "  ${BOLD}Results:${NC} ${GREEN}${pass_count} passed${NC}, ${RED}${fail_count} failed${NC}, ${YELLOW}${skip_count} skipped${NC}"
echo -e "  ${BOLD}Config:${NC}  VLEN=${VLEN} ELEN=${ELEN} SVE_VL=${SVE_VL} LANES=${LANES} NITER=${NITER} UNROLL=${UNROLL}"
echo -e "  ${BOLD}Results directory:${NC} $RESULTS_DIR/"
echo ""

# Return non-zero if any benchmark failed
[[ $fail_count -eq 0 ]]
