#!/bin/bash
set -euo pipefail

# Experiment 1: Burst throughput across quantization levels.
# Runs llama-bench with pp512/tg128 for each model, outputs CSV.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.env"

if [ -z "${LLAMA_BENCH}" ] || [ ! -x "${LLAMA_BENCH}" ]; then
    echo "ERROR: llama-bench not found. Run setup.sh first."
    exit 1
fi

REPS="${REPS:-5}"
OUTPUT="${RESULTS_DIR}/throughput.csv"
mkdir -p "${RESULTS_DIR}"

echo "=== Experiment 1: Burst Throughput ==="
echo "Repetitions: ${REPS}"
echo "Output: ${OUTPUT}"
echo ""

# Collect all model files to benchmark
MODELS=()

# Bonsai model (requires PrismML fork)
if [ -n "${BONSAI_GGUF}" ] && [ -f "${BONSAI_GGUF}" ]; then
    MODELS+=("${BONSAI_GGUF}")
    echo "Found: Bonsai ${BONSAI_MODEL} (${BONSAI_GGUF})"
fi

# Additional Qwen3-8B quantizations (download separately)
QWEN_DIR="${WORK_DIR}/qwen3-8b"
if [ -d "${QWEN_DIR}" ]; then
    for f in "${QWEN_DIR}"/*.gguf; do
        if [ -f "$f" ]; then
            MODELS+=("$f")
            echo "Found: $(basename "$f")"
        fi
    done
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No model files found. Run setup.sh first."
    exit 1
fi

echo ""
echo "Running llama-bench for ${#MODELS[@]} model(s)..."
echo ""

# Run llama-bench with CSV output for each model
FIRST=1
for model in "${MODELS[@]}"; do
    echo "--- Benchmarking: $(basename "$model") ---"

    if [ $FIRST -eq 1 ]; then
        # First run: include CSV header
        "${LLAMA_BENCH}" \
            -m "$model" \
            -p 512 -n 128 \
            -ngl 99 \
            -r "${REPS}" \
            -o csv 2>/dev/null > "${OUTPUT}"
        FIRST=0
    else
        # Subsequent runs: append without header
        "${LLAMA_BENCH}" \
            -m "$model" \
            -p 512 -n 128 \
            -ngl 99 \
            -r "${REPS}" \
            -o csv 2>/dev/null | tail -n +2 >> "${OUTPUT}"
    fi

    echo ""
done

echo "=== Results written to: ${OUTPUT} ==="
echo ""

# Print summary using the bench_parse module
cd "${SCRIPT_DIR}"
uv run --with pytest python3 -c "
import sys
sys.path.insert(0, '.')
from bench_parse import parse_csv, format_comparison_table
with open('${OUTPUT}') as f:
    results = parse_csv(f.read())
print(format_comparison_table(results))
" 2>/dev/null || echo "(Install dependencies to see formatted summary)"
