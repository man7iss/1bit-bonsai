#!/bin/bash
set -euo pipefail

# Experiment 2: Sustained throughput over time.
# Runs llama-bench repeatedly for DURATION seconds, logging tok/s at each iteration.
# Captures die temperature via powermetrics (requires sudo for temp, optional).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.env"

if [ -z "${LLAMA_BENCH}" ] || [ ! -x "${LLAMA_BENCH}" ]; then
    echo "ERROR: llama-bench not found. Run setup.sh first."
    exit 1
fi

DURATION="${DURATION:-300}"  # 5 minutes default
MODEL="${1:-${BONSAI_GGUF}}"
LABEL="${2:-bonsai}"
OUTPUT="${RESULTS_DIR}/sustained_${LABEL}.csv"

if [ -z "${MODEL}" ] || [ ! -f "${MODEL}" ]; then
    echo "ERROR: Model file not found: ${MODEL}"
    echo "Usage: $0 <model.gguf> [label]"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "=== Experiment 2: Sustained Throughput ==="
echo "Model: $(basename "${MODEL}")"
echo "Duration: ${DURATION}s"
echo "Output: ${OUTPUT}"
echo ""

# CSV header
echo "elapsed_s,tg128_tok_s,model" > "${OUTPUT}"

START=$(date +%s)
ITER=0

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [ ${ELAPSED} -ge ${DURATION} ]; then
        break
    fi

    # Run a single llama-bench iteration (tg128 only for speed)
    TPS=$("${LLAMA_BENCH}" \
        -m "${MODEL}" \
        -p 0 -n 128 \
        -ngl 99 \
        -r 1 \
        -o csv 2>/dev/null | tail -1 | awk -F',' '{print $(NF-1)}')

    if [ -n "${TPS}" ] && [ "${TPS}" != "avg_ts" ]; then
        echo "${ELAPSED},${TPS},${LABEL}" >> "${OUTPUT}"
        ITER=$((ITER + 1))
        printf "\r  [%3ds / %ds] iter=%d  tg128=%.1f tok/s" \
            "${ELAPSED}" "${DURATION}" "${ITER}" "${TPS}"
    fi
done

echo ""
echo ""
echo "=== ${ITER} measurements written to: ${OUTPUT} ==="
echo ""
echo "To capture thermal data simultaneously (requires sudo):"
echo "  sudo powermetrics --samplers smc -i 10000 | grep -i 'die temp' > thermal.log &"
