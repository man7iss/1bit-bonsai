#!/bin/bash
set -euo pipefail

# Experiment 3: Perplexity across quantization levels.
# Runs llama-perplexity on WikiText-2 for each model.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.env"

if [ -z "${LLAMA_PPL}" ] || [ ! -x "${LLAMA_PPL}" ]; then
    echo "ERROR: llama-perplexity not found. Run setup.sh first."
    exit 1
fi

if [ ! -f "${WIKITEXT}" ]; then
    echo "ERROR: WikiText-2 not found at ${WIKITEXT}. Run setup.sh first."
    exit 1
fi

CTX_SIZE="${CTX_SIZE:-2048}"
OUTPUT="${RESULTS_DIR}/perplexity.csv"
RAW_DIR="${RESULTS_DIR}/ppl_raw"
mkdir -p "${RESULTS_DIR}" "${RAW_DIR}"

echo "=== Experiment 3: Perplexity ==="
echo "Context size: ${CTX_SIZE}"
echo "WikiText-2: ${WIKITEXT}"
echo "Output: ${OUTPUT}"
echo ""

# CSV header
echo "model,format,size_bytes,perplexity,stderr" > "${OUTPUT}"

# Collect models
MODELS=()
LABELS=()

if [ -n "${BONSAI_GGUF}" ] && [ -f "${BONSAI_GGUF}" ]; then
    MODELS+=("${BONSAI_GGUF}")
    LABELS+=("Bonsai-${BONSAI_MODEL}")
fi

QWEN_DIR="${WORK_DIR}/qwen3-8b"
if [ -d "${QWEN_DIR}" ]; then
    for f in "${QWEN_DIR}"/*.gguf; do
        if [ -f "$f" ]; then
            MODELS+=("$f")
            LABELS+=("$(basename "$f" .gguf)")
        fi
    done
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No model files found."
    exit 1
fi

echo "Found ${#MODELS[@]} model(s)."
echo ""

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    label="${LABELS[$i]}"
    raw_file="${RAW_DIR}/${label}.txt"
    size=$(stat -f%z "$model" 2>/dev/null || stat --printf="%s" "$model" 2>/dev/null || echo 0)

    echo "--- Measuring perplexity: ${label} ---"
    echo "    Model: $(basename "$model") ($(echo "scale=2; $size / 1000000000" | bc) GB)"

    "${LLAMA_PPL}" \
        -m "$model" \
        -f "${WIKITEXT}" \
        -ngl 99 \
        --ctx-size "${CTX_SIZE}" \
        2>&1 | tee "${raw_file}"

    # Extract final perplexity from output
    PPL=$(grep -oE 'PPL\s*=\s*[0-9.]+' "${raw_file}" | tail -1 | grep -oE '[0-9.]+$' || echo "")
    STDERR=$(grep -oE '\+/-\s*[0-9.]+' "${raw_file}" | tail -1 | grep -oE '[0-9.]+$' || echo "0")

    if [ -z "${PPL}" ]; then
        # Try alternative format
        PPL=$(grep -oE 'perplexity\s*=\s*[0-9.]+' "${raw_file}" | tail -1 | grep -oE '[0-9.]+$' || echo "N/A")
    fi

    echo "${label},$(basename "$model"),${size},${PPL},${STDERR}" >> "${OUTPUT}"
    echo "    Perplexity: ${PPL} +/- ${STDERR}"
    echo ""
done

echo "=== Results written to: ${OUTPUT} ==="
cat "${OUTPUT}"
