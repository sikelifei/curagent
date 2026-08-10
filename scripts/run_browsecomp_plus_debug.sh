#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BCOMP_ROOT="${BCOMP_ROOT:-/data2/zhangwenjian/agent/bench/BrowseComp-Plus}"
PYTHON_BIN="${PYTHON_BIN:-/data2/zhangwenjian/miniconda3/envs/bcomp/bin/python}"
JAVA_HOME="${JAVA_HOME:-/data2/zhangwenjian/miniconda3/envs/bcomp/lib/jvm}"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/model_api.local.yaml}"

PORT="${PORT:-38081}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-3}"
SNIPPET_MAX_CHARS="${SNIPPET_MAX_CHARS:-1000}"
RUN_TAG="${RUN_TAG:-current_prompt_chars1000_flash_n3_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/bcomp/${RUN_TAG}}"

mkdir -p "${OUTPUT_DIR}"
BM25_LOG="${OUTPUT_DIR}/bm25_server.log"

if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "Port ${PORT} is already in use. Choose another PORT." >&2
    exit 1
fi

bm25_pid=""
cleanup() {
    if [[ -n "${bm25_pid}" ]] && kill -0 "${bm25_pid}" 2>/dev/null; then
        kill "${bm25_pid}" 2>/dev/null || true
        wait "${bm25_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting tokenizer-free BM25 on port ${PORT}"
(
    cd "${BCOMP_ROOT}"
    env \
        JAVA_HOME="${JAVA_HOME}" \
        PATH="${JAVA_HOME}/bin:${PATH}" \
        "${PYTHON_BIN}" -u searcher/mcp_server.py \
        --searcher-type bm25 \
        --index-path indexes/bm25 \
        --k 5 \
        --snippet-max-tokens -1 \
        --port "${PORT}"
) >"${BM25_LOG}" 2>&1 &
bm25_pid=$!

server_ready=0
for _ in $(seq 1 60); do
    if ! kill -0 "${bm25_pid}" 2>/dev/null; then
        echo "BM25 exited during startup. See ${BM25_LOG}" >&2
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        exec 3>&-
        server_ready=1
        break
    fi
    sleep 1
done

if [[ "${server_ready}" != "1" ]]; then
    echo "BM25 did not become ready. See ${BM25_LOG}" >&2
    exit 1
fi

echo "Running ${LIMIT} BrowseComp-Plus question(s)"
echo "Output: ${OUTPUT_DIR}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u -m recursive_agent.envs.browsecomp_plus.runner \
    --queries "${BCOMP_ROOT}/topics-qrels/queries.tsv" \
    --model-config "${MODEL_CONFIG}" \
    --bm25-url "http://127.0.0.1:${PORT}/mcp" \
    --output-dir "${OUTPUT_DIR}" \
    --start-index "${START_INDEX}" \
    --limit "${LIMIT}" \
    --max-search-calls 20 \
    --max-recursion-depth 2 \
    --max-concurrent-subagents 4 \
    --max-subagents-per-agent 4 \
    --agent-max-steps 20 \
    --max-run-seconds 900 \
    --max-observation-chars 16000 \
    --snippet-max-chars "${SNIPPET_MAX_CHARS}" \
    --concurrency 1 \
    --skip-local-evaluator \
    "$@"

echo "Finished. Step logs: ${OUTPUT_DIR}/logs/*_steps.jsonl"
