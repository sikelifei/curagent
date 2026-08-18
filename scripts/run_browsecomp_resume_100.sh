#!/usr/bin/env bash
set -euo pipefail

output_dir="output/bcomp/qwen3_4b_official_100"
query_file="/data2/zhangwenjian/agent/bench/BrowseComp-Plus/topics-qrels/queries.tsv"
python_bin="/data2/zhangwenjian/miniconda3/envs/bcomp/bin/python"

while true; do
    completed=$(find "${output_dir}/runs" -name '*.json' -type f 2>/dev/null | wc -l)
    if [[ "${completed}" -ge 100 ]]; then
        exit 0
    fi
    "${python_bin}" -u -m recursive_agent.envs.browsecomp_plus.runner \
        --queries "${query_file}" \
        --model-config configs/model_api_qwen3_4b_instruct_2507_vllm.yaml \
        --bm25-url http://127.0.0.1:38081/mcp \
        --output-dir "${output_dir}" \
        --start-index 0 --limit 100 --resume \
        --max-search-calls 20 --max-recursion-depth 2 \
        --max-concurrent-subagents 4 --max-subagents-per-agent 4 \
        --agent-max-steps 20 --max-run-seconds 300 \
        --max-observation-chars 16000 --snippet-max-chars 1000 \
        --concurrency 4 --skip-local-evaluator
done
