# BrowseComp-Plus BM25 adapter

This environment gives root and delegated agents only a question and one
search(query) tool. The tool calls the official BrowseComp-Plus BM25 MCP server,
which is started with k=5 and the default 512-token snippet limit. No answer,
gold/evidence documents, qrels, evaluator prompt, or judge response enters an
agent context.

BrowseCompTrace reserves each call under a lock before contacting MCP. Root and
all subagents share the same environment tool object, so parallel agents cannot
exceed the global budget and retrieved_docids is the global de-duplicated union.
The runner derives recursion counts and depth from actual child AgentTrace
objects rather than prompt wording.

Install the client dependency:

    python -m pip install -e '.[browsecomp]'

Provider-specific request options belong in the local YAML rather than the
adapter. For example, if a DashScope reasoning model consumes `max_tokens`
without producing final `content`, disable thinking in
`configs/model_api.local.yaml`:

    sampling_args:
      extra_body:
        enable_thinking: false

If topics-qrels/queries.tsv is absent, derive only its query_id/query columns
from the local official parquet:

    python -m recursive_agent.envs.browsecomp_plus.dataset --generate-queries

Start retrieval from the BrowseComp-Plus checkout:

    CUDA_VISIBLE_DEVICES="" python searcher/mcp_server.py \
      --searcher-type bm25 \
      --index-path indexes/bm25 \
      --k 5 \
      --snippet-max-tokens 512 \
      --port 8080

Run the smoke test from the curagent checkout:

    python -m recursive_agent.envs.browsecomp_plus.runner \
      --queries /data2/zhangwenjian/agent/bench/BrowseComp-Plus/topics-qrels/queries.tsv \
      --model-config configs/model_api.local.yaml \
      --bm25-url http://127.0.0.1:8080 \
      --output-dir outputs/browsecomp_plus_smoke \
      --limit 5 \
      --max-search-calls 20 \
      --max-recursion-depth 2 \
      --max-subagents-per-agent 4 \
      --concurrency 1

The runner invokes the local evaluator separately after each agent run. It can
also be rerun independently:

    python -m recursive_agent.envs.browsecomp_plus.evaluator \
      --output-dir outputs/browsecomp_plus_smoke \
      --model-config configs/model_api.local.yaml

Each official-compatible result is under output-dir/runs; full curagent traces
are under output-dir/trajectories, and summary.json contains aggregate and
per-question smoke metrics. The official evaluator can read the runs directory
directly when supplied its required decrypted ground-truth JSONL and judge
model.

After the smoke test, run all 830 questions resumably without the development
judge:

    python -m recursive_agent.envs.browsecomp_plus.runner \
      --queries /data2/zhangwenjian/agent/bench/BrowseComp-Plus/topics-qrels/queries.tsv \
      --model-config configs/model_api.local.yaml \
      --bm25-url http://127.0.0.1:8080 \
      --output-dir outputs/browsecomp_plus_full \
      --limit 830 \
      --resume \
      --max-search-calls 20 \
      --max-recursion-depth 2 \
      --max-subagents-per-agent 4 \
      --concurrency 1 \
      --skip-local-evaluator

From the BrowseComp-Plus checkout, pass the generated `runs` directory directly
to the official evaluator:

    python scripts_evaluation/evaluate_run.py \
      --input_dir /data2/zhangwenjian/agent/curagent/outputs/browsecomp_plus_full/runs \
      --ground_truth data/browsecomp_plus_decrypted.jsonl \
      --qrel_evidence topics-qrels/qrel_evidence.txt \
      --eval_dir evals/curagent_recursive

本地模型 evaluator 的结果不能直接作为 BrowseComp-Plus 官方榜单结果。
正式全量评测和提交时，应使用 BrowseComp-Plus 官方 evaluation script
及其规定的 judge 模型和版本。
