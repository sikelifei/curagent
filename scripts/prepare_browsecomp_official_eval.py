"""Prepare prompt snapshots and local ground truth for official BrowseComp eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recursive_agent.envs.browsecomp_plus.dataset import (
    DEFAULT_DATA_PATH,
    DEFAULT_QUERIES_PATH,
    load_gold_answers,
    load_queries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES_PATH))
    parser.add_argument("--gold-data", default=str(DEFAULT_DATA_PATH))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_paths = sorted(runs_dir.glob("*.json"))
    if not run_paths:
        raise ValueError(f"No run JSON files in {runs_dir}")

    runs = [_read_json(path) for path in run_paths]
    query_ids = [str(run["query_id"]) for run in runs]
    query_map = {item.query_id: item.query for item in load_queries(args.queries)}
    answers = load_gold_answers(args.gold_data, query_ids=query_ids)

    gold_path = output_dir / "official_ground_truth.jsonl"
    with gold_path.open("w", encoding="utf-8") as stream:
        for query_id in query_ids:
            json.dump(
                {
                    "query_id": query_id,
                    "query": query_map[query_id],
                    "answer": answers[query_id],
                },
                stream,
                ensure_ascii=False,
            )
            stream.write("\n")

    prompts = [_prompt_record(run) for run in runs]
    _write_json(
        output_dir / "prompt_result_manifest.json",
        {
            "runs_dir": str(runs_dir),
            "ground_truth": str(gold_path),
            "count": len(prompts),
            "records": prompts,
        },
    )
    print(f"Prepared {len(prompts)} prompt snapshots and {gold_path}")


def _prompt_record(run: dict[str, Any]) -> dict[str, Any]:
    trajectory_path = Path(str(run["trajectory"]["path"]))
    trace_payload = _read_json(trajectory_path)
    trace = (trace_payload.get("agent_result") or {}).get("trace") or {}
    return {
        "query_id": str(run["query_id"]),
        "query": str(run["query"]),
        "system_prompt": trace.get("system_prompt"),
        "task_prompt": trace.get("task"),
        "final_answer": run.get("final_answer"),
        "status": run.get("status"),
        "search_calls": run.get("search_calls"),
        "recursive_calls": run.get("recursive_calls"),
        "trajectory_path": str(trajectory_path),
        "run_path": str(run.get("trajectory", {}).get("path", "")),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
