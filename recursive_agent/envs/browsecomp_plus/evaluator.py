"""Run the separate local-model smoke evaluator over saved agent outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .dataset import DEFAULT_DATA_PATH, load_gold_answers
from .runner import _write_json, build_summary
from .scoring import judge_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-config",
        default="configs/model_api.local.yaml",
    )
    parser.add_argument("--gold-data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error("attempts must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    runs_dir = output_dir / "runs" if (output_dir / "runs").is_dir() else output_dir
    paths = sorted(runs_dir.glob("run_*.json"))
    all_records: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        all_records.append((path, record))
    records = [
        (path, record)
        for path, record in all_records
        if record.get("status") == "completed"
    ]

    answers = (
        load_gold_answers(
            args.gold_data,
            query_ids=(str(record["query_id"]) for _, record in records),
        )
        if records
        else {}
    )
    started = time.time()
    for path, record in records:
        current = record.get("local_evaluator")
        if current and not current.get("error") and not args.force:
            continue
        try:
            judge = judge_answer(
                model_config=args.model_config,
                question=str(record["query"]),
                correct_answer=answers[str(record["query_id"])],
                response=str(record.get("final_answer", "")),
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_attempts=args.attempts,
            )
            record["local_evaluator"] = judge.to_dict()
        except Exception as exc:
            record["local_evaluator"] = {
                "correct": False,
                "score": 0,
                "reason": "Local evaluator call failed.",
                "response": "",
                "error": f"{type(exc).__name__}: {exc}",
                "model": "unknown",
                "attempts": 0,
            }
        _write_json(path, record)
        print(
            f"evaluated query_id={record['query_id']} "
            f"correct={record['local_evaluator']['correct']} "
            f"error={record['local_evaluator']['error']}",
            flush=True,
        )

    refreshed = []
    for path, _ in all_records:
        with path.open(encoding="utf-8") as handle:
            refreshed.append(json.load(handle))
    summary = build_summary(
        refreshed,
        requested=len(all_records),
        elapsed_seconds=time.time() - started,
    )
    summary_path = output_dir / "summary.json" if (output_dir / "runs").is_dir() else output_dir.parent / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
