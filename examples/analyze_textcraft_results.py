"""Analyze TextCraft JSONL results for recursion quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recursive_agent.envs.textcraft_synth.trace_analysis import (
    aggregate_textcraft_results,
    analyze_textcraft_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="One or more result JSONL files")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    analyses = []
    for raw_path in args.results:
        path = Path(raw_path)
        rows = _read_jsonl(path)
        metrics = [analyze_textcraft_result(row) for row in rows]
        analyses.append(
            {
                "path": str(path),
                "summary": aggregate_textcraft_results(metrics),
                "rows": metrics,
            }
        )
    result = {"analyses": analyses}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


if __name__ == "__main__":
    main()
