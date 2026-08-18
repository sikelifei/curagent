"""Run the official BrowseComp OpenAI evaluator with a curagent model config."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from recursive_agent.config import load_model_config


OFFICIAL_EVALUATOR = Path(
    "/data2/zhangwenjian/agent/bench/BrowseComp-Plus/"
    "scripts_evaluation/evaluate_with_openai.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--qrel-evidence", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    backend, config = load_model_config(args.model_config)
    if backend != "openai":
        raise ValueError("Official OpenAI evaluator requires an OpenAI-compatible config")
    api_key = str(config.get("api_key", ""))
    base_url = str(config.get("base_url", ""))
    model = str(config.get("model_name", ""))
    if not api_key or not base_url or not model:
        raise ValueError("Judge config requires api_key, base_url, and model")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    sys.argv = [
        str(OFFICIAL_EVALUATOR),
        "--input_dir",
        args.input_dir,
        "--ground_truth",
        args.ground_truth,
        "--eval_dir",
        args.eval_dir,
        "--model",
        model,
        "--max_output_tokens",
        str(args.max_output_tokens),
        "--qrel_evidence",
        args.qrel_evidence,
    ]
    if args.force:
        sys.argv.append("--force")
    runpy.run_path(str(OFFICIAL_EVALUATOR), run_name="__main__")


if __name__ == "__main__":
    main()
