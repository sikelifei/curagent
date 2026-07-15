"""Run a small recursive smoke test against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse

from recursive_agent import RecursiveAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    args = parser.parse_args()

    agent = RecursiveAgent.from_config(
        args.config,
        max_steps=4,
        max_depth=1,
        max_concurrent_subagents=2,
        max_run_seconds=180,
    )
    result = agent.run(
        task=(
            "Compute 17 * 23. Delegate an independent verification to one subagent, "
            "then return a concise final answer containing the number."
        ),
        context={"purpose": "curagent end-to-end smoke test"},
    )
    print(result.answer)
    print(
        f"status={result.status} steps={result.steps} "
        f"calls={result.usage.total_calls} tokens="
        f"{result.usage.total_input_tokens + result.usage.total_output_tokens}"
    )


if __name__ == "__main__":
    main()
