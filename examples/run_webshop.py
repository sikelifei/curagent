"""Run curagent on one ReCode WebShop dataset episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recursive_agent.envs import run_registered_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--recode-root", default=None)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--instance-id", type=int, default=0)
    parser.add_argument("--env-max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--agent-max-steps", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-concurrent-subagents", type=int, default=4)
    parser.add_argument("--max-run-seconds", type=float, default=900)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument(
        "--prompt-file",
        help="Optional UTF-8 task template containing the literal {instruction} placeholder.",
    )
    parser.add_argument(
        "--trace-file",
        help="Optional path for the full credential-free JSON execution trace.",
    )
    args = parser.parse_args()

    environment_kwargs = {
        "recode_root": args.recode_root,
        "split": args.split,
        "instance_id": args.instance_id,
        "max_steps": args.env_max_steps,
        "seed": args.seed,
    }
    if args.prompt_file:
        environment_kwargs["prompt_template"] = Path(args.prompt_file).read_text(
            encoding="utf-8"
        )

    model_overrides = {}
    if args.model_name:
        model_overrides["model_name"] = args.model_name
    run = run_registered_environment(
        "webshop",
        model_config=args.config,
        environment_kwargs=environment_kwargs,
        agent_kwargs={
            "max_steps": args.agent_max_steps,
            "max_depth": args.max_depth,
            "max_concurrent_subagents": args.max_concurrent_subagents,
            "max_run_seconds": args.max_run_seconds,
            "max_observation_chars": args.max_observation_chars,
        },
        model_overrides=model_overrides,
    )
    if args.trace_file:
        trace_path = Path(args.trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(run.to_trace_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"trace_file={trace_path.resolve()}")
    print(run.agent_result.answer)
    print(
        json.dumps(
            {
                "status": run.agent_result.status,
                "agent_steps": run.agent_result.steps,
                "usage": run.agent_result.usage.to_dict(),
                "environment": {
                    key: value
                    for key, value in run.environment_report.items()
                    if key != "trajectory"
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
