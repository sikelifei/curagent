"""Run curagent on one Oolong-real DnD example."""

from __future__ import annotations

import argparse
import json

from recursive_agent.envs import run_registered_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_api.local.yaml")
    parser.add_argument("--oolong-root", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="oolongbench/oolong-real")
    parser.add_argument("--config-name", default="dnd")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-id", type=int, default=0)
    parser.add_argument("--agent-max-steps", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-concurrent-subagents", type=int, default=2)
    parser.add_argument("--max-run-seconds", type=float, default=900)
    parser.add_argument("--max-observation-chars", type=int, default=8000)
    parser.add_argument("--trace-json", default=None)
    args = parser.parse_args()

    run = run_registered_environment(
        "oolong",
        model_config=args.config,
        environment_kwargs={
            "oolong_root": args.oolong_root,
            "data_path": args.data_path,
            "dataset_name": args.dataset_name,
            "config_name": args.config_name,
            "split": args.split,
            "instance_id": args.instance_id,
        },
        agent_kwargs={
            "max_steps": args.agent_max_steps,
            "max_depth": args.max_depth,
            "max_concurrent_subagents": args.max_concurrent_subagents,
            "max_run_seconds": args.max_run_seconds,
            "max_observation_chars": args.max_observation_chars,
        },
    )
    trace = run.to_trace_dict()
    print(run.agent_result.answer)
    print(json.dumps(run.environment_report, ensure_ascii=False, indent=2, default=str))
    if args.trace_json:
        from pathlib import Path

        path = Path(args.trace_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
