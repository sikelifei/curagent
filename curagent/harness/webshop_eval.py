"""Small or batch ReCode WebShop evaluation for the simplified harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from curagent.core.agent import AgentNode
from curagent.core.budget import SharedBudget
from curagent.core.model import ToolCallingModel
from curagent.core.trace import TraceRecorder
from curagent.core.types import AgentLimits
from curagent.environments.recode_webshop import ReCodeWebShopEnvironment
from curagent.models.openai_compatible import OpenAICompatibleModel, load_model_config


def _trace_metrics(trace: Sequence[dict[str, Any]]) -> dict[str, int]:
    tool_calls = 0
    children = 0
    for event in trace:
        call = event.get("parsed_tool_call")
        if not isinstance(call, dict):
            continue
        tool_calls += 1
        if call.get("name") == "spawn_agent":
            children += 1
        elif call.get("name") == "spawn_agents":
            arguments = call.get("arguments")
            specs = arguments.get("specs") if isinstance(arguments, dict) else None
            if isinstance(specs, list):
                children += len(specs)
    return {"tool_calls_observed": tool_calls, "children_spawned": children}


async def run_episode(
    *,
    task_id: str,
    split: str,
    project_root: str,
    model: ToolCallingModel,
    limits: AgentLimits,
    environment_max_steps: int = 30,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = ReCodeWebShopEnvironment(
        project_root=project_root,
        split=split,
        max_steps=environment_max_steps,
    )
    trace = TraceRecorder()
    budget = SharedBudget(limits)
    started = datetime.now(timezone.utc)
    try:
        observation = await environment.reset({"task_id": task_id, "split": split})
        observation_value = observation.to_dict() if hasattr(observation, "to_dict") else observation
        metadata = observation_value.get("metadata", {}) if isinstance(observation_value, dict) else {}
        instruction = str(metadata.get("instruction") or getattr(observation, "text", observation_value))
        agent = AgentNode(
            agent_id="root",
            task=instruction,
            context={"task_id": task_id, "split": split},
            environment=environment,
            model=model,
            limits=limits,
            budget=budget,
            trace=trace,
        )
        result = await agent.run()
        reward = environment.reward()
        final_observation = await environment.observe()
        final_observation_value = (
            final_observation.to_dict()
            if hasattr(final_observation, "to_dict")
            else final_observation
        )
        snapshot = await budget.snapshot()
        events = trace.all()
        episode = {
            "task_id": task_id,
            "status": "ok" if result.error is None else "error",
            "result": result.result,
            "error": result.error,
            "reward": reward,
            "success": reward >= 1.0,
            "final_observation": final_observation_value,
            "total_steps": snapshot.total_steps_used,
            **_trace_metrics(events),
            "elapsed_s": (datetime.now(timezone.utc) - started).total_seconds(),
        }
        return episode, events
    finally:
        await environment.close()


async def run_eval(
    *,
    task_ids: Sequence[str],
    split: str,
    project_root: str,
    model: ToolCallingModel,
    limits: AgentLimits,
    episode_concurrency: int = 4,
    environment_max_steps: int = 30,
    trace_dir: Path | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, episode_concurrency))
    episodes: dict[int, dict[str, Any]] = {}
    lock = asyncio.Lock()
    started = datetime.now(timezone.utc)
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
    model_info = _model_info(model)

    async def one(index: int, task_id: str) -> None:
        async with semaphore:
            try:
                episode, trace = await run_episode(
                    task_id=task_id,
                    split=split,
                    project_root=project_root,
                    model=model,
                    limits=limits,
                    environment_max_steps=environment_max_steps,
                )
            except Exception as exc:
                episode = {
                    "task_id": task_id,
                    "status": "error",
                    "result": None,
                    "error": f"episode runtime error: {exc}",
                    "reward": 0.0,
                    "success": False,
                    "total_steps": 0,
                    "tool_calls_observed": 0,
                    "children_spawned": 0,
                    "elapsed_s": 0.0,
                }
                trace = []
            if trace_dir:
                (trace_dir / f"instance_{index:04d}.json").write_text(
                    json.dumps(trace, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            async with lock:
                episodes[index] = episode
                ordered = [episodes[item] for item in sorted(episodes)]
                if progress_path:
                    _write_report(
                        progress_path,
                        ordered,
                        started=started,
                        requested=len(task_ids),
                        split=split,
                        limits=limits,
                        model_info=model_info,
                        episode_concurrency=episode_concurrency,
                        environment_max_steps=environment_max_steps,
                        complete=False,
                    )
                print(
                    json.dumps(
                        {
                            "completed": len(episodes),
                            "total": len(task_ids),
                            "task_id": task_id,
                            "status": episode["status"],
                            "reward": episode["reward"],
                            "steps": episode["total_steps"],
                            "children": episode["children_spawned"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    await asyncio.gather(
        *(one(index, str(task_id)) for index, task_id in enumerate(task_ids))
    )
    ordered = [episodes[index] for index in range(len(task_ids))]
    return _build_report(
        ordered,
        started=started,
        requested=len(task_ids),
        split=split,
        limits=limits,
        model_info=model_info,
        episode_concurrency=episode_concurrency,
        environment_max_steps=environment_max_steps,
        complete=True,
    )


def summarize(episodes: Sequence[dict[str, Any]], *, requested: int) -> dict[str, Any]:
    rewards = [float(episode.get("reward", 0.0)) for episode in episodes]
    statuses = Counter(str(episode.get("status", "error")) for episode in episodes)
    return {
        "requested": requested,
        "completed": len(episodes),
        "average_reward": statistics.fmean(rewards) if rewards else 0.0,
        "median_reward": statistics.median(rewards) if rewards else 0.0,
        "success_count": sum(reward >= 1.0 for reward in rewards),
        "success_rate": (
            sum(reward >= 1.0 for reward in rewards) / len(rewards) if rewards else 0.0
        ),
        "zero_reward_count": sum(reward == 0.0 for reward in rewards),
        "reward_at_least_0_5_count": sum(reward >= 0.5 for reward in rewards),
        "status_counts": dict(sorted(statuses.items())),
        "total_steps": sum(int(item.get("total_steps", 0)) for item in episodes),
        "tool_calls_observed": sum(
            int(item.get("tool_calls_observed", 0)) for item in episodes
        ),
        "children_spawned": sum(
            int(item.get("children_spawned", 0)) for item in episodes
        ),
    }


def _build_report(
    episodes: Sequence[dict[str, Any]],
    *,
    started: datetime,
    requested: int,
    split: str,
    limits: AgentLimits,
    model_info: dict[str, Any],
    episode_concurrency: int,
    environment_max_steps: int,
    complete: bool,
) -> dict[str, Any]:
    return {
        "protocol": "simplified_recursive_agent_v1",
        "benchmark": "ReCode WebShop",
        "split": split,
        "model": model_info,
        "episode_concurrency": episode_concurrency,
        "environment_max_steps": environment_max_steps,
        "complete": complete,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "limits_per_episode": {
            "max_total_steps": limits.max_total_steps,
            "max_depth": limits.max_depth,
        },
        "summary": summarize(episodes, requested=requested),
        "episodes": list(episodes),
    }


def _write_report(
    path: Path,
    episodes: Sequence[dict[str, Any]],
    *,
    started: datetime,
    requested: int,
    split: str,
    limits: AgentLimits,
    model_info: dict[str, Any],
    episode_concurrency: int,
    environment_max_steps: int,
    complete: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report(
        episodes,
        started=started,
        requested=requested,
        split=split,
        limits=limits,
        model_info=model_info,
        episode_concurrency=episode_concurrency,
        environment_max_steps=environment_max_steps,
        complete=complete,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _model_info(model: ToolCallingModel) -> dict[str, Any]:
    fields = (
        "base_url",
        "model",
        "temperature",
        "max_tokens",
        "timeout",
        "native_tools",
        "tool_choice",
        "chat_template_kwargs",
    )
    return {
        "adapter": f"{model.__class__.__module__}.{model.__class__.__name__}",
        **{name: getattr(model, name) for name in fields if hasattr(model, name)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the simplified recursive harness on ReCode WebShop."
    )
    parser.add_argument("--project-root", default="/data2/zhangwenjian/agent/ReCode")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--num-instances", type=int, default=5)
    parser.add_argument("--ids", default="")
    parser.add_argument("--episode-concurrency", type=int, default=2)
    parser.add_argument("--environment-max-steps", type=int, default=30)
    parser.add_argument("--max-total-steps", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output", default="outputs/curagent_webshop_small.json")
    parser.add_argument("--trace-dir", default="outputs/curagent_webshop_small_traces")
    args = parser.parse_args()

    model = OpenAICompatibleModel(**load_model_config(args.config))
    limits = AgentLimits(
        max_total_steps=args.max_total_steps,
        max_depth=args.max_depth,
    )
    task_ids = (
        [item.strip() for item in args.ids.split(",") if item.strip()]
        if args.ids
        else [
            str(index)
            for index in range(args.start_id, args.start_id + args.num_instances)
        ]
    )
    output = Path(args.output)
    report = asyncio.run(
        run_eval(
            task_ids=task_ids,
            split=args.split,
            project_root=args.project_root,
            model=model,
            limits=limits,
            episode_concurrency=args.episode_concurrency,
            environment_max_steps=args.environment_max_steps,
            trace_dir=Path(args.trace_dir) if args.trace_dir else None,
            progress_path=output,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output), "summary": report["summary"]},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
