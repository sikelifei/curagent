"""Print every prompt layer used by the Oolong-Synthetic environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from recursive_agent.envs.oolong_synth import OolongSynthEnvironment
from recursive_agent.prompts import FORCED_FINAL_USER, build_initial_user, build_system_prompt
from recursive_agent.tools import format_tools_for_prompt, parse_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    sample = {
        "id": "PROMPT_PREVIEW",
        "context_window_id": "PROMPT_PREVIEW",
        "context_window_text": (
            "The following lines contain general-knowledge questions whose labels "
            "are entity or human being.\n"
            "Date: Jan 01, 2024 || User: 10001 || Instance: Who invented radio?"
        ),
        "question": (
            "How many data points should be classified as label 'human being'? "
            "Give your final answer in the form 'Answer: number'."
        ),
        "answer": "[1]",
        "answer_type": "ANSWER_TYPE.NUMERIC",
        "context_len": 1024,
        "dataset": "trec_coarse",
        "task_group": "counting",
        "task": "TASK_TYPE.NUMERIC_ONE_CLASS",
        "input_subset": "False",
    }
    environment = OolongSynthEnvironment(samples=[sample])
    tools = format_tools_for_prompt(parse_tools(environment.tools()))
    sections = [
        (
            "ROOT AND CHILD SYSTEM PROMPT",
            build_system_prompt(tools, prompt_addendum=environment.agent_prompt),
        ),
        ("ROOT INITIAL USER PROMPT", build_initial_user(environment.task)),
        (
            "DELEGATED CHILD INITIAL USER PROMPT",
            build_initial_user(
                environment.context["child_task_template"]
                + "\n\nQuestion: "
                + sample["question"],
                delegated=True,
            ),
        ),
        (
            "ROOT PRIVATE CONTEXT FIELDS",
            "\n".join(sorted(environment.context)),
        ),
        ("FORCED FINAL USER PROMPT", FORCED_FINAL_USER),
    ]
    rendered = "\n".join(
        f"\n{'=' * 24} {title} {'=' * 24}\n\n{content}"
        for title, content in sections
    ).strip() + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(path.resolve())
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
