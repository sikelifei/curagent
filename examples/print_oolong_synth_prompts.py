"""Print every prompt layer used by the Oolong-Synthetic environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from recursive_agent.envs.oolong_synth import OolongSynthEnvironment
from recursive_agent.prompts import FORCED_FINAL_USER


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
    delegated_task = (
        "Process the assigned chunk and return concise mergeable text "
        "with chunk_id, rows processed, and the required statistics."
    )
    sections = [
        ("ROOT PROMPT", environment.root_prompt),
        ("CHILD PROMPT", environment.child_prompt),
        ("ROOT INITIAL USER PROMPT", f"Task:\n{environment.task}"),
        (
            "DELEGATED CHILD INITIAL USER PROMPT",
            f"Delegated task:\n{delegated_task}",
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
