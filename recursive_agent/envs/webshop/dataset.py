"""WebShop split and sample metadata backed by ReCode's dataset files."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .prompts import DEFAULT_WEBSHOP_TASK_TEMPLATE, build_webshop_task_prompt


@dataclass(frozen=True)
class WebShopSample:
    index: int
    session_id: int
    split: str
    instruction: str | None = None
    initial_observation: str | None = None

    def with_episode_data(self, *, instruction: str, observation: str) -> "WebShopSample":
        return replace(
            self,
            instruction=str(instruction).strip(),
            initial_observation=str(observation),
        )


class WebShopDataset:
    """Read train/test episode positions without loading the product index."""

    def __init__(self, recode_root: str | Path, split: str = "test") -> None:
        self.recode_root = Path(recode_root).expanduser().resolve()
        self.split = _validate_split(split)
        path = self.recode_root / "envs" / "webshop" / "data" / f"{self.split}_indices.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                values = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"WebShop split file not found: {path}") from exc
        if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
            raise ValueError(f"WebShop split file must contain a list of integer session IDs: {path}")
        self._session_ids = tuple(values)

    def __len__(self) -> int:
        return len(self._session_ids)

    def __getitem__(self, index: int) -> WebShopSample:
        if not isinstance(index, int):
            raise TypeError("WebShop sample index must be an integer")
        try:
            session_id = self._session_ids[index]
        except IndexError as exc:
            raise IndexError(
                f"WebShop sample index {index} is outside split {self.split!r} "
                f"with {len(self)} samples"
            ) from exc
        normalized_index = index if index >= 0 else len(self) + index
        return WebShopSample(
            index=normalized_index,
            session_id=session_id,
            split=self.split,
        )

    def build_task(
        self,
        sample: WebShopSample,
        *,
        template: str = DEFAULT_WEBSHOP_TASK_TEMPLATE,
    ) -> str:
        if sample.instruction is None:
            raise ValueError("Reset the WebShop episode before building its task prompt")
        return build_webshop_task_prompt(sample.instruction, template=template)


def _validate_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in {"train", "test"}:
        raise ValueError("WebShop split must be 'train' or 'test'")
    return normalized

