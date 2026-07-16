"""Oolong-Synthetic dataset loading and protocol sampling."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import EnvironmentDependencyError

DEFAULT_OOLONG_ROOT = Path("/data2/zhangwenjian/agent/bench/oolong")
DEFAULT_DATASET_NAME = "oolongbench/oolong-synth"
DEFAULT_SPLIT = "validation"


@dataclass(frozen=True)
class OolongSynthSample:
    index: int
    source_index: int
    sample_id: str
    context_window_id: str
    context_window_text: str
    question: str
    answer: Any
    answer_type: str
    context_len: int
    dataset: str
    task_group: str | None = None
    task: str | None = None
    input_subset: str | None = None

    @classmethod
    def from_mapping(cls, index: int, row: Mapping[str, Any]) -> "OolongSynthSample":
        context = _required_text(row, "context_window_text", index)
        question = _required_text(row, "question", index)
        if "answer" not in row:
            raise ValueError(f"Oolong-Synth sample {index} must contain answer")
        try:
            context_len = int(row["context_len"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Oolong-Synth sample {index} must contain integer context_len"
            ) from exc
        answer_type = _required_text(row, "answer_type", index)
        return cls(
            index=index,
            source_index=int(row.get("_source_index", index)),
            sample_id=str(row.get("id", index)),
            context_window_id=str(row.get("context_window_id", row.get("id", index))),
            context_window_text=context,
            question=question,
            answer=row["answer"],
            answer_type=answer_type,
            context_len=context_len,
            dataset=str(row.get("dataset", "unknown")),
            task_group=_optional_text(row.get("task_group")),
            task=_optional_text(row.get("task")),
            input_subset=_optional_text(row.get("input_subset")),
        )


class OolongSynthDataset:
    """Load final Oolong-Synthetic tasks, never the base validated examples."""

    def __init__(
        self,
        *,
        oolong_root: str | Path | None = None,
        split: str = DEFAULT_SPLIT,
        dataset_name: str = DEFAULT_DATASET_NAME,
        data_path: str | Path | None = None,
        samples: Sequence[Mapping[str, Any]] | None = None,
        loader: Callable[..., Any] | None = None,
        load_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.oolong_root = Path(
            oolong_root or os.getenv("OOLONG_ROOT") or DEFAULT_OOLONG_ROOT
        ).expanduser().resolve()
        self.split = str(split).strip()
        self.dataset_name = str(dataset_name).strip()
        if not self.split or not self.dataset_name:
            raise ValueError("Oolong-Synth split and dataset name must be non-empty")

        self._source = "in_memory"
        if samples is not None:
            self._rows: list[dict[str, Any]] | None = [dict(row) for row in samples]
            self._dataset = None
        else:
            local_path = _resolve_local_path(self.oolong_root, data_path, self.split)
            if local_path is not None:
                if local_path.is_dir() or local_path.suffix.lower() == ".parquet":
                    self._rows = None
                    self._dataset = _load_local_parquet(local_path, self.split)
                else:
                    self._rows = _read_local_rows(local_path, self.split)
                    self._dataset = None
                self._source = str(local_path)
            else:
                self._rows = None
                self._dataset = _load_huggingface_dataset(
                    loader=loader,
                    dataset_name=self.dataset_name,
                    split=self.split,
                    load_kwargs=load_kwargs,
                )
                self._source = f"{self.dataset_name}:{self.split}"
        if len(self) == 0:
            raise ValueError(f"Oolong-Synth split {self.split!r} is empty")

    def __len__(self) -> int:
        source = self._rows if self._rows is not None else self._dataset
        return len(source)

    def raw_row(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Oolong-Synth sample index must be an integer")
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(f"Oolong-Synth sample index {index} is out of range")
        row = self._rows[normalized] if self._rows is not None else self._dataset[normalized]
        if not isinstance(row, Mapping):
            raise ValueError(f"Oolong-Synth row {normalized} is not an object")
        return {"_source_index": normalized, **dict(row)}

    def __getitem__(self, index: int) -> OolongSynthSample:
        return OolongSynthSample.from_mapping(index, self.raw_row(index))

    def selection_metadata(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return [
                {
                    "index": index,
                    "id": row.get("id", index),
                    "context_len": int(row["context_len"]),
                    "dataset": str(row.get("dataset", "unknown")),
                }
                for index, row in enumerate(self._rows)
            ]
        return [
            {
                "index": index,
                "id": sample_id,
                "context_len": int(context_len),
                "dataset": str(dataset),
            }
            for index, (sample_id, context_len, dataset) in enumerate(
                zip(
                    self._dataset["id"],
                    self._dataset["context_len"],
                    self._dataset["dataset"],
                )
            )
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "source": self._source,
            "oolong_root": str(self.oolong_root),
            "size": len(self),
        }


def select_protocol_indices(
    metadata: Sequence[Mapping[str, Any]],
    *,
    sample_count: int = 199,
    seed: int = 42,
    dataset_filter: str | None = None,
    min_context_len: int | None = None,
    max_context_len: int | None = None,
) -> list[int]:
    """Select a deterministic sample stratified over every context-length bucket."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    eligible = []
    for row in metadata:
        context_len = int(row["context_len"])
        if dataset_filter and str(row.get("dataset")) != dataset_filter:
            continue
        if min_context_len is not None and context_len < min_context_len:
            continue
        if max_context_len is not None and context_len > max_context_len:
            continue
        eligible.append(row)
    if sample_count > len(eligible):
        raise ValueError(
            f"Cannot select {sample_count} rows from {len(eligible)} eligible rows"
        )

    groups: dict[int, list[int]] = defaultdict(list)
    for row in eligible:
        groups[int(row["context_len"])].append(int(row["index"]))
    bucket_lengths = sorted(groups)
    if not bucket_lengths:
        raise ValueError("No Oolong-Synth rows match the selection filters")

    base, remainder = divmod(sample_count, len(bucket_lengths))
    rng = random.Random(seed)
    selected: list[int] = []
    unselected: list[int] = []
    for position, context_len in enumerate(bucket_lengths):
        candidates = sorted(groups[context_len])
        rng.shuffle(candidates)
        quota = base + (1 if position < remainder else 0)
        take = min(quota, len(candidates))
        selected.extend(candidates[:take])
        unselected.extend(candidates[take:])

    if len(selected) < sample_count:
        rng.shuffle(unselected)
        selected.extend(unselected[: sample_count - len(selected)])
    context_by_index = {
        int(row["index"]): int(row["context_len"]) for row in eligible
    }
    return sorted(selected, key=lambda index: (context_by_index[index], index))


def _load_huggingface_dataset(
    *,
    loader: Callable[..., Any] | None,
    dataset_name: str,
    split: str,
    load_kwargs: Mapping[str, Any] | None,
) -> Any:
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise EnvironmentDependencyError(
                "Install Oolong support with `pip install -e '.[oolong]'`."
            ) from exc
        loader = load_dataset
    try:
        return loader(dataset_name, split=split, **dict(load_kwargs or {}))
    except Exception as exc:
        raise EnvironmentDependencyError(
            f"Failed to load Oolong-Synth {dataset_name!r} split {split!r}: {exc}"
        ) from exc


def _resolve_local_path(
    root: Path, data_path: str | Path | None, split: str
) -> Path | None:
    if data_path is not None:
        candidate = Path(data_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Oolong-Synth data path not found: {candidate}")
        return candidate.resolve()
    candidates = (
        root / "syth",
        root / "oolong-synth-hf",
        root / "data" / "synth" / f"{split}.jsonl",
        root / "data" / "synth" / f"{split}.json",
        root / "src" / "data_gen" / "oolong-synth" / f"{split}.jsonl",
    )
    for path in candidates:
        if path.is_dir() and (
            list((path / "data").glob(f"{split}-*.parquet"))
            or list(path.glob(f"{split}-*.parquet"))
        ):
            return path.resolve()
        if path.is_file():
            return path.resolve()
    return None


def _load_local_parquet(path: Path, split: str) -> Any:
    if path.is_dir():
        files = sorted((path / "data").glob(f"{split}-*.parquet"))
        if not files:
            files = sorted(path.glob(f"{split}-*.parquet"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(
            f"No {split}-*.parquet files found under Oolong-Synth path: {path}"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise EnvironmentDependencyError(
            "Install Oolong support with `pip install -e '.[oolong]'`."
        ) from exc
    try:
        return load_dataset(
            "parquet",
            data_files=[str(file) for file in files],
            split="train",
            cache_dir=os.getenv("HF_DATASETS_CACHE", "/tmp/curagent_hf_datasets"),
        )
    except Exception as exc:
        raise EnvironmentDependencyError(
            f"Failed to load local Oolong-Synth parquet files from {path}: {exc}"
        ) from exc


def _read_local_rows(path: Path, split: str) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                rows.append(dict(row))
        return rows
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if isinstance(document, Mapping):
            document = document.get(split)
        if not isinstance(document, list) or not all(
            isinstance(row, Mapping) for row in document
        ):
            raise ValueError(f"Oolong-Synth JSON must contain a row list: {path}")
        return [dict(row) for row in document]
    raise ValueError(f"Unsupported Oolong-Synth data format: {path.suffix}")


def _required_text(row: Mapping[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Oolong-Synth sample {index} must contain non-empty {field}")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_OOLONG_ROOT",
    "DEFAULT_SPLIT",
    "OolongSynthDataset",
    "OolongSynthSample",
    "select_protocol_indices",
]
