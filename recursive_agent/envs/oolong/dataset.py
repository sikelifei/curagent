"""Oolong-real dataset loading and sample normalization."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import EnvironmentDependencyError

DEFAULT_OOLONG_ROOT = Path("/data2/zhangwenjian/agent/bench/oolong")
DEFAULT_DATASET_NAME = "oolongbench/oolong-real"
DEFAULT_CONFIG_NAME = "dnd"


@dataclass(frozen=True)
class OolongSample:
    """One normalized Oolong example."""

    index: int
    sample_id: str
    context_window_id: str
    context_window_text: str
    question: str
    answer: Any
    dataset: str = "real"
    answer_type: str | None = None
    question_type: str | None = None
    episodes: tuple[Any, ...] = ()
    campaign: str | None = None
    context_len: int | None = None

    @classmethod
    def from_mapping(cls, index: int, row: Mapping[str, Any]) -> "OolongSample":
        context = _required_text(row, "context_window_text", index)
        question = _required_text(row, "question", index)
        if "answer" not in row:
            raise ValueError(f"Oolong sample {index} must contain answer")
        sample_id = str(row.get("id", index))
        context_window_id = str(row.get("context_window_id", sample_id))
        context_len = row.get("context_len")
        if context_len is not None:
            try:
                context_len = int(context_len)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Oolong sample {index} has a non-integer context_len"
                ) from exc
        return cls(
            index=index,
            sample_id=sample_id,
            context_window_id=context_window_id,
            context_window_text=context,
            question=question,
            answer=row["answer"],
            dataset=str(row.get("dataset", "real")),
            answer_type=_optional_text(row.get("answer_type")),
            question_type=_optional_text(row.get("question_type")),
            episodes=_normalize_sequence(row.get("episodes")),
            campaign=_optional_text(row.get("campaign")),
            context_len=context_len,
        )


class OolongDataset:
    """Load Oolong-real from a local generated split or Hugging Face datasets.

    The checked-out Oolong repository contains the data-generation and scoring
    code, but its generated transcripts/QA pairs are not checked in. We prefer
    a generated local JSONL split when present and fall back to the published
    ``oolongbench/oolong-real`` dataset.
    """

    def __init__(
        self,
        *,
        oolong_root: str | Path | None = None,
        split: str = "test",
        dataset_name: str = DEFAULT_DATASET_NAME,
        config_name: str = DEFAULT_CONFIG_NAME,
        data_path: str | Path | None = None,
        samples: Sequence[Mapping[str, Any]] | None = None,
        loader: Callable[..., Any] | None = None,
        load_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.oolong_root = _resolve_root(oolong_root)
        self.split = _validate_split(split)
        self.dataset_name = str(dataset_name)
        self.config_name = str(config_name)
        self._source = "in_memory"

        if samples is not None:
            self._rows = [dict(row) for row in samples]
            self._dataset = None
        else:
            local_path = _resolve_local_path(self.oolong_root, data_path, self.split)
            if local_path is not None:
                self._rows = _read_local_rows(local_path, self.split)
                self._dataset = None
                self._source = str(local_path)
            else:
                self._rows = None
                self._dataset = _load_huggingface_dataset(
                    loader=loader,
                    dataset_name=self.dataset_name,
                    config_name=self.config_name,
                    split=self.split,
                    load_kwargs=load_kwargs,
                )
                self._source = f"{self.dataset_name}/{self.config_name}:{self.split}"

        if len(self) == 0:
            raise ValueError(f"Oolong split {self.split!r} is empty")

    def __len__(self) -> int:
        source = self._rows if self._rows is not None else self._dataset
        return len(source)

    def __getitem__(self, index: int) -> OolongSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Oolong sample index must be an integer")
        normalized_index = index if index >= 0 else len(self) + index
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError(
                f"Oolong sample index {index} is outside split {self.split!r} "
                f"with {len(self)} samples"
            )
        row = self._rows[normalized_index] if self._rows is not None else self._dataset[normalized_index]
        if not isinstance(row, Mapping):
            raise ValueError(f"Oolong dataset row {normalized_index} is not a mapping")
        return OolongSample.from_mapping(normalized_index, row)

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "config_name": self.config_name,
            "split": self.split,
            "source": self._source,
            "oolong_root": str(self.oolong_root),
            "size": len(self),
        }


def _load_huggingface_dataset(
    *,
    loader: Callable[..., Any] | None,
    dataset_name: str,
    config_name: str,
    split: str,
    load_kwargs: Mapping[str, Any] | None,
) -> Any:
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise EnvironmentDependencyError(
                "Oolong-real data was not found locally and the optional datasets "
                "dependency is missing. Install with `pip install -e '.[oolong]'`."
            ) from exc
        loader = load_dataset
    try:
        return loader(
            dataset_name,
            config_name,
            split=split,
            **dict(load_kwargs or {}),
        )
    except Exception as exc:
        raise EnvironmentDependencyError(
            f"Failed to load Oolong dataset {dataset_name!r} config "
            f"{config_name!r} split {split!r}: {exc}"
        ) from exc


def _resolve_root(value: str | Path | None) -> Path:
    candidate = value or os.getenv("OOLONG_ROOT") or DEFAULT_OOLONG_ROOT
    return Path(candidate).expanduser().resolve()


def _resolve_local_path(root: Path, data_path: str | Path | None, split: str) -> Path | None:
    if data_path is not None:
        candidate = Path(data_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Oolong data file not found: {candidate}")
        return candidate.resolve()

    candidates = (
        root / "src" / "data_gen" / "oolong-real" / "qa_pairs" / "dnd" / f"{split}.jsonl",
        root / "data_gen" / "oolong-real" / "qa_pairs" / "dnd" / f"{split}.jsonl",
        root / "data" / "dnd" / f"{split}.jsonl",
        root / f"{split}.jsonl",
        root / f"{split}.json",
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


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
                    raise ValueError(f"Invalid Oolong JSONL at {path}:{line_number}") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"Oolong row at {path}:{line_number} is not an object")
                rows.append(dict(row))
        return rows

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if isinstance(document, Mapping) and isinstance(document.get(split), list):
            document = document[split]
        if not isinstance(document, list) or not all(isinstance(row, Mapping) for row in document):
            raise ValueError(f"Oolong JSON file must contain a list of objects: {path}")
        return [dict(row) for row in document]

    raise ValueError(f"Unsupported Oolong local data format: {path.suffix}")


def _required_text(row: Mapping[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Oolong sample {index} must contain non-empty {field}")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return tuple(value)
    return (value,)


def _validate_split(split: str) -> str:
    normalized = str(split).strip()
    if not normalized:
        raise ValueError("Oolong split must be non-empty")
    return normalized
