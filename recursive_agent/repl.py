"""Persistent in-process Python namespace for one agent."""

from __future__ import annotations

import builtins
import io
import time
from dataclasses import dataclass
from typing import Any, Callable

from .types import CodeExecutionTrace

_ALLOWED_BUILTIN_NAMES = {
    "__build_class__",
    "__import__",
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "complex",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "hasattr",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "memoryview",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    "BaseException",
    "Exception",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "EOFError",
    "FileNotFoundError",
    "ImportError",
    "IndexError",
    "KeyError",
    "LookupError",
    "NameError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "SyntaxError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
}

ALLOWED_BUILTINS = {
    name: getattr(builtins, name)
    for name in _ALLOWED_BUILTIN_NAMES
    if hasattr(builtins, name)
}


@dataclass
class ExecutionResult:
    trace: CodeExecutionTrace
    answer_ready: bool
    answer_content: str | None


class ReplSession:
    def __init__(
        self,
        *,
        context: Any,
        tools: dict[str, Any],
        spawn_subagent: Callable[[str, Any | None], str],
        spawn_subagents: Callable[[list[dict[str, Any]]], list[str]],
    ) -> None:
        self._tools = tools
        self._context_fallback = context
        self._answer: dict[str, Any] = {"content": "", "ready": False}
        self._builtins = {
            "SHOW_VARS": self._show_vars,
            "spawn_subagent": spawn_subagent,
            "spawn_subagents": spawn_subagents,
        }
        self.namespace: dict[str, Any] = {
            "__builtins__": dict(ALLOWED_BUILTINS),
            "__name__": "__main__",
            "context": context,
            "answer": self._answer,
            **self._builtins,
            **tools,
        }

    def _show_vars(self) -> str:
        hidden = {
            "__builtins__",
            "__name__",
            "print",
            "answer",
            *self._builtins,
            *self._tools,
        }
        variables = {
            name: type(value).__name__
            for name, value in self.namespace.items()
            if name not in hidden and not name.startswith("_")
        }
        if not variables:
            return "No variables created yet."
        return f"Available variables: {variables}"

    def execute(self, code: str) -> ExecutionResult:
        started = time.perf_counter()
        stdout = io.StringIO()

        def local_print(*args: Any, **kwargs: Any) -> None:
            kwargs = dict(kwargs)
            kwargs["file"] = stdout
            builtins.print(*args, **kwargs)

        self.namespace["print"] = local_print
        error: str | None = None
        try:
            exec(code, self.namespace, self.namespace)
        except Exception as exc:
            error = f"Error: {type(exc).__name__}: {exc}"

        answer = self.namespace.get("answer")
        answer_ready = isinstance(answer, dict) and answer.get("ready") is True
        answer_content = str(answer.get("content", "")) if answer_ready else None
        output = stdout.getvalue().rstrip("\n")
        if error:
            output = f"{output}\n{error}".strip()
        variables = self._variable_names()
        self._restore_scaffold(answer)
        return ExecutionResult(
            trace=CodeExecutionTrace(
                code=code,
                output=output,
                error=error,
                duration_seconds=time.perf_counter() - started,
                variables=variables,
            ),
            answer_ready=answer_ready,
            answer_content=answer_content,
        )

    def _variable_names(self) -> list[str]:
        reserved = {
            "__builtins__",
            "__name__",
            "print",
            "answer",
            *self._builtins,
            *self._tools,
        }
        return sorted(
            name
            for name in self.namespace
            if name not in reserved and not name.startswith("_")
        )

    def _restore_scaffold(self, answer: Any) -> None:
        if "context" in self.namespace:
            self._context_fallback = self.namespace["context"]
        else:
            self.namespace["context"] = self._context_fallback

        if isinstance(answer, dict):
            answer.setdefault("content", "")
            answer.setdefault("ready", False)
            self._answer = answer
        else:
            self._answer = {"content": "", "ready": False}
            self.namespace["answer"] = self._answer

        self.namespace["__builtins__"] = dict(ALLOWED_BUILTINS)
        self.namespace["__name__"] = "__main__"
        self.namespace.pop("print", None)
        self.namespace.update(self._builtins)
        self.namespace.update(self._tools)


def find_repl_blocks(text: str) -> list[str]:
    """Extract fenced or XML-style repl blocks in their source order."""
    import re

    pattern = re.compile(
        r"```repl[ \t]*\r?\n(.*?)(?:\r?\n)?```"
        r"|<repl[ \t]*>(.*?)</repl[ \t]*>",
        re.DOTALL | re.IGNORECASE,
    )
    blocks = []
    for match in pattern.finditer(text):
        blocks.append((match.start(), (match.group(1) or match.group(2)).strip()))
    return [code for _, code in sorted(blocks)]
