"""Persistent in-process Python namespace for one agent."""

from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
import io
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .exceptions import ConfigurationError
from .tools import FRAMEWORK_NAMES, CapabilityCollection
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
    termination_kind: str | None = None
    termination_result: Any = None
    termination: "NodeTermination | None" = None

    @property
    def terminated(self) -> bool:
        """Whether the executed block requested node termination."""
        return self.termination is not None


class NodeTermination(BaseException):
    """Control signal raised by a framework capability to stop one node."""

    def __init__(self, kind: str = "terminated", result: Any = None) -> None:
        if not isinstance(kind, str) or not kind:
            raise ValueError("termination kind must be a non-empty string")
        self.kind = kind
        self.result = result
        super().__init__(kind)

    @property
    def termination_kind(self) -> str:
        return self.kind

    @property
    def termination_result(self) -> Any:
        return self.result


# Descriptive aliases keep the protocol easy to discover for callers that
# refer to the signal as either a REPL or generic framework termination.
ReplTermination = NodeTermination
TerminationSignal = NodeTermination


class ReplSession:
    def __init__(
        self,
        *,
        context: Any = None,
        tools: Mapping[str, Any] | CapabilityCollection | None = None,
        capabilities: Mapping[str, Any] | CapabilityCollection | None = None,
        spawn_subagent: Callable[[str, Any | None], Any] | None = None,
        spawn_subagents: Callable[[list[dict[str, Any]]], Any] | None = None,
        disabled_builtins: frozenset[str] | set[str] | None = None,
    ) -> None:
        if tools is not None and capabilities is not None:
            raise ValueError("Pass either tools or capabilities, not both")
        disabled = frozenset(disabled_builtins or ())
        unknown = disabled - ALLOWED_BUILTINS.keys()
        if unknown:
            raise ValueError(f"Unknown disabled REPL built-ins: {sorted(unknown)}")
        self._allowed_builtins = {
            name: value
            for name, value in ALLOWED_BUILTINS.items()
            if name not in disabled
        }
        self._context_fallback = context
        self._answer: dict[str, Any] = {"content": "", "ready": False}
        self._builtins: dict[str, Any] = {"SHOW_VARS": self._show_vars}
        if spawn_subagent is not None:
            self._builtins["spawn_subagent"] = spawn_subagent
        if spawn_subagents is not None:
            self._builtins["spawn_subagents"] = spawn_subagents
        self._capabilities = CapabilityCollection()
        self._tools: dict[str, Any] = {}
        self._bound_capability_names: set[str] = set()
        self.namespace: dict[str, Any] = {
            "__builtins__": dict(self._allowed_builtins),
            "__name__": "__main__",
            "context": context,
            "answer": self._answer,
            **self._builtins,
        }
        self.bind_capabilities(
            capabilities if capabilities is not None else tools,
            _allow_legacy_framework=capabilities is None,
        )

    @property
    def capabilities(self) -> CapabilityCollection:
        """Return the currently bound action-space collection."""
        return self._capabilities

    def bind_capabilities(
        self,
        capabilities: Mapping[str, Any] | CapabilityCollection | None,
        *,
        _allow_legacy_framework: bool = False,
    ) -> None:
        """Replace the current action space and remove stale bindings.

        User variables persist across calls, but names that were previously
        capabilities are tracked separately so a capability update cannot
        leave an action from an older node role in the namespace.
        """
        if isinstance(capabilities, CapabilityCollection):
            # Preserve the trusted framework marker and the exact collection
            # produced by the scheduler for prompt/runtime identity.
            collection = capabilities
        else:
            try:
                collection = CapabilityCollection(capabilities)
            except ConfigurationError:
                # Legacy RecursiveAgent integrations historically supplied a
                # terminal ``finish`` tool through ``tools=``. Keep that path
                # working without weakening the new capabilities= contract.
                if not isinstance(capabilities, Mapping):
                    raise
                if not _allow_legacy_framework and not set(capabilities) <= FRAMEWORK_NAMES:
                    raise
                framework = {
                    name: value
                    for name, value in capabilities.items()
                    if name in FRAMEWORK_NAMES
                }
                ordinary = {
                    name: value
                    for name, value in capabilities.items()
                    if name not in FRAMEWORK_NAMES
                }
                collection = CapabilityCollection(ordinary).merge_framework(framework)
        bindings = collection.bind()
        for name in self._bound_capability_names - set(bindings):
            self.namespace.pop(name, None)
        self._capabilities = collection
        self._tools = bindings
        self._bound_capability_names = set(bindings)
        self.namespace.update(bindings)

    # These names make the explicit update path convenient for callers while
    # retaining one implementation and one source of truth.
    update_capabilities = bind_capabilities
    set_capabilities = bind_capabilities

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

        def local_display(value: Any) -> None:
            if value is not None:
                local_print(repr(value))

        self.namespace["print"] = local_print
        self.namespace["__repl_display__"] = local_display
        error: str | None = None
        termination: NodeTermination | None = None
        try:
            tree = ast.parse(code, mode="exec")
            for index, statement in enumerate(tree.body):
                if not isinstance(statement, ast.Expr):
                    continue
                displayed = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="__repl_display__", ctx=ast.Load()),
                        args=[statement.value],
                        keywords=[],
                    )
                )
                tree.body[index] = ast.copy_location(displayed, statement)
            ast.fix_missing_locations(tree)
            compiled = compile(
                tree,
                "<repl>",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
            value = eval(compiled, self.namespace, self.namespace)
            if inspect.isawaitable(value):
                _run_awaitable(value)
        except NodeTermination as signal:
            termination = signal
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
            termination_kind=termination.kind if termination else None,
            termination_result=termination.result if termination else None,
            termination=termination,
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

        self.namespace["__builtins__"] = dict(self._allowed_builtins)
        self.namespace["__name__"] = "__main__"
        self.namespace.pop("print", None)
        self.namespace.pop("__repl_display__", None)
        self.namespace.update(self._builtins)
        self.namespace.update(self._tools)


async def _await_value(awaitable: Any) -> Any:
    return await awaitable


def _run_awaitable(awaitable: Any) -> Any:
    """Resolve one top-level await without requiring an async REPL caller."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(awaitable))

    # A synchronous caller may itself be running in an event-loop thread.
    # Running the small private loop in a worker avoids trying to nest loops.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, _await_value(awaitable)).result()


def find_repl_blocks(text: str) -> list[str]:
    """Extract fenced/XML REPL blocks in source order.

    ``repl`` is the documented label. ``python`` and ``py`` are accepted as
    compatibility aliases because OpenAI-compatible models commonly emit those
    labels for code that is still intended for the persistent REPL.
    """
    import re

    pattern = re.compile(
        r"```(?:repl|python|py)[ \t]*\r?\n(.*?)(?:\r?\n)?```"
        r"|<(?:repl|python|py)[ \t]*>(.*?)</(?:repl|python|py)[ \t]*>",
        re.DOTALL | re.IGNORECASE,
    )
    blocks = []
    for match in pattern.finditer(text):
        blocks.append((match.start(), (match.group(1) or match.group(2)).strip()))
    return [code for _, code in sorted(blocks)]
