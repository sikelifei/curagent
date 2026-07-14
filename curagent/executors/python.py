"""Isolated, side-effect-free-by-policy Python computation tool."""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
import sys
import tempfile
from typing import Any

from curagent.core.types import ToolCall


_DENIED_IMPORTS = {
    "ctypes",
    "http",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
_DENIED_CALLS = {"__import__", "compile", "eval", "exec", "input", "open"}


class PythonExecutor:
    def __init__(self, *, timeout_s: float = 10.0, memory_mb: int = 256) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    async def execute(self, call: ToolCall) -> Any:
        code = call.arguments["code"]
        policy_error = self._policy_error(code)
        if policy_error:
            return policy_error

        with tempfile.TemporaryDirectory(prefix="curagent-python-") as directory:
            prlimit = shutil.which("prlimit")
            if prlimit is None:
                return "python executor requires the prlimit executable"
            memory = self.memory_mb * 1024 * 1024
            cpu = max(1, int(self.timeout_s))
            command = [
                prlimit,
                f"--as={memory}",
                f"--cpu={cpu}",
                "--fsize=1048576",
                "--nofile=16",
                "--",
                sys.executable,
                "-I",
                "-c",
                code,
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=directory,
                    env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout_s
                )
            except OSError as exc:
                return f"could not start isolated Python: {exc}"
            except asyncio.TimeoutError:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                return f"python execution timed out after {self.timeout_s}s"
        output = stdout.decode("utf-8", errors="replace")
        error = stderr.decode("utf-8", errors="replace")
        return {
            "stdout": output,
            "stderr": error,
            "returncode": process.returncode,
        }

    @staticmethod
    def _policy_error(code: str) -> str | None:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return f"invalid Python syntax: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module.split(".", 1)[0])
                denied = sorted(set(names) & _DENIED_IMPORTS)
                if denied:
                    return f"imports are not allowed for pure computation: {denied}"
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _DENIED_CALLS
            ):
                return f"{node.func.id}() is not allowed for pure computation"
        return None
