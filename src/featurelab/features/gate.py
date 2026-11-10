"""Static gate: inspect LLM-written code *before* it runs.

This is the first of three defensive layers for generated code:

1. **Static gate (here)** - parse the code into an AST and reject anything
   outside the contract: unknown imports, file/OS/network access, dunder tricks,
   wrong signature.  Cheap, and it gives the inventor a precise error to repair.
2. **Process isolation** (``runner.py``) - the code runs in a separate Python
   process with a stripped environment (no API keys), a CPU-time limit and a
   wall-clock timeout.  A crash or infinite loop kills that process, not the loop.
3. **Container** (Dockerfile) - the real security boundary.

Be honest about the limits of (1): Python is too dynamic for an AST check to be
a sandbox.  It catches mistakes and naive misuse; it does not stop a determined
attacker.  That is why layers (2) and (3) exist.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .contract import ALLOWED_IMPORTS

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit",
    "memoryview", "help", "dir",
}
FORBIDDEN_MODULE_HINTS = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "pickle",
                          "requests", "urllib", "ctypes", "importlib", "multiprocessing", "threading"}


@dataclass
class GateResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def message(self) -> str:
        return "; ".join(self.errors) if self.errors else "ok"


def check_code(code: str, max_chars: int = 6000) -> GateResult:
    errors: list[str] = []
    if len(code) > max_chars:
        errors.append(f"code is {len(code)} chars; limit is {max_chars}")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return GateResult(False, [f"SyntaxError line {exc.lineno}: {exc.msg}"])

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "feature"]
    if not functions:
        errors.append("no top-level function named `feature`")
    else:
        args = functions[0].args
        names = [a.arg for a in args.posonlyargs + args.args]
        if len(names) != 4 or args.vararg or args.kwarg:
            errors.append(f"`feature` must take exactly (x, fs, freqs, psd); got ({', '.join(names)})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    errors.append(f"import of `{alias.name}` is not allowed (only numpy, scipy, math)")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                errors.append(f"`from {node.module} import ...` is not allowed (only numpy, scipy, math)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"use of `{node.id}` is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_MODULE_HINTS:
            errors.append(f"reference to `{node.id}` is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder attribute access `.{node.attr}` is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal state is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in {"random", "default_rng", "seed"}:
            errors.append(f"randomness (`.{node.attr}`) is not allowed: features must be deterministic")

    for node in tree.body:  # top level: only imports, the function, constants, docstring
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Tuple, ast.List, ast.Dict)):
            continue
        errors.append(f"top-level `{type(node).__name__}` is not allowed; put logic inside `feature`")

    return GateResult(not errors, sorted(set(errors)))
