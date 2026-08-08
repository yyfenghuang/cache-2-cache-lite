# SPDX-License-Identifier: Apache-2.0

"""Every global a function reaches for exists in its module.

Silent on success, exit 0. No third-party linter, because a gate that fails
because a tool is missing fails for a reason unrelated to what it gates.

This catches one class of bug and catches it exactly: a name referenced from
inside a function body that resolves to nothing at module level. That class
is invisible to py_compile, which only checks that the source parses, and it
stays invisible until the line runs. For a script that cannot be executed
without model weights on disk, "until the line runs" can mean much later than
the moment the mistake was made.

The method is to read LOAD_GLOBAL opcodes rather than to scan the source.
Attribute access compiles to LOAD_ATTR and never appears, so `x.read_text`
raises nothing here. Imports made inside a function compile to local stores,
so `import torch` in a function body is a local and never appears either.
Both are false positives a text scan would produce and this does not.

A module that will not import at all is reported rather than raised. Dying on
the first broken file hides every other broken file behind it, and the next
run then finds the second one, which is a round trip per fault. One run
should show the whole picture.
"""

from __future__ import annotations

import builtins
import dis
import importlib.util
import sys
from pathlib import Path
from types import CodeType, ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SEARCH_DIRS = ("c2c", "scripts", "tests")

SELF = Path(__file__).resolve()


def python_files() -> list[Path]:
    out = []
    for d in SEARCH_DIRS:
        out.extend(sorted((REPO_ROOT / d).glob("*.py")))
    return [p for p in out if p.resolve() != SELF]


def load(path: Path) -> ModuleType:
    name = f"_names_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def code_objects(code: CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, CodeType):
            yield from code_objects(const)


def globals_referenced(code: CodeType) -> set[str]:
    names = set()
    for block in code_objects(code):
        for instruction in dis.get_instructions(block):
            if instruction.opname == "LOAD_GLOBAL":
                names.add(instruction.argval)
    return names


def test_every_module_imports_and_every_global_resolves():
    known_builtins = set(dir(builtins))
    problems = []

    for path in python_files():
        where = path.relative_to(REPO_ROOT)
        try:
            module = load(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{where}: will not import, {type(exc).__name__}: {exc}")
            continue
        defined = set(vars(module)) | known_builtins
        source = path.read_text(encoding="utf-8")
        missing = globals_referenced(compile(source, str(path), "exec")) - defined
        for name in sorted(missing):
            problems.append(f"{where}: undefined global {name}")

    if problems:
        raise AssertionError(
            f"{len(problems)} problem(s):\n  " + "\n  ".join(problems)
        )


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    main()