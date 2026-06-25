"""Discover ``bench_*`` functions by importing Python files under a path."""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from pybench.config import BenchConfig, extract_config


class DiscoveryError(ValueError):
    """Raised when discovery cannot satisfy the request."""


@dataclass(frozen=True)
class Benchmark:
    """A discovered benchmark function and its resolved configuration."""

    name: str
    fn: Callable[[int], object]
    config: BenchConfig
    file: Path


def import_file(file: Path) -> ModuleType:
    """Import a Python file as an anonymous module.

    Args:
        file: Path to the ``.py`` file.

    Returns:
        The imported module object.

    Raises:
        DiscoveryError: If the file cannot be loaded as a module.
    """
    mod_name = "_pybench_bench_" + file.stem
    spec = importlib.util.spec_from_file_location(mod_name, file)
    if spec is None or spec.loader is None:
        msg = f"cannot import benchmark file: {file}"
        raise DiscoveryError(msg)
    module = importlib.util.module_from_spec(spec)
    # Compile + exec the source directly rather than via the loader: this skips
    # the .pyc cache (whose second-granularity mtime check can serve stale
    # bytecode when a benchmark is edited and re-run quickly) and avoids
    # littering __pycache__ in the user's benchmark directory.
    code = compile(file.read_text(), str(file), "exec")
    exec(code, module.__dict__)
    return module


def discover(path: Path, names: Sequence[str] | None = None) -> list[Benchmark]:
    """Find ``bench_*`` functions defined under ``path``.

    A file is any ``.py`` file (recursively, when ``path`` is a directory).
    Only functions *defined* in the imported file are collected, so a
    ``bench_*`` imported from elsewhere is ignored.

    Args:
        path: A benchmark file or a directory to walk.
        names: If given, keep only these benchmark names (``--bench``).

    Returns:
        Benchmarks sorted by name.

    Raises:
        DiscoveryError: If ``path`` does not exist, a benchmark name is defined
            twice, or a requested ``names`` entry is not found.
    """
    if not path.exists():
        msg = f"path does not exist: {path}"
        raise DiscoveryError(msg)
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    found: dict[str, Benchmark] = {}
    for file in files:
        module = import_file(file)
        for attr, obj in vars(module).items():
            if not attr.startswith("bench_") or not inspect.isfunction(obj):
                continue
            if obj.__module__ != module.__name__:  # imported, not defined here
                continue
            if attr in found:
                msg = f"benchmark {attr!r} is defined more than once"
                raise DiscoveryError(msg)
            found[attr] = Benchmark(attr, obj, extract_config(obj), file)
    if names is not None:
        missing = [n for n in names if n not in found]
        if missing:
            msg = f"benchmark(s) not found: {', '.join(sorted(missing))}"
            raise DiscoveryError(msg)
        found = {n: found[n] for n in names}
    return sorted(found.values(), key=lambda b: b.name)
