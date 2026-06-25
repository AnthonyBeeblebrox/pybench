"""Developer task CLI for pybench.

Run with ``uv run dev.py <command>`` (it is not part of the shipped package).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

ROOT = Path(__file__).parent


@click.group()
def cli() -> None:
    """Developer tasks for working on pybench."""


@cli.command()
def docs() -> None:
    """Build the HTML documentation (warnings are errors)."""
    out = ROOT / "docs" / "_build" / "html"
    result = subprocess.run(
        [
            "uv",
            "run",
            "--group",
            "docs",
            "sphinx-build",
            "-W",
            "-b",
            "html",
            "docs",
            str(out),
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode == 0:
        click.echo(f"Documentation built at {out / 'index.html'}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    cli()
