"""Sphinx configuration for the pybench documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable for autodoc even without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Make the synthetic example importable so its plots can use sample_loss_curves.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "src")
)

project = "pybench"
author = "Anthony"
copyright = "2026, Anthony"  # noqa: A001

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "matplotlib.sphinxext.plot_directive",
]

# Google-style docstrings (NumPy style off).
napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Loss plots rendered at build time; no committed binaries, always in sync.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 100)]

html_theme = "furo"
html_title = "pybench"

# Markdown narrative pages; generate anchors for headings (for cross-doc links).
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_heading_anchors = 3
