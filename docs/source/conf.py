import sys
from pathlib import Path

# This project is not packaged or installed (see the note in pyproject.toml): it is run from
# the repo root, so `src` has to be importable as a top-level package for autodoc to find
# `src.simulation.*`. parents[2] is the repo root: source/ -> docs/ -> root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "Experience and Reasoning"
copyright = "2026, Adam Costarino"
author = "Adam Costarino"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    # the docstrings carry real mathematics, so render it
    "sphinx.ext.mathjax",
]

autosummary_generate = True

# The docstrings carry the maths, so keep them in source order rather than alphabetising:
# a class reads the way it was written.
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Nothing here is installed, so the scientific stack is mocked rather than pulled into the
# docs build: autodoc only needs to import the modules to read their docstrings, and these
# are the third-party packages `src/` imports.
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "quantecon",
    "tqdm",
    "cycler",
]

# Both live in docs/, one level up from this file; Sphinx resolves them relative to the
# config directory, so they need the "../" rather than an absolute path.
templates_path = ["../_templates"]

html_theme = "furo"
html_static_path = ["../_static"]
