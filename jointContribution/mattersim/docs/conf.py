import os
import sys

from recommonmark.parser import CommonMarkParser

sys.path.append(os.path.abspath("../src"))
project = "MatterSim"
copyright = "2024"
author = "Microsoft Corporation"
release = "1.0.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "recommonmark",
    "nbsphinx",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/microsoft/mattersim",
    "repository_provider": "github",
    "use_repository_button": True,
}
html_static_path = ["_static"]
html_logo = "_static/mattersim-with-name.png"
nbsphinx_allow_errors = True
nbsphinx_execute = "never"
