
# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SQUINT'
copyright = '2023, Brendan Barnes'
author = 'Brendan Barnes'

release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'breathe',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

breathe_projects = {"SQUINT": "_build/xml"}
breathe_default_project = "SQUINT"

intersphinx_mapping = {'https://docs.python.org/': None}

todo_include_todos = True
