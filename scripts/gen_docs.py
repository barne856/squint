import os
import re

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def create_file(path, content):
    create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)

def generate_api_docs(include_dir, docs_dir):
    api_dir = os.path.join(docs_dir, 'api')
    create_directory(api_dir)

    skip_files = ['tensor.hpp', 'geometry.hpp', 'quantity.hpp', 'squint.hpp']

    for root, dirs, files in os.walk(include_dir):
        relative_path = os.path.relpath(root, include_dir)
        if relative_path == '.':
            continue
        
        rst_path = os.path.join(api_dir, f"{relative_path}.rst")
        create_directory(os.path.dirname(rst_path))

        content = f"""
{os.path.basename(root)}
{'=' * len(os.path.basename(root))}

"""
        for file in files:
            if file.endswith('.hpp') and (file not in skip_files or relative_path != '.'):
                module_name = os.path.splitext(file)[0]
                content += f"""
{module_name}
{'-' * len(module_name)}

.. doxygenfile:: {os.path.join(relative_path, file)}
   :project: SQUINT

"""

        create_file(rst_path, content)

def generate_index_rst(readme_path, docs_dir):
    with open(readme_path, 'r') as f:
        readme_content = f.read()

    # Split the content into lines
    lines = readme_content.split('\n')

    index_content = """
Welcome to SQUINT's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

"""
    current_file = None
    current_content = ""
    current_level = 0
    in_code_block = False
    skip_next = False
    introduction_added = False

    for line in lines:
        if line.startswith('# SQUINT Tensor Library') or line.startswith('## Table of Contents'):
            skip_next = True
            continue
        
        if skip_next:
            skip_next = False
            continue

        if line.startswith('```'):
            in_code_block = not in_code_block
            if in_code_block:
                current_content += ".. code-block::\n\n"
            continue

        if line.startswith('#') and not in_code_block:
            level = len(line.split()[0])
            section = line.split(' ', 1)[1].strip()

            if level <= 2:  # Only # and ## get their own pages
                if current_file and current_file != "introduction":
                    create_file(os.path.join(docs_dir, f"{current_file}.rst"), current_content)
                slug = re.sub(r'[^\w\s-]', '', section.lower()).replace(' ', '_')
                if slug == "introduction" and introduction_added:
                    continue
                current_file = slug
                current_content = f"""
{section}
{'=' * len(section)}

"""
                index_content += f"   {slug}\n"
                current_level = level
                if slug == "introduction":
                    introduction_added = True
            else:
                current_content += f"""
{section}
{'-' * len(section)}

"""
        elif current_file:
            if in_code_block:
                current_content += "   " + line + "\n"
            else:
                current_content += line + "\n"

    if current_file:
        create_file(os.path.join(docs_dir, f"{current_file}.rst"), current_content)

    index_content += """
.. toctree::
   :maxdepth: 2
   :caption: API Reference

"""
    api_dir = os.path.join(docs_dir, 'api')
    for item in sorted(os.listdir(api_dir)):
        if item.endswith('.rst'):
            index_content += f"   api/{os.path.splitext(item)[0]}\n"

    create_file(os.path.join(docs_dir, 'index.rst'), index_content)

def create_conf_py(docs_dir):
    conf_py_content = '''
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
'''
    create_file(os.path.join(docs_dir, 'conf.py'), conf_py_content)

def create_doxyfile_in(docs_dir, include_dir):
    doxyfile_content = f'''
PROJECT_NAME           = "SQUINT"
PROJECT_NUMBER         = 1.0.0
PROJECT_BRIEF          = "SQUINT (Static Quantities in Tensors) is a modern, header-only C++ library designed to bring together compile-time dimensional analysis, unit conversion, and linear algebra operations in C++."
OUTPUT_DIRECTORY       = @CMAKE_CURRENT_BINARY_DIR@/doxygen
INPUT                  = {include_dir}
FILE_PATTERNS          = *.hpp
RECURSIVE              = YES
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = YES
EXTRACT_STATIC         = YES
GENERATE_HTML          = YES
GENERATE_XML           = YES
XML_OUTPUT             = xml
HAVE_DOT               = YES
UML_LOOK               = YES
TEMPLATE_RELATIONS     = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
STRIP_FROM_PATH        = {os.path.dirname(include_dir)}
STRIP_FROM_INC_PATH    = {os.path.dirname(include_dir)}
'''
    create_file(os.path.join(docs_dir, 'Doxyfile.in'), doxyfile_content)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    include_dir = os.path.join(project_root, 'include', 'squint')
    docs_dir = os.path.join(project_root, 'docs')
    readme_path = os.path.join(project_root, 'README.md')

    create_directory(include_dir)
    create_directory(docs_dir)

    if not os.path.exists(readme_path):
        create_file(readme_path, "# SQUINT\n\n## Introduction\n\nSQUINT is a C++ library for...\n\n## Installation\n\n## Usage\n\n## API Reference\n")

    generate_api_docs(include_dir, docs_dir)
    generate_index_rst(readme_path, docs_dir)
    create_conf_py(docs_dir)
    create_doxyfile_in(docs_dir, include_dir)

    print("Documentation structure generated successfully!")

if __name__ == "__main__":
    main()