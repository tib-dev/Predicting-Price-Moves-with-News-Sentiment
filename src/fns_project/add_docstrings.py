#!/usr/bin/env python3
"""
add_docstrings.py

Auto-generate docstring stubs for all Python files in a folder.
Supports module-level, class-level, and function-level docstrings.
Default style: Google. Can be customized to Numpy or others manually.
"""

import ast
import os
import argparse
from pathlib import Path


DOCSTRING_TEMPLATE = {
    "module": '"""Module description."""\n\n',
    "class": '"""Class description."""\n',
    "function": '"""Function description."""\n',
}


def add_docstrings_to_file(file_path: Path):
    """
    Parse a Python file, add missing docstrings for module, classes, and functions.
    Returns True if file modified, False otherwise.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    lines = source.splitlines()
    modified = False

    # ----- Module docstring -----
    if not (len(tree.body) > 0 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str)):
        # Insert module docstring at the top
        lines.insert(0, DOCSTRING_TEMPLATE["module"].rstrip())
        modified = True

    # ----- Class and function docstrings -----
    class FunctionDocVisitor(ast.NodeVisitor):
        def __init__(self, lines):
            self.lines = lines
            self.modified = False

        def visit_ClassDef(self, node: ast.ClassDef):
            if (len(node.body) == 0) or not (isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                insert_line = node.body[0].lineno - \
                    1 if node.body else node.lineno
                self.lines.insert(insert_line, DOCSTRING_TEMPLATE["class"])
                self.modified = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if (len(node.body) == 0) or not (isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                insert_line = node.body[0].lineno - \
                    1 if node.body else node.lineno
                self.lines.insert(insert_line, DOCSTRING_TEMPLATE["function"])
                self.modified = True
            self.generic_visit(node)

    visitor = FunctionDocVisitor(lines)
    visitor.visit(tree)
    if visitor.modified:
        modified = True

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[UPDATED] {file_path}")
    else:
        print(f"[SKIPPED] {file_path}")
    return modified


def process_folder(folder: Path, recursive=True):
    """Walk folder and add docstrings to all .py files."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist.")
    for path in folder.rglob("*.py") if recursive else folder.glob("*.py"):
        add_docstrings_to_file(path)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate docstring stubs for a Python project."
    )
    parser.add_argument("--folder", required=True,
                        help="Folder containing Python files (e.g., src/fns_project)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively process subfolders")
    args = parser.parse_args()

    process_folder(Path(args.folder), recursive=args.recursive)


if __name__ == "__main__":
    main()
