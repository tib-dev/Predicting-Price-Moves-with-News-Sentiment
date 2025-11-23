"""
Auto-generate placeholder docstrings for Python modules, classes, and functions.

Usage:
    python add_docstrings.py --folder src/fns_project --recursive
"""

import ast
import argparse
from pathlib import Path

MODULE_DOCSTRING = '"""Module description."""\n\n'
CLASS_DOCSTRING = '    """Class description."""\n'
FUNCTION_DOCSTRING = '    """Function description."""\n'


class DocstringAdder(ast.NodeVisitor):
    """AST visitor that adds placeholder docstrings to functions, classes, and modules."""

    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.edits = []

    def visit_Module(self, node):
        if not (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            self.edits.append((0, MODULE_DOCSTRING))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if not (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            # Insert after class line
            line_no = node.body[0].lineno - 1 if node.body else node.lineno
            indent = self._get_indent(node)
            self.edits.append((line_no, indent + CLASS_DOCSTRING))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            # Insert after function line
            line_no = node.body[0].lineno - 1 if node.body else node.lineno
            indent = self._get_indent(node)
            self.edits.append((line_no, indent + FUNCTION_DOCSTRING))
        self.generic_visit(node)

    def _get_indent(self, node):
        """Detect indentation level for a node."""
        line = self.source_lines[node.lineno - 1]
        return line[:len(line) - len(line.lstrip())]


def add_docstrings_to_file(file_path: Path):
    """Add docstrings to a single Python file."""
    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    visitor = DocstringAdder(source_lines)
    visitor.visit(tree)

    # Apply edits in reverse to keep line numbers valid
    for lineno, doc in sorted(visitor.edits, reverse=True):
        source_lines.insert(lineno, doc)

    new_source = "".join(source_lines)
    file_path.write_text(new_source, encoding="utf-8")
    print(f"[UPDATED] {file_path}")


def process_folder(folder: Path, recursive: bool = True):
    """Process all Python files in a folder, optionally recursively."""
    if recursive:
        files = folder.rglob("*.py")
    else:
        files = folder.glob("*.py")
    for path in files:
        add_docstrings_to_file(path)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-add placeholder docstrings.")
    parser.add_argument("--folder", required=True, help="Folder to process")
    parser.add_argument("--recursive", action="store_true",
                        help="Process recursively")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return

    process_folder(folder_path, recursive=args.recursive)


if __name__ == "__main__":
    main()
