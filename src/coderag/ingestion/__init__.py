"""
Ingestion sub-package — repo resolution, file walking, AST parsing.

Public API:
  resolve_repo(source)    → Path
  FileWalker(repo_root)   → yields SourceFile
  ASTParser()             → parses SourceFile → ParsedFile
"""

from coderag.ingestion.parser import ASTNode, ASTParser, ParsedFile
from coderag.ingestion.repo import resolve_repo
from coderag.ingestion.walker import FileWalker, SourceFile, detect_language

__all__ = [
    "resolve_repo",
    "FileWalker",
    "SourceFile",
    "detect_language",
    "ASTParser",
    "ParsedFile",
    "ASTNode",
]
