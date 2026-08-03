"""
ingestion/walker.py — Filesystem walker with gitignore + binary filtering.

Responsibilities:
  - Walk a repo directory recursively
  - Honour .gitignore patterns at every directory level (pathspec)
  - Skip known non-code directories (node_modules, vendor, .venv, …)
  - Skip binary files and lock/build artefacts by extension
  - Yield SourceFile objects — lightweight descriptors for each code file

Design choice: pathspec (used by pip and poetry internally) correctly
implements the full gitignore glob spec including negations, which plain
fnmatch does not.  It's small and has no heavy transitive deps.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pathspec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directories that are *always* skipped, regardless of .gitignore content.
# These tend to contain generated / vendored / binary content that would
# pollute the index and slow down parsing.
_ALWAYS_SKIP_DIRS: frozenset[str] = frozenset(
    {
        # JavaScript / Node
        "node_modules",
        ".yarn",
        ".pnp",
        # Python
        ".venv",
        "venv",
        "__pycache__",
        ".eggs",
        "*.egg-info",
        # Build outputs
        "dist",
        "build",
        "target",  # Rust / Maven
        "out",
        "_build",
        "cmake-build-debug",
        "cmake-build-release",
        # VCS / IDE
        ".git",
        ".svn",
        ".hg",
        ".idea",
        ".vscode",
        # Misc vendored / generated
        "vendor",
        "third_party",
        "third-party",
        "generated",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        # CodeRAG own data
        ".coderag",
        "_repos",
    }
)

# File extensions we know are parseable code (must be in this set OR the
# walker will skip them unless _BINARY_EXTENSIONS doesn't catch them first).
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".kt",  # Kotlin
        ".scala",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".cs",  # C#
        ".rb",
        ".php",
        ".swift",
        ".dart",
        ".lua",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".r",
        ".R",
        # Config / markup (useful for context)
        ".md",
        ".rst",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".scss",
        ".sql",
        ".tf",  # Terraform
        ".proto",  # Protobuf
    }
)

# Extensions that are *definitely* binary — always skipped regardless.
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Images
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
        ".tiff", ".raw",
        # Audio / video
        ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".avi", ".mkv", ".mov",
        # Archives / packages
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".whl",
        ".jar", ".war", ".ear", ".apk", ".ipa",
        # Compiled / binary
        ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".o", ".a",
        ".lib", ".pdb", ".class",
        # Fonts
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
        # Documents
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        # Database
        ".db", ".sqlite", ".sqlite3",
        # Lock files (by extension — filenames are handled separately)
        ".lock",
    }
)

# Exact filenames (case-sensitive) that are always skipped even if they have
# a code extension (e.g. package-lock.json, yarn.lock with .lock above).
_SKIP_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "Pipfile.lock",
        "poetry.lock",
        "Cargo.lock",
        "composer.lock",
        "Gemfile.lock",
        "go.sum",
        ".DS_Store",
        "Thumbs.db",
    }
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFile:
    """Lightweight descriptor for a single source file found by the walker."""

    path: Path
    """Absolute path to the file."""

    relative_path: Path
    """Path relative to the repository root."""

    extension: str
    """Lowercase file extension including the dot (e.g. ``'.py'``)."""

    size_bytes: int
    """File size in bytes at walk time."""

    language: str | None = field(default=None)
    """Detected programming language, or ``None`` if not recognised."""


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".dart": "dart",
    ".lua": "lua",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".proto": "protobuf",
}


def detect_language(extension: str) -> str | None:
    """Return a canonical language name for *extension*, or ``None``."""
    return _EXT_TO_LANGUAGE.get(extension.lower())


# ---------------------------------------------------------------------------
# GitignoreSpec helper
# ---------------------------------------------------------------------------


def _load_gitignore(directory: Path) -> pathspec.PathSpec | None:
    """Load a ``.gitignore`` file from *directory* if one exists."""
    gitignore = directory / ".gitignore"
    if gitignore.is_file():
        try:
            patterns = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
            return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except OSError as exc:
            logger.warning("Could not read %s: %s", gitignore, exc)
    return None


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


class FileWalker:
    """Recursively walks a repository and yields :class:`SourceFile` objects.

    Usage::

        walker = FileWalker(repo_root=Path("/path/to/repo"))
        for source_file in walker.walk():
            print(source_file.relative_path, source_file.language)
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        languages: list[str] | None = None,
        max_file_size_bytes: int = 1_000_000,  # 1 MB — skip huge generated files
    ) -> None:
        """
        Args:
            repo_root:           Root directory of the repository.
            languages:           If provided, only files whose detected language
                                 is in this list are yielded.  Pass ``None``
                                 to yield all recognised code files.
            max_file_size_bytes: Files larger than this are skipped (they are
                                 likely auto-generated or minified).
        """
        self.repo_root = repo_root.resolve()
        self.languages = {lang.lower() for lang in languages} if languages else None
        self.max_file_size_bytes = max_file_size_bytes

        # Cache of gitignore specs per directory (populated lazily during walk)
        self._gitignore_cache: dict[Path, pathspec.PathSpec | None] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def walk(self) -> Iterator[SourceFile]:
        """Yield every qualifying :class:`SourceFile` under ``repo_root``."""
        stats = {"visited": 0, "skipped_dir": 0, "skipped_file": 0, "yielded": 0}

        for dirpath_str, dirnames, filenames in os.walk(self.repo_root, topdown=True):
            dirpath = Path(dirpath_str)

            # --- prune directories in-place (os.walk respects this) ---
            dirnames[:] = [
                d for d in dirnames
                if not self._should_skip_dir(dirpath, d)
            ]
            stats["skipped_dir"] += (
                len([d for d in os.listdir(dirpath) if (dirpath / d).is_dir()])
                - len(dirnames)
            )

            for filename in filenames:
                stats["visited"] += 1
                filepath = dirpath / filename
                source_file = self._evaluate_file(filepath)
                if source_file is None:
                    stats["skipped_file"] += 1
                    continue
                stats["yielded"] += 1
                yield source_file

        logger.debug(
            "Walk complete: visited=%d yielded=%d skipped_files=%d",
            stats["visited"],
            stats["yielded"],
            stats["skipped_file"],
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _should_skip_dir(self, parent: Path, dirname: str) -> bool:
        """Return True if *dirname* inside *parent* should be skipped."""
        # Hard-coded always-skip list (exact name match)
        if dirname in _ALWAYS_SKIP_DIRS:
            return True

        # Also skip any directory whose name ends with ".egg-info"
        if dirname.endswith(".egg-info"):
            return True

        # Check gitignore for the directory entry
        spec = self._get_gitignore(parent)
        if spec is not None:
            rel = (parent / dirname).relative_to(self.repo_root)
            # pathspec expects forward slashes and a trailing "/" for directories
            if spec.match_file(str(rel).replace(os.sep, "/") + "/"):
                return True

        return False

    def _evaluate_file(self, filepath: Path) -> SourceFile | None:
        """Return a :class:`SourceFile` for *filepath*, or ``None`` to skip."""
        filename = filepath.name
        extension = filepath.suffix.lower()

        # Exact filename blocklist
        if filename in _SKIP_FILENAMES:
            logger.debug("Skipping blocklisted filename: %s", filepath)
            return None

        # Binary extension
        if extension in _BINARY_EXTENSIONS:
            return None

        # Must be a recognised code extension (or no extension filter applied)
        if extension not in _CODE_EXTENSIONS:
            logger.debug("Skipping unknown extension %s: %s", extension, filepath)
            return None

        # Size gate
        try:
            size = filepath.stat().st_size
        except OSError:
            logger.debug("Could not stat %s — skipping", filepath)
            return None
        if size > self.max_file_size_bytes:
            logger.debug("Skipping oversized file (%d bytes): %s", size, filepath)
            return None

        # gitignore check for the file itself
        spec = self._get_gitignore(filepath.parent)
        if spec is not None:
            rel = filepath.relative_to(self.repo_root)
            if spec.match_file(str(rel).replace(os.sep, "/")):
                logger.debug("Skipping gitignore-matched file: %s", filepath)
                return None

        # Language filter
        language = detect_language(extension)
        if self.languages is not None and (language is None or language not in self.languages):
            return None

        return SourceFile(
            path=filepath,
            relative_path=filepath.relative_to(self.repo_root),
            extension=extension,
            size_bytes=size,
            language=language,
        )

    def _get_gitignore(self, directory: Path) -> pathspec.PathSpec | None:
        """Return (cached) gitignore spec for *directory*, or ``None``."""
        if directory not in self._gitignore_cache:
            self._gitignore_cache[directory] = _load_gitignore(directory)
        return self._gitignore_cache[directory]
