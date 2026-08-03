"""
ingestion/repo.py — Repository source resolution.

Responsibilities:
  - Accept either a local directory path or a GitHub URL
  - Clone remote repos into a local working directory (under _repos/)
  - Return a resolved local Path that subsequent phases can walk

Design choice: GitPython for cloning because it is already a dependency
(we'll also use it for git-diff-based incremental indexing in Phase 4) and
gives us fine-grained control over clone options without shelling out.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex that matches "https://github.com/owner/repo" and variants
_GITHUB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)

# Default directory (inside the project root) where clones are stored
_DEFAULT_CLONE_ROOT = Path("_repos")


def resolve_repo(
    source: str,
    clone_root: Path = _DEFAULT_CLONE_ROOT,
) -> Path:
    """Resolve *source* to a local directory ready for ingestion.

    Args:
        source:     A local path (absolute or relative) **or** a GitHub URL
                    of the form ``https://github.com/owner/repo``.
        clone_root: Directory under which remote repos are cloned.
                    Defaults to ``_repos/`` in the current working directory.

    Returns:
        Absolute ``Path`` to the repository root directory.

    Raises:
        ValueError:   If *source* is not a recognised local path or GitHub URL.
        FileNotFoundError: If a local path does not exist.
        RuntimeError: If the clone fails.
    """
    # --- local path ---
    local = Path(source)
    if local.exists():
        resolved = local.resolve()
        logger.info("Using local repo: %s", resolved)
        return resolved

    # --- GitHub URL ---
    match = _GITHUB_RE.match(source)
    if match:
        owner = match.group("owner")
        repo = match.group("repo")
        return _clone_github(owner, repo, clone_root)

    raise ValueError(
        f"Cannot resolve repo source {source!r}. "
        "Pass a local directory path or a GitHub URL "
        "(e.g. https://github.com/owner/repo)."
    )


def _clone_github(owner: str, repo: str, clone_root: Path) -> Path:
    """Clone *owner/repo* from GitHub into *clone_root* and return the path.

    If the directory already exists, a ``git pull`` is performed instead
    (so repeated calls don't re-download from scratch).
    """
    try:
        import git  # GitPython
    except ImportError as exc:
        raise RuntimeError(
            "GitPython is required to clone repositories. "
            "Install it with: pip install gitpython"
        ) from exc

    clone_dir = clone_root / owner / repo
    url = f"https://github.com/{owner}/{repo}.git"

    if clone_dir.exists():
        logger.info("Repo already cloned at %s — pulling latest changes", clone_dir)
        try:
            git_repo = git.Repo(clone_dir)
            git_repo.remotes.origin.pull()
        except git.GitCommandError as exc:
            logger.warning("git pull failed (%s); using cached clone as-is", exc)
    else:
        logger.info("Cloning %s into %s …", url, clone_dir)
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            git.Repo.clone_from(url, str(clone_dir), depth=1)
        except git.GitCommandError as exc:
            # Clean up any partial clone
            if clone_dir.exists():
                shutil.rmtree(clone_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to clone {url}: {exc}") from exc

    logger.info("Repo ready at %s", clone_dir.resolve())
    return clone_dir.resolve()
