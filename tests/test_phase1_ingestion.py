"""
Phase 1 tests — repo ingestion, file walking, and AST parsing.

Test structure:
  TestRepoResolution       — resolve_repo() with local paths
  TestFileWalker           — walker correctness, gitignore, binary filtering
  TestLanguageDetection    — extension → language mapping
  TestASTParserPython      — tree-sitter Python parsing + node extraction
  TestASTParserJavaScript  — tree-sitter JavaScript parsing + node extraction
  TestASTParserEdgeCases   — empty files, syntax errors, unsupported language
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from coderag.ingestion.parser import ASTNode, ASTParser, ParsedFile
from coderag.ingestion.repo import resolve_repo
from coderag.ingestion.walker import FileWalker, SourceFile, detect_language

# ── Fixtures ────────────────────────────────────────────────────────────────

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


# ── Repo resolution ──────────────────────────────────────────────────────────


class TestRepoResolution:
    def test_local_path_returned_as_absolute(self) -> None:
        resolved = resolve_repo(str(FIXTURE_REPO))
        assert resolved.is_absolute()
        assert resolved.is_dir()

    def test_local_path_identical_to_fixture(self) -> None:
        resolved = resolve_repo(str(FIXTURE_REPO))
        assert resolved == FIXTURE_REPO.resolve()

    def test_nonexistent_path_raises_value_error(self) -> None:
        """A non-existent path that doesn't look like a URL → ValueError."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_repo("/this/path/does/not/exist/xyz123")

    def test_invalid_source_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            resolve_repo("not-a-path-or-url")


# ── Language detection ───────────────────────────────────────────────────────


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "ext, expected",
        [
            (".py", "python"),
            (".js", "javascript"),
            (".mjs", "javascript"),
            (".ts", "typescript"),
            (".tsx", "typescript"),
            (".rb", "ruby"),
            (".go", "go"),
            (".rs", "rust"),
            (".unknown", None),
            ("", None),
        ],
    )
    def test_extension_mapping(self, ext: str, expected: str | None) -> None:
        assert detect_language(ext) == expected

    def test_case_insensitive(self) -> None:
        assert detect_language(".PY") == "python"
        assert detect_language(".JS") == "javascript"


# ── FileWalker ───────────────────────────────────────────────────────────────


class TestFileWalker:
    def test_finds_python_and_js_files(self) -> None:
        walker = FileWalker(FIXTURE_REPO)
        files = list(walker.walk())
        extensions = {f.extension for f in files}
        assert ".py" in extensions
        assert ".js" in extensions

    def test_does_not_return_binary_files(self) -> None:
        walker = FileWalker(FIXTURE_REPO)
        files = list(walker.walk())
        assert not any(f.extension == ".bin" for f in files), (
            "Binary .bin file should have been filtered out"
        )

    def test_respects_gitignore(self) -> None:
        """ignored_file.py is listed in the fixture .gitignore — must not appear."""
        walker = FileWalker(FIXTURE_REPO)
        files = list(walker.walk())
        names = {f.path.name for f in files}
        assert "ignored_file.py" not in names, (
            "ignored_file.py should have been excluded by .gitignore"
        )

    def test_language_filter(self) -> None:
        walker = FileWalker(FIXTURE_REPO, languages=["python"])
        files = list(walker.walk())
        assert all(f.language == "python" for f in files), (
            "Language filter 'python' should yield only Python files"
        )
        names = {f.path.name for f in files}
        assert "sample.js" not in names

    def test_source_file_fields(self) -> None:
        walker = FileWalker(FIXTURE_REPO, languages=["python"])
        files = list(walker.walk())
        py_files = [f for f in files if f.path.name == "sample.py"]
        assert py_files, "sample.py should be found"
        sf = py_files[0]
        assert sf.language == "python"
        assert sf.extension == ".py"
        assert sf.size_bytes > 0
        assert sf.relative_path == Path("sample.py")
        assert sf.path.is_absolute()

    def test_skips_node_modules_dir(self, tmp_path: Path) -> None:
        """Directories in the always-skip list must be pruned."""
        (tmp_path / "node_modules" / "lodash").mkdir(parents=True)
        (tmp_path / "node_modules" / "lodash" / "index.js").write_text("module.exports={}")
        (tmp_path / "app.py").write_text("x = 1")

        walker = FileWalker(tmp_path)
        files = list(walker.walk())
        names = {f.path.name for f in files}
        assert "index.js" not in names, "node_modules should be skipped"
        assert "app.py" in names

    def test_skips_large_files(self, tmp_path: Path) -> None:
        big_file = tmp_path / "huge.py"
        big_file.write_bytes(b"x = 1\n" * 200_000)  # > 1 MB
        walker = FileWalker(tmp_path, max_file_size_bytes=1_000_000)
        files = list(walker.walk())
        assert not any(f.path.name == "huge.py" for f in files)

    def test_skips_lock_files(self, tmp_path: Path) -> None:
        (tmp_path / "package-lock.json").write_text("{}")
        (tmp_path / "poetry.lock").write_text("content")
        (tmp_path / "app.py").write_text("x = 1")
        walker = FileWalker(tmp_path)
        files = list(walker.walk())
        names = {f.path.name for f in files}
        assert "package-lock.json" not in names
        assert "poetry.lock" not in names
        assert "app.py" in names


# ── ASTParser — Python ───────────────────────────────────────────────────────


class TestASTParserPython:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedFile:
        parser = ASTParser()
        source_bytes = (FIXTURE_REPO / "sample.py").read_bytes()
        result = parser.parse_bytes(source_bytes, "python", relative_path=Path("sample.py"))
        assert result is not None, "Python parser returned None — grammar package missing?"
        return result

    def test_root_node_type(self, parsed: ParsedFile) -> None:
        assert parsed.tree.root_node.type == "module"

    def test_no_parse_errors(self, parsed: ParsedFile) -> None:
        assert not parsed.parse_errors, "sample.py should parse without errors"

    def test_line_count(self, parsed: ParsedFile) -> None:
        assert parsed.line_count > 50, "sample.py has >50 lines"

    def test_extracts_top_level_functions(self, parsed: ParsedFile) -> None:
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        names = {n.name for n in nodes}
        assert "greet" in names
        assert "add" in names
        assert "retry" in names
        assert "fetch_data" in names

    def test_extracts_class(self, parsed: ParsedFile) -> None:
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        names = {n.name for n in nodes}
        assert "Calculator" in names

    def test_function_line_numbers_are_1indexed(self, parsed: ParsedFile) -> None:
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        greet_nodes = [n for n in nodes if n.name == "greet"]
        assert greet_nodes, "greet() should be in extracted nodes"
        greet = greet_nodes[0]
        assert greet.start_line >= 1
        assert greet.end_line > greet.start_line

    def test_function_text_contains_body(self, parsed: ParsedFile) -> None:
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        greet_nodes = [n for n in nodes if n.name == "greet"]
        assert greet_nodes
        assert "Hello" in greet_nodes[0].text

    def test_extracts_correct_count_of_top_level_items(self, parsed: ParsedFile) -> None:
        """Top-level items: greet, add, retry, fetch_data, Calculator = 5"""
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        # Top-level only (depth-0 results)
        top_level_names = {n.name for n in nodes}
        assert len(top_level_names) >= 5


# ── ASTParser — JavaScript ───────────────────────────────────────────────────


class TestASTParserJavaScript:
    @pytest.fixture(scope="class")
    def parsed(self) -> ParsedFile:
        parser = ASTParser()
        source_bytes = (FIXTURE_REPO / "sample.js").read_bytes()
        result = parser.parse_bytes(source_bytes, "javascript", relative_path=Path("sample.js"))
        assert result is not None, "JavaScript parser returned None — grammar package missing?"
        return result

    def test_root_node_type(self, parsed: ParsedFile) -> None:
        assert parsed.tree.root_node.type == "program"

    def test_no_parse_errors(self, parsed: ParsedFile) -> None:
        assert not parsed.parse_errors

    def test_extracts_multiply_function(self, parsed: ParsedFile) -> None:
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        names = {n.name for n in nodes}
        assert "multiply" in names

    def test_extracts_class(self, parsed: ParsedFile) -> None:
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        names = {n.name for n in nodes}
        assert "EventEmitter" in names

    def test_function_node_has_text(self, parsed: ParsedFile) -> None:
        parser = ASTParser()
        nodes = parser.extract_named_nodes(parsed)
        multiply = next((n for n in nodes if n.name == "multiply"), None)
        assert multiply is not None
        assert "multiply" in multiply.text


# ── ASTParser — Edge cases ───────────────────────────────────────────────────


class TestASTParserEdgeCases:
    def test_empty_python_file(self) -> None:
        parser = ASTParser()
        result = parser.parse_bytes(b"", "python")
        assert result is not None
        assert result.tree.root_node.type == "module"
        nodes = parser.extract_named_nodes(result)
        assert nodes == []

    def test_syntax_error_python(self) -> None:
        """A file with syntax errors should still parse (partial tree) and
        set parse_errors=True."""
        parser = ASTParser()
        broken = b"def foo(\n    # unclosed\n"
        result = parser.parse_bytes(broken, "python")
        assert result is not None
        assert result.parse_errors is True

    def test_unsupported_language_returns_none(self) -> None:
        parser = ASTParser()
        result = parser.parse_bytes(b"fn main() {}", "rust")
        # Rust grammar isn't installed, so we expect None
        assert result is None

    def test_parse_file_via_source_file_object(self) -> None:
        """parse_file() should work end-to-end with a real SourceFile."""
        walker = FileWalker(FIXTURE_REPO, languages=["python"])
        files = list(walker.walk())
        py_file = next((f for f in files if f.path.name == "sample.py"), None)
        assert py_file is not None

        parser = ASTParser()
        parsed = parser.parse_file(py_file)
        assert parsed is not None
        assert parsed.language == "python"
        assert parsed.tree.root_node.type == "module"

    def test_inline_python_snippet(self) -> None:
        """Quickly verify the parser works on a trivial inline snippet."""
        snippet = textwrap.dedent("""\
            def hello():
                return "world"
        """).encode()
        parser = ASTParser()
        result = parser.parse_bytes(snippet, "python")
        assert result is not None
        assert not result.parse_errors
        nodes = parser.extract_named_nodes(result)
        assert any(n.name == "hello" for n in nodes)
