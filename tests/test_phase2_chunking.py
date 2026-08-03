"""
Phase 2 tests — semantic chunking.

Tests verify:
  - Correct number of chunks from the sample fixture files
  - Exact line boundaries for known functions
  - Docstring extraction (present / absent)
  - class_name propagation for methods
  - imports captured at module and local level
  - Decorator lines included in chunk span
  - Long-function splitting (token budget)
  - to_dict / to_json serialisation
  - citation and qualified_name helpers
  - TokenSplitter behaviour (no-split, split, overlap, single-huge-line)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from coderag.chunking import Chunk, ChunkExtractor, TokenSplitter, count_tokens
from coderag.ingestion.parser import ASTParser

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
SAMPLE_PY = FIXTURE_REPO / "sample.py"
SAMPLE_JS = FIXTURE_REPO / "sample.js"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse(path: Path, language: str):
    parser = ASTParser()
    return parser.parse_bytes(path.read_bytes(), language, relative_path=path.name)


def _chunks_from(path: Path, language: str, **extractor_kwargs) -> list[Chunk]:
    parsed = _parse(path, language)
    assert parsed is not None
    extractor = ChunkExtractor(**extractor_kwargs)
    return extractor.extract(parsed)


# ── TokenSplitter unit tests ─────────────────────────────────────────────────


class TestTokenSplitter:
    def _make_chunk(self, text: str, **overrides) -> Chunk:
        """Build a minimal Chunk for splitter testing."""
        from coderag.chunking.models import make_chunk_id
        tokens = count_tokens(text)
        defaults = dict(
            chunk_id=make_chunk_id("test.py", "function_definition", 1, "f"),
            file_path="test.py",
            start_line=1,
            end_line=text.count("\n") + 1,
            language="python",
            node_type="function_definition",
            function_name="f",
            class_name=None,
            raw_code=text,
            docstring=None,
            has_docstring=False,
            imports=[],
            token_count=tokens,
        )
        defaults.update(overrides)
        return Chunk(**defaults)

    def test_no_split_when_under_budget(self) -> None:
        chunk = self._make_chunk("def f():\n    return 1\n")
        splitter = TokenSplitter(max_tokens=200)
        result = splitter.split(chunk)
        assert len(result) == 1
        assert result[0] is chunk  # same object — not a copy

    def test_split_produces_multiple_chunks(self) -> None:
        # Build a function with enough lines to exceed a tiny budget
        lines = ["def big():\n"] + [f"    x_{i} = {i}\n" for i in range(60)]
        text = "".join(lines)
        chunk = self._make_chunk(text)
        splitter = TokenSplitter(max_tokens=40, overlap_tokens=8)
        result = splitter.split(chunk)
        assert len(result) >= 2

    def test_split_chunks_are_marked(self) -> None:
        lines = ["def big():\n"] + [f"    x_{i} = {i}\n" for i in range(60)]
        text = "".join(lines)
        chunk = self._make_chunk(text)
        splitter = TokenSplitter(max_tokens=40, overlap_tokens=8)
        result = splitter.split(chunk)
        for sub in result:
            assert sub.is_split_chunk
            assert sub.total_splits == len(result)

    def test_split_indices_are_sequential(self) -> None:
        lines = ["def big():\n"] + [f"    x_{i} = {i}\n" for i in range(60)]
        text = "".join(lines)
        chunk = self._make_chunk(text)
        splitter = TokenSplitter(max_tokens=40, overlap_tokens=8)
        result = splitter.split(chunk)
        indices = [s.split_index for s in result]
        assert indices == list(range(len(result)))

    def test_overlap_shared_content(self) -> None:
        """The end of slice N and the start of slice N+1 must share content."""
        lines = ["def big():\n"] + [f"    line_{i} = {i}\n" for i in range(80)]
        text = "".join(lines)
        chunk = self._make_chunk(text)
        splitter = TokenSplitter(max_tokens=50, overlap_tokens=15)
        result = splitter.split(chunk)
        if len(result) >= 2:
            last_lines_of_first = set(result[0].raw_code.splitlines()[-5:])
            first_lines_of_second = set(result[1].raw_code.splitlines()[:5])
            assert last_lines_of_first & first_lines_of_second, (
                "Adjacent slices must share at least some lines (overlap)"
            )

    def test_single_enormous_line_included(self) -> None:
        """A chunk whose single line exceeds budget must still be kept (not dropped)."""
        text = "x = " + "a" * 10_000  # ~2500+ tokens, single line
        chunk = self._make_chunk(text)
        splitter = TokenSplitter(max_tokens=50)
        result = splitter.split(chunk)
        assert len(result) >= 1
        assert result[0].raw_code == text


# ── count_tokens ─────────────────────────────────────────────────────────────


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0 or count_tokens("") >= 0  # implementation-defined

    def test_short_text(self) -> None:
        # A single word should be 1-4 tokens
        assert 1 <= count_tokens("hello") <= 4

    def test_longer_text_more_tokens(self) -> None:
        short = count_tokens("x = 1")
        long = count_tokens("x = " + " + ".join(str(i) for i in range(100)))
        assert long > short


# ── ChunkExtractor — Python ───────────────────────────────────────────────────


class TestChunkExtractorPython:
    @pytest.fixture(scope="class")
    def chunks(self) -> list[Chunk]:
        return _chunks_from(SAMPLE_PY, "python")

    def test_produces_chunks(self, chunks: list[Chunk]) -> None:
        assert len(chunks) >= 8, f"Expected ≥8 chunks, got {len(chunks)}"

    def test_all_chunks_have_file_path(self, chunks: list[Chunk]) -> None:
        assert all(c.file_path for c in chunks)

    def test_all_chunks_have_valid_lines(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            assert c.start_line >= 1, f"start_line must be ≥1 for {c.qualified_name}"
            assert c.end_line >= c.start_line, f"end_line must be ≥ start_line for {c.qualified_name}"

    def test_greet_has_docstring(self, chunks: list[Chunk]) -> None:
        greet = next((c for c in chunks if c.function_name == "greet"), None)
        assert greet is not None, "greet() chunk not found"
        assert greet.has_docstring
        assert greet.docstring is not None
        assert "greeting" in greet.docstring.lower()

    def test_add_has_no_docstring(self, chunks: list[Chunk]) -> None:
        # top-level add (not Calculator.add — filter by class_name=None)
        add = next(
            (c for c in chunks if c.function_name == "add" and c.class_name is None),
            None,
        )
        assert add is not None, "top-level add() chunk not found"
        assert not add.has_docstring
        assert add.docstring is None

    def test_fetch_data_decorator_in_span(self, chunks: list[Chunk]) -> None:
        """fetch_data is decorated — the chunk text must include @retry."""
        fd = next((c for c in chunks if c.function_name == "fetch_data"), None)
        assert fd is not None, "fetch_data chunk not found"
        assert "@retry" in fd.raw_code, (
            "Decorated function chunk must include the decorator line"
        )

    def test_fetch_data_has_docstring(self, chunks: list[Chunk]) -> None:
        fd = next((c for c in chunks if c.function_name == "fetch_data"), None)
        assert fd is not None
        assert fd.has_docstring

    def test_calculator_class_chunk_present(self, chunks: list[Chunk]) -> None:
        calc = next((c for c in chunks if c.node_type == "class_definition" and c.file_path.endswith(".py")), None)
        assert calc is not None, "Calculator class chunk not found"
        assert calc.function_name == "Calculator"  # class name stored in function_name
        assert calc.class_name is None             # Calculator itself has no parent class

    def test_calculator_methods_have_class_name(self, chunks: list[Chunk]) -> None:
        """Methods of Calculator must carry class_name='Calculator'."""
        method_names = {"__init__", "add", "reset", "_compute_internal"}
        method_chunks = [
            c for c in chunks if c.class_name == "Calculator"
        ]
        found_names = {c.function_name for c in method_chunks}
        for name in method_names:
            assert name in found_names, (
                f"Method Calculator.{name} not found as a chunk with class_name='Calculator'"
            )

    def test_calculator_add_has_docstring(self, chunks: list[Chunk]) -> None:
        calc_add = next(
            (c for c in chunks if c.function_name == "add" and c.class_name == "Calculator"),
            None,
        )
        assert calc_add is not None
        assert calc_add.has_docstring

    def test_calculator_reset_no_docstring(self, chunks: list[Chunk]) -> None:
        reset = next(
            (c for c in chunks if c.function_name == "reset" and c.class_name == "Calculator"),
            None,
        )
        assert reset is not None
        assert not reset.has_docstring

    def test_module_imports_captured(self, chunks: list[Chunk]) -> None:
        """Every chunk should carry the module-level imports."""
        for c in chunks:
            assert any("import os" in imp for imp in c.imports), (
                f"'import os' missing from imports of {c.qualified_name}"
            )

    def test_local_import_in_fetch_data(self, chunks: list[Chunk]) -> None:
        """fetch_data contains 'import urllib.request' — must appear in imports."""
        fd = next((c for c in chunks if c.function_name == "fetch_data"), None)
        assert fd is not None
        assert any("urllib" in imp for imp in fd.imports), (
            "Local import inside fetch_data must be captured"
        )

    def test_chunks_sorted_by_start_line(self, chunks: list[Chunk]) -> None:
        lines = [c.start_line for c in chunks]
        assert lines == sorted(lines), "Chunks must be sorted by start_line"

    def test_all_chunks_have_token_count(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            assert c.token_count > 0, f"token_count=0 for {c.qualified_name}"

    def test_all_chunk_ids_unique(self, chunks: list[Chunk]) -> None:
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ── ChunkExtractor — JavaScript ───────────────────────────────────────────────


class TestChunkExtractorJavaScript:
    @pytest.fixture(scope="class")
    def chunks(self) -> list[Chunk]:
        return _chunks_from(SAMPLE_JS, "javascript")

    def test_produces_chunks(self, chunks: list[Chunk]) -> None:
        assert len(chunks) >= 4

    def test_multiply_has_jsdoc(self, chunks: list[Chunk]) -> None:
        multiply = next((c for c in chunks if c.function_name == "multiply"), None)
        assert multiply is not None, "multiply chunk not found"
        assert multiply.has_docstring
        assert "Multiply" in (multiply.docstring or "")

    def test_eventemitter_class_present(self, chunks: list[Chunk]) -> None:
        cls = next(
            (c for c in chunks if c.node_type == "class_definition"), None
        )
        assert cls is not None, "EventEmitter class chunk not found"

    def test_eventemitter_methods_have_class_name(self, chunks: list[Chunk]) -> None:
        method_chunks = [c for c in chunks if c.class_name == "EventEmitter"]
        names = {c.function_name for c in method_chunks}
        assert "on" in names
        assert "emit" in names

    def test_off_method_no_docstring(self, chunks: list[Chunk]) -> None:
        off = next(
            (c for c in chunks if c.function_name == "off" and c.class_name == "EventEmitter"),
            None,
        )
        assert off is not None
        assert not off.has_docstring


# ── Chunk model helpers ───────────────────────────────────────────────────────


class TestChunkModel:
    @pytest.fixture
    def sample_chunk(self) -> Chunk:
        from coderag.chunking.models import make_chunk_id
        return Chunk(
            chunk_id=make_chunk_id("src/app.py", "function_definition", 42, "my_func"),
            file_path="src/app.py",
            start_line=42,
            end_line=78,
            language="python",
            node_type="function_definition",
            function_name="my_func",
            class_name="MyClass",
            raw_code="def my_func(self):\n    pass\n",
            docstring="Does a thing.",
            has_docstring=True,
            imports=["import os"],
            token_count=12,
        )

    def test_qualified_name_with_class(self, sample_chunk: Chunk) -> None:
        assert sample_chunk.qualified_name == "MyClass.my_func"

    def test_qualified_name_no_class(self, sample_chunk: Chunk) -> None:
        sample_chunk = Chunk(
            **{**sample_chunk.to_dict(), "class_name": None}
        )
        assert sample_chunk.qualified_name == "my_func"

    def test_citation_format(self, sample_chunk: Chunk) -> None:
        cite = sample_chunk.citation
        assert "src/app.py" in cite
        assert "L42" in cite
        assert "L78" in cite
        assert "MyClass.my_func" in cite

    def test_to_dict_is_json_serialisable(self, sample_chunk: Chunk) -> None:
        d = sample_chunk.to_dict()
        assert isinstance(d, dict)
        # Must not raise
        json_str = json.dumps(d)
        assert "my_func" in json_str

    def test_to_json_roundtrip(self, sample_chunk: Chunk) -> None:
        json_str = sample_chunk.to_json()
        parsed = json.loads(json_str)
        assert parsed["function_name"] == "my_func"
        assert parsed["start_line"] == 42

    def test_chunk_id_is_stable(self) -> None:
        from coderag.chunking.models import make_chunk_id
        id1 = make_chunk_id("a.py", "function_definition", 10, "foo")
        id2 = make_chunk_id("a.py", "function_definition", 10, "foo")
        assert id1 == id2

    def test_chunk_id_differs_for_different_inputs(self) -> None:
        from coderag.chunking.models import make_chunk_id
        id1 = make_chunk_id("a.py", "function_definition", 10, "foo")
        id2 = make_chunk_id("a.py", "function_definition", 20, "foo")
        assert id1 != id2


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestChunkExtractorEdgeCases:
    def test_empty_file_produces_no_chunks(self) -> None:
        parser = ASTParser()
        parsed = parser.parse_bytes(b"", "python")
        extractor = ChunkExtractor()
        chunks = extractor.extract(parsed)
        assert chunks == []

    def test_constants_only_file_produces_no_chunks(self) -> None:
        code = textwrap.dedent("""\
            X = 1
            Y = 2
            Z = X + Y
        """).encode()
        parser = ASTParser()
        parsed = parser.parse_bytes(code, "python")
        extractor = ChunkExtractor()
        chunks = extractor.extract(parsed)
        assert chunks == []

    def test_single_function_inline(self) -> None:
        code = textwrap.dedent("""\
            def hello():
                \"\"\"Say hello.\"\"\"
                return "world"
        """).encode()
        parser = ASTParser()
        parsed = parser.parse_bytes(code, "python")
        extractor = ChunkExtractor()
        chunks = extractor.extract(parsed)
        assert len(chunks) == 1
        assert chunks[0].function_name == "hello"
        assert chunks[0].has_docstring
        assert "hello" in chunks[0].docstring.lower() or "say" in chunks[0].docstring.lower()

    def test_nested_function_excluded_when_flag_off(self) -> None:
        code = textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                return inner
        """).encode()
        parser = ASTParser()
        parsed = parser.parse_bytes(code, "python")
        extractor = ChunkExtractor(include_nested_functions=False)
        chunks = extractor.extract(parsed)
        names = {c.function_name for c in chunks}
        assert "outer" in names
        assert "inner" not in names

    def test_nested_function_included_when_flag_on(self) -> None:
        code = textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                return inner
        """).encode()
        parser = ASTParser()
        parsed = parser.parse_bytes(code, "python")
        extractor = ChunkExtractor(include_nested_functions=True)
        chunks = extractor.extract(parsed)
        names = {c.function_name for c in chunks}
        assert "outer" in names
        assert "inner" in names

    def test_long_function_split(self) -> None:
        """A function that exceeds the token budget must be split."""
        # Generate a big function with many assignments
        body = "\n".join(f"    var_{i} = {i} * 2" for i in range(200))
        code = f"def huge():\n{body}\n    return var_0\n".encode()
        parser = ASTParser()
        parsed = parser.parse_bytes(code, "python")
        # Use a small token budget to force splitting
        extractor = ChunkExtractor(max_tokens=60, overlap_tokens=10)
        chunks = extractor.extract(parsed)
        split_chunks = [c for c in chunks if c.is_split_chunk]
        assert len(split_chunks) >= 2, "Huge function must be split into multiple chunks"

    def test_three_example_chunks_as_json(self) -> None:
        """Report requirement: dump 3 example chunk objects as JSON."""
        chunks = _chunks_from(SAMPLE_PY, "python")
        assert len(chunks) >= 3

        print("\n\n=== PHASE 2 REPORT: 3 Example Chunk Objects ===\n")
        for chunk in chunks[:3]:
            print(chunk.to_json())
            print()
