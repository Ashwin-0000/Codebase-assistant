"""
Chunking sub-package — semantic extraction from parsed source files.

Public API:
  ChunkExtractor(max_tokens, overlap_tokens) → produces Chunk objects
  Chunk                                      → the data model
  TokenSplitter                              → split oversized chunks
  count_tokens(text)                         → token counting utility
"""

from coderag.chunking.extractor import ChunkExtractor
from coderag.chunking.models import Chunk, make_chunk_id
from coderag.chunking.splitter import TokenSplitter, count_tokens

__all__ = [
    "ChunkExtractor",
    "Chunk",
    "make_chunk_id",
    "TokenSplitter",
    "count_tokens",
]
