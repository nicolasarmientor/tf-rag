
import pytest
from src.ingest.chunker import find_code_block_spans, chunk_text, find_overlap_start

def test_find_code_block_spans_detects_single_block():
    text = "Some text.\n\n```python\nx = 1\n```\n\nMore text."
    spans = find_code_block_spans(text)

    assert len(spans) == 1

def test_find_code_block_spans_detects_multiple_blocks():
    text = "Intro.\n\n```python\nx = 1\n```\n\nMiddle.\n\n```python\ny = 2\n```\n\nEnd."
    spans = find_code_block_spans(text)

    assert len(spans) == 2

def test_find_code_block_spans_span_matches_actual_text():
    text = "Before.\n\n```python\ncode_here\n```\n\nAfter."
    spans = find_code_block_spans(text)
    start, end = spans[0]
    extracted = text[start:end]

    assert extracted.startswith("```")
    assert extracted.endswith("```")

def test_chunk_text_never_splits_code_block():
    filler_before = "word " * 350
    code_block = "```python\n" + "x = 1\n" * 30 + "```"
    filler_after = "word " * 350
    text = filler_before + code_block + filler_after

    spans = find_code_block_spans(text)
    chunks = chunk_text(text, spans)

    code_start, code_end = spans[0]
    full_code_block = text[code_start:code_end]

    found_intact = any(full_code_block in chunk for chunk in chunks)

    assert found_intact

def test_overlap_produces_shared_content():
    text = "word " * 1000
    chunk_end = 1995

    overlap_start = find_overlap_start(text, chunk_end)

    assert overlap_start < chunk_end
    assert overlap_start > 0

def test_chunk_text_handles_empty_string():
    chunks = chunk_text("", [])

    assert chunks == [] or chunks == [""]