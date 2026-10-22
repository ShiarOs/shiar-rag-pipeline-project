"""Tests for source loading and paragraph-aware chunk construction."""

from pathlib import Path

import pytest

from northstar_rag.documents import chunk_documents, load_documents, split_paragraph
from northstar_rag.models import SourceDocument

PROJECT_ROOT = Path(__file__).parents[1]


def test_loads_all_knowledge_base_sources() -> None:
    documents = load_documents(PROJECT_ROOT / "data" / "knowledge_base")
    assert len(documents) == 11
    assert documents[0].document_id == "account-mfa"


def test_chunks_keep_source_metadata_and_separate_injection_paragraph() -> None:
    documents = load_documents(PROJECT_ROOT / "data" / "knowledge_base")
    chunks = chunk_documents(documents)

    safety_chunks = [
        chunk for chunk in chunks if chunk.document_id == "retrieved-content-safety"
    ]
    malicious = [
        chunk
        for chunk in safety_chunks
        if "IGNORE ALL PRIOR INSTRUCTIONS" in chunk.text
    ]

    assert len(safety_chunks) == 3
    assert len(malicious) == 1
    assert malicious[0].chunk_id == "retrieved-content-safety-1"
    assert all(chunk.title and chunk.section for chunk in chunks)


def test_long_paragraph_windows_retain_tail_and_overlap() -> None:
    text = " ".join(str(number) for number in range(151))
    chunks = split_paragraph(text, max_words=90, overlap_words=15)

    assert len(chunks) == 2
    assert len(chunks[0].split()) == 90
    assert chunks[0].split()[-15:] == chunks[1].split()[:15]
    assert chunks[1].split()[-1] == "150"


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_chunk_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_paragraph("short text", max_words=size, overlap_words=overlap)


def test_invalid_markdown_source_has_actionable_error(tmp_path: Path) -> None:
    (tmp_path / "invalid.md").write_text("No heading", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.md"):
        load_documents(tmp_path)


def test_chunk_ids_and_positions_are_stable_per_source() -> None:
    document = SourceDocument(
        document_id="example",
        title="Example",
        text="First paragraph.\n\nSecond paragraph.",
        path=Path("example.md"),
    )
    chunks = chunk_documents([document])

    assert [chunk.chunk_id for chunk in chunks] == ["example-0", "example-1"]
    assert [chunk.position for chunk in chunks] == [0, 1]
