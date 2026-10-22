"""Markdown source loading and paragraph-aware chunk construction."""

from itertools import count
from pathlib import Path

from .models import Chunk, SourceDocument


def load_documents(source_dir: Path) -> list[SourceDocument]:
    """Load Markdown sources and derive stable IDs from their filenames."""

    documents: list[SourceDocument] = []
    for path in sorted(source_dir.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8").strip()
        lines = raw_text.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"Expected a level-one title in {path.name}")
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        if not title or not body:
            raise ValueError(f"Expected a nonempty title and body in {path.name}")
        documents.append(
            SourceDocument(
                document_id=path.stem,
                title=title,
                text=body,
                path=path,
            )
        )

    if not documents:
        raise ValueError(f"No Markdown sources found in {source_dir}")
    return documents


def paragraphs_by_section(text: str, title: str) -> list[tuple[str, str]]:
    """Split a Markdown body into paragraph and section pairs."""

    section = title
    current_lines: list[str] = []
    paragraphs: list[tuple[str, str]] = []

    def flush() -> None:
        if current_lines:
            paragraphs.append((section, " ".join(current_lines)))
            current_lines.clear()

    for line in text.splitlines():
        value = line.strip()
        if value.startswith("## "):
            flush()
            section = value[3:].strip()
        elif value:
            current_lines.append(value)
        else:
            flush()

    flush()
    return paragraphs


def split_paragraph(
    text: str, max_words: int = 100, overlap_words: int = 12
) -> list[str]:
    """Keep short paragraphs whole and window only overlong paragraphs."""

    if max_words <= 0 or not 0 <= overlap_words < max_words:
        raise ValueError("Use max_words > 0 and 0 <= overlap_words < max_words.")
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []

    step = max_words - overlap_words
    last_start = max(len(words) - overlap_words, 1)
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, last_start, step)
        if words[start : start + max_words]
    ]


def chunk_documents(
    documents: list[SourceDocument],
    max_words: int = 100,
    overlap_words: int = 12,
) -> list[Chunk]:
    """Create citable chunks while preserving paragraph and heading context."""

    chunks: list[Chunk] = []
    for document in documents:
        position = count()
        for section, paragraph in paragraphs_by_section(document.text, document.title):
            for passage in split_paragraph(paragraph, max_words, overlap_words):
                chunk_position = next(position)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}-{chunk_position}",
                        document_id=document.document_id,
                        title=document.title,
                        section=section,
                        position=chunk_position,
                        text=passage,
                    )
                )
    if not chunks:
        raise ValueError("No nonempty chunks were produced from the source documents.")
    return chunks
