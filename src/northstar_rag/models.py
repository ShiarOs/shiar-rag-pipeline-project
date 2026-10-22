"""Typed records shared by ingestion, retrieval, generation, and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    text: str
    path: Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    section: str
    position: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    dense_score: float
    lexical_score: float
    fused_score: float


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_sources: tuple[str, ...]
    required_terms: tuple[str, ...]
    should_abstain: bool


@dataclass
class AnswerResult:
    question: str
    answer: str
    status: str
    citations: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    top_score: float | None = None
    abstention_reason: str | None = None
    safety_events: list[str] = field(default_factory=list)
