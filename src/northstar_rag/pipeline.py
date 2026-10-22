"""End-to-end retrieval, safety, abstention, and citation validation."""

import re

from .embeddings import Embedder
from .generation import (
    NO_ANSWER,
    ExtractiveGenerator,
    Generator,
    citations_are_valid,
    citations_in,
    is_no_answer,
)
from .hybrid_search import HybridSearcher
from .models import AnswerResult, SearchHit
from .settings import Settings

UNSAFE_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(prior|previous)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?api\s+keys?", re.IGNORECASE),
)


def contains_unsafe_instruction(text: str) -> bool:
    return any(pattern.search(text) for pattern in UNSAFE_PATTERNS)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        searcher: HybridSearcher,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.searcher = searcher
        self.generator = generator or ExtractiveGenerator()

    def retrieve(self, question: str) -> tuple[list[SearchHit], list[str]]:
        query_vector = self.embedder.encode_query(question)
        candidates = self.searcher.search(
            question,
            query_vector,
            top_k=self.settings.candidate_k,
            candidate_k=self.settings.candidate_k,
        )
        safe_hits = []
        events = []
        for rank, hit in enumerate(candidates):
            if contains_unsafe_instruction(hit.chunk.text):
                if rank < self.settings.top_k:
                    events.append(
                        f"filtered_retrieved_instruction:{hit.chunk.chunk_id}"
                    )
            else:
                safe_hits.append(hit)
        return safe_hits[: self.settings.top_k], events

    def answer(self, question: str, min_score: float | None = None) -> AnswerResult:
        threshold = self.settings.min_score if min_score is None else min_score
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if not question.strip():
            return AnswerResult(
                question=question,
                answer=NO_ANSWER,
                status="abstained",
                abstention_reason="empty_question",
            )
        if contains_unsafe_instruction(question):
            return AnswerResult(
                question=question,
                answer="Request blocked by input safety policy.",
                status="blocked_input",
                safety_events=["blocked_user_instruction"],
            )

        hits, events = self.retrieve(question)
        retrieved_sources = list(dict.fromkeys(hit.chunk.document_id for hit in hits))
        top_score = max((hit.dense_score for hit in hits), default=0.0)
        if top_score <= 0 or top_score < threshold:
            return AnswerResult(
                question=question,
                answer=NO_ANSWER,
                status="abstained",
                retrieved_sources=retrieved_sources,
                top_score=top_score,
                abstention_reason="no_accepted_evidence",
                safety_events=events,
            )

        accepted = [
            hit for hit in hits if hit.dense_score > 0 and hit.dense_score >= threshold
        ]
        answer = self.generator.generate(question, accepted)
        if not answer.strip() or is_no_answer(answer):
            return AnswerResult(
                question=question,
                answer=NO_ANSWER,
                status="abstained",
                retrieved_sources=retrieved_sources,
                top_score=top_score,
                abstention_reason="generator_fallback",
                safety_events=events,
            )
        if not citations_are_valid(answer, accepted):
            return AnswerResult(
                question=question,
                answer=NO_ANSWER,
                status="invalid_citations",
                retrieved_sources=retrieved_sources,
                top_score=top_score,
                safety_events=events + ["invalid_citations_rejected"],
            )
        return AnswerResult(
            question=question,
            answer=answer,
            status="answered",
            citations=citations_in(answer),
            retrieved_sources=retrieved_sources,
            top_score=top_score,
            safety_events=events,
        )

    def close(self) -> None:
        close = getattr(self.generator, "close", None)
        if close is not None:
            close()
