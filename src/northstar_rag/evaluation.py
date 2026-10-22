"""Deterministic evaluation metrics and Markdown/JSON report generation."""

import json
from pathlib import Path
from typing import Any

from .generation import CITATION_PATTERN, ExtractiveGenerator
from .models import EvaluationCase
from .pipeline import RAGPipeline


def load_cases(path: Path) -> list[EvaluationCase]:
    records = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        EvaluationCase(
            case_id=str(row["id"]),
            question=str(row["question"]),
            expected_sources=tuple(row["expected_sources"]),
            required_terms=tuple(row["required_terms"]),
            should_abstain=bool(row["should_abstain"]),
        )
        for row in records
    ]
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def reciprocal_rank(ranked_sources: list[str], expected_sources: set[str]) -> float:
    for rank, source_id in enumerate(ranked_sources, start=1):
        if source_id in expected_sources:
            return 1.0 / rank
    return 0.0


def _literal_support(answer: str, citations: list[str], hits) -> bool:
    source_text: dict[str, str] = {}
    for hit in hits:
        if hit.chunk.document_id in citations:
            source_text[hit.chunk.document_id] = (
                source_text.get(hit.chunk.document_id, "") + " " + hit.chunk.text
            )

    paragraphs = [
        paragraph.strip() for paragraph in answer.split("\n\n") if paragraph.strip()
    ]
    if not paragraphs:
        return False
    for paragraph in paragraphs:
        paragraph_citations = [
            source.strip()
            for group in CITATION_PATTERN.findall(paragraph)
            for source in group.split(",")
        ]
        content = " ".join(CITATION_PATTERN.sub("", paragraph).split()).casefold()
        if not content or not paragraph_citations:
            return False
        supported = any(
            source_id in source_text
            and content in " ".join(source_text[source_id].split()).casefold()
            for source_id in paragraph_citations
        )
        if not supported:
            return False
    return True


def evaluate(pipeline: RAGPipeline, cases: list[EvaluationCase]) -> dict[str, Any]:
    """Run cases and return per-question evidence plus aggregate metrics."""

    if not cases:
        raise ValueError("Evaluation requires at least one case")
    rows = []
    for case in cases:
        result = pipeline.answer(case.question)
        hits, retrieval_events = (
            pipeline.retrieve(case.question) if case.question.strip() else ([], [])
        )
        ranked_sources = [hit.chunk.document_id for hit in hits]
        expected = set(case.expected_sources)
        first_relevant_rank = next(
            (
                rank
                for rank, source in enumerate(ranked_sources, start=1)
                if source in expected
            ),
            None,
        )
        if case.should_abstain:
            relevance = None
            citation_correct = None
        else:
            answer_terms = result.answer.casefold()
            relevance = (
                sum(term.casefold() in answer_terms for term in case.required_terms)
                / len(case.required_terms)
                if case.required_terms
                else float(result.status == "answered")
            )
            citation_correct = (
                result.status == "answered"
                and expected.issubset(set(result.citations))
                and set(result.citations).issubset(set(ranked_sources))
            )

        faithfulness: bool | None = None
        if result.status == "answered":
            if isinstance(pipeline.generator, ExtractiveGenerator):
                faithfulness = _literal_support(result.answer, result.citations, hits)
            else:
                faithfulness = None

        decision_correct = (
            result.status == "abstained"
            if case.should_abstain
            else result.status == "answered"
        )
        safety_events = list(dict.fromkeys(result.safety_events + retrieval_events))
        rows.append(
            {
                "id": case.case_id,
                "question": case.question,
                "expected_sources": list(case.expected_sources),
                "should_abstain": case.should_abstain,
                "retrieved_sources": ranked_sources,
                "first_relevant_rank": first_relevant_rank,
                "top_score": result.top_score,
                "status": result.status,
                "abstention_reason": result.abstention_reason,
                "answer": result.answer,
                "citations": result.citations,
                "relevance": relevance,
                "citation_correct": citation_correct,
                "faithful": faithfulness,
                "decision_correct": decision_correct,
                "safety_events": safety_events,
            }
        )

    supported = [row for row in rows if not row["should_abstain"]]
    answered = [row for row in rows if row["status"] == "answered"]
    retrieval_recall = (
        sum(
            len(set(row["retrieved_sources"]) & set(row["expected_sources"]))
            / len(set(row["expected_sources"]))
            for row in supported
        )
        / len(supported)
        if supported
        else None
    )
    faithfulness_values = [row["faithful"] for row in answered]
    faithfulness = (
        sum(value is True for value in faithfulness_values) / len(faithfulness_values)
        if faithfulness_values
        and all(value is not None for value in faithfulness_values)
        else None
    )
    return {
        "configuration": {
            "embedding_model": pipeline.settings.embedding_model,
            "embedding_revision": pipeline.settings.embedding_revision,
            "vector_index": "FAISS IndexFlatIP with normalized vectors",
            "lexical_index": "BM25Okapi",
            "top_k": pipeline.settings.top_k,
            "min_score": pipeline.settings.min_score,
            "generation_mode": pipeline.generator.mode,
        },
        "metrics": {
            "recall_at_k": retrieval_recall,
            "mean_reciprocal_rank": (
                sum(
                    reciprocal_rank(
                        row["retrieved_sources"], set(row["expected_sources"])
                    )
                    for row in supported
                )
                / len(supported)
                if supported
                else None
            ),
            "answer_relevance": (
                sum(row["relevance"] or 0.0 for row in supported) / len(supported)
                if supported
                else None
            ),
            "citation_correctness": (
                sum(row["citation_correct"] is True for row in supported)
                / len(supported)
                if supported
                else None
            ),
            "faithfulness_literal": faithfulness,
            "decision_accuracy": sum(row["decision_correct"] for row in rows)
            / len(rows),
        },
        "rows": rows,
    }


def render_report(
    summary: dict[str, Any],
    question_set: str,
    calibration: dict[str, Any] | None = None,
) -> str:
    metrics = summary["metrics"]
    config = summary["configuration"]

    def fmt(value: float | None) -> str:
        return "not rated" if value is None else f"{value:.3f}"

    lines = [
        "# Northstar RAG Evaluation Report",
        "",
        "## Configuration",
        "",
        f"- Question set: `{question_set}`",
        f"- Embedding: `{config['embedding_model']}` revision `{config['embedding_revision']}`",
        f"- Retrieval: `{config['vector_index']}` plus `{config['lexical_index']}`",
        f"- `top_k={config['top_k']}`, `min_score={config['min_score']:.3f}`",
        f"- Generation: `{config['generation_mode']}`",
        "",
        "## Metric Rubric",
        "",
        "- Recall@k: mean fraction of labeled relevant source IDs present in the retrieved source set for supported cases.",
        "- MRR: reciprocal rank of the first relevant retrieved chunk, averaged over supported cases.",
        "- Answer relevance: literal coverage of the expected terms; this is not a semantic judge.",
        "- Citation correctness: expected source IDs are cited and every cited ID was retrieved.",
        "- Faithfulness: exact textual support for extractive answers; live paraphrases are not rated automatically.",
        "- Decision accuracy: supported questions are answered and unsupported questions are abstained.",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| Recall@{config['top_k']} | {fmt(metrics['recall_at_k'])} |",
        f"| MRR | {fmt(metrics['mean_reciprocal_rank'])} |",
        f"| Answer relevance | {fmt(metrics['answer_relevance'])} |",
        f"| Citation correctness | {fmt(metrics['citation_correctness'])} |",
        f"| Faithfulness, literal | {fmt(metrics['faithfulness_literal'])} |",
        f"| Decision accuracy | {fmt(metrics['decision_accuracy'])} |",
        "",
        "## Per-Question Results",
        "",
        "| ID | Expected sources | Retrieved sources | Status | Top score | Citations | Relevant | Faithful | Safety events |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        expected = ", ".join(row["expected_sources"]) or "-"
        retrieved = ", ".join(row["retrieved_sources"]) or "-"
        citations = ", ".join(row["citations"]) or "-"
        score = "-" if row["top_score"] is None else f"{row['top_score']:.3f}"
        relevance = "n/a" if row["relevance"] is None else f"{row['relevance']:.2f}"
        faithful = "n/a" if row["faithful"] is None else str(row["faithful"]).lower()
        events = ", ".join(row["safety_events"]) or "-"
        lines.append(
            f"| {row['id']} | {expected} | {retrieved} | {row['status']} | "
            f"{score} | {citations} | {relevance} | {faithful} | {events} |"
        )
    lines.extend(["", "## Failure And Fragility Analysis", ""])
    failures = [row for row in summary["rows"] if not row["decision_correct"]]
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['id']}`: status `{row['status']}`, expected sources "
                f"`{', '.join(row['expected_sources']) or '-'}`, retrieved "
                f"`{', '.join(row['retrieved_sources']) or '-'}`. Inspect its evidence and query wording."
            )
    else:
        lines.append(
            "- No decision errors on this small labeled set; this is not a production reliability claim."
        )
    fragile = sorted(
        (
            row
            for row in summary["rows"]
            if not row["should_abstain"] and row["top_score"] is not None
        ),
        key=lambda row: row["top_score"],
    )[:2]
    for row in fragile:
        lines.append(
            f"- Fragile supported case `{row['id']}` has top score `{row['top_score']:.3f}`; add paraphrases before trusting the threshold."
        )
    lines.extend(
        [
            "",
        ]
    )
    lines.extend(["## Threshold Decision", ""])
    if calibration is None:
        lines.append(
            "No calibration trace was available. Run calibration first, record the selected threshold, then rerun final evaluation."
        )
    else:
        lines.append(
            f"Configured threshold: `{calibration['threshold']:.3f}`. On the separate calibration set, it produced "
            f"`{calibration['false_rejections']}` false rejections and "
            f"`{calibration['false_acceptances']}` false acceptances. This is a small tuning fixture, not a reliability guarantee."
        )
        lines.extend(
            [
                "",
                "| Calibration ID | Expected decision | Top score | Threshold decision |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for row in calibration["rows"]:
            expected = "abstain" if row["should_abstain"] else "answer"
            decision = "abstain" if row["would_abstain"] else "answer"
            lines.append(
                f"| {row['id']} | {expected} | {row['top_score']:.3f} | {decision} |"
            )
    lines.extend(
        [
            "",
            "Similarity alone does not prove policy applicability; inspect near-miss questions and generator decisions too.",
            "",
            "## Prioritized Recommendations",
            "",
            "1. Improve multi-source coverage and citation completeness for `q09`; retrieve both privacy and incident evidence before answering.",
            "2. Expand calibration with new human-authored near-miss questions; the six supplied cases do not prove that the selected threshold generalizes.",
            "3. Independently review live-generated claims and broaden prompt-injection and privacy tests before operational use.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    summary: dict[str, Any],
    output: Path,
    question_set: str,
    calibration: dict[str, Any] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_report(summary, question_set, calibration), encoding="utf-8"
    )
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
