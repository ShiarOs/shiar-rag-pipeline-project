"""Tests for evaluation labels, metrics, and report outputs."""

import json
from pathlib import Path

import numpy as np

from northstar_rag.embeddings import Embedder
from northstar_rag.evaluation import load_cases, render_report, write_report
from northstar_rag.generation import ExtractiveGenerator
from northstar_rag.hybrid_search import HybridSearcher
from northstar_rag.pipeline import RAGPipeline
from northstar_rag.settings import Settings

PROJECT_ROOT = Path(__file__).parents[1]


class FakeEmbedder:
    dimension = 8

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        categories = (
            ("refund", "payment", "invoice", "charge"),
            ("password", "authentication", "authenticator", "sign-in"),
            ("api", "429", "rate limit", "request"),
            ("privacy", "personal", "passport", "email"),
            ("extension", "deadline", "assignment"),
            ("attendance", "absence", "session"),
            ("safety", "instruction", "prompt", "retrieved"),
        )
        for row, text in enumerate(texts):
            lowered = text.casefold()
            category = next(
                (
                    index
                    for index, terms in enumerate(categories)
                    if any(term in lowered for term in terms)
                ),
                7,
            )
            vectors[row, category] = 1.0
        return vectors

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]


def make_test_pipeline(tmp_path: Path) -> RAGPipeline:
    settings = Settings(
        data_dir=PROJECT_ROOT / "data" / "knowledge_base",
        calibration_file=PROJECT_ROOT / "data" / "evaluation" / "calibration.json",
        evaluation_file=PROJECT_ROOT / "data" / "evaluation" / "questions.json",
        index_dir=tmp_path / "index",
        min_score=0.2,
        top_k=8,
        candidate_k=20,
    )
    settings = settings.model_copy(
        update={"embedding_model": "fake-test", "embedding_revision": "test"}
    )
    embedder: Embedder = FakeEmbedder()
    searcher = HybridSearcher.build_or_load(settings, embedder, force_rebuild=True)
    return RAGPipeline(settings, embedder, searcher, ExtractiveGenerator())


def test_supplied_question_sets_have_valid_expected_sources() -> None:
    data_dir = PROJECT_ROOT / "data"
    source_ids = {path.stem for path in (data_dir / "knowledge_base").glob("*.md")}
    calibration = load_cases(data_dir / "evaluation" / "calibration.json")
    final_cases = load_cases(data_dir / "evaluation" / "questions.json")

    assert len(calibration) == 6
    assert len(final_cases) == 16
    assert all(
        set(case.expected_sources) <= source_ids for case in calibration + final_cases
    )
    assert {case.case_id for case in calibration}.isdisjoint(
        case.case_id for case in final_cases
    )


def test_report_writer_emits_markdown_and_json(tmp_path: Path) -> None:
    summary = {
        "configuration": {
            "embedding_model": "test-model",
            "embedding_revision": "test-revision",
            "vector_index": "FAISS IndexFlatIP",
            "lexical_index": "BM25Okapi",
            "top_k": 3,
            "min_score": 0.3,
            "generation_mode": "test",
        },
        "metrics": {
            "recall_at_k": 1.0,
            "mean_reciprocal_rank": 1.0,
            "answer_relevance": 1.0,
            "citation_correctness": 1.0,
            "faithfulness_literal": 1.0,
            "decision_accuracy": 1.0,
        },
        "rows": [],
    }
    output = tmp_path / "report.md"

    write_report(summary, output, "test.json")

    assert "Metric Rubric" in output.read_text(encoding="utf-8")
    assert (
        json.loads(output.with_suffix(".json").read_text(encoding="utf-8")) == summary
    )
    assert "test.json" in render_report(summary, "test.json")


def test_final_evaluation_runs_every_case_without_external_services(
    tmp_path: Path,
) -> None:
    from northstar_rag.evaluation import evaluate

    pipeline = make_test_pipeline(tmp_path)
    cases = load_cases(PROJECT_ROOT / "data" / "evaluation" / "questions.json")
    summary = evaluate(pipeline, cases)

    assert len(summary["rows"]) == 16
    assert 0 <= summary["metrics"]["decision_accuracy"] <= 1
    assert "citation_correctness" in summary["metrics"]
    assert all("safety_events" in row for row in summary["rows"])


def test_calibration_evidence_appears_in_final_report() -> None:
    summary = {
        "configuration": {
            "embedding_model": "test-model",
            "embedding_revision": "test-revision",
            "vector_index": "FAISS IndexFlatIP",
            "lexical_index": "BM25Okapi",
            "top_k": 3,
            "min_score": 0.3,
            "generation_mode": "test",
        },
        "metrics": {
            "recall_at_k": 1.0,
            "mean_reciprocal_rank": 1.0,
            "answer_relevance": 1.0,
            "citation_correctness": 1.0,
            "faithfulness_literal": 1.0,
            "decision_accuracy": 1.0,
        },
        "rows": [],
    }
    calibration = {
        "threshold": 0.3,
        "false_rejections": 1,
        "false_acceptances": 2,
        "rows": [
            {
                "id": "cal01",
                "should_abstain": False,
                "top_score": 0.42,
                "would_abstain": False,
            }
        ],
    }

    report = render_report(summary, "questions.json", calibration)

    assert "false rejections" in report
    assert "false acceptances" in report
    assert "cal01" in report


def test_final_benchmark_keeps_multisource_case_as_a_regression_target() -> None:
    cases = load_cases(PROJECT_ROOT / "data" / "evaluation" / "questions.json")
    multisource = next(case for case in cases if case.case_id == "q09")

    assert set(multisource.expected_sources) == {"privacy-data", "technical-incidents"}
    assert not multisource.should_abstain
