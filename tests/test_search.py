"""Tests for rank fusion, hybrid retrieval, and index persistence."""

from pathlib import Path

import numpy as np

from northstar_rag.embeddings import Embedder
from northstar_rag.hybrid_search import HybridSearcher, fuse_rankings
from northstar_rag.settings import Settings

PROJECT_ROOT = Path(__file__).parents[1]


class FakeEmbedder:
    dimension = 8

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.casefold()
            if "refund" in lowered or "payment" in lowered:
                output[row, 0] = 1
            elif "password" in lowered:
                output[row, 1] = 1
            elif "api" in lowered or "429" in lowered:
                output[row, 2] = 1
            else:
                output[row, 3] = 1
        return output

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]


def test_rank_fusion_rewards_agreement_and_preserves_first_ranks() -> None:
    fused = fuse_rankings([[5, 2, 7], [2, 5, 4]])
    assert fused[:2] == [5, 2] or fused[:2] == [2, 5]
    assert 4 in fused


def test_hybrid_search_returns_metadata_and_exact_429_source(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=PROJECT_ROOT / "data" / "knowledge_base",
        index_dir=tmp_path / "index",
        top_k=4,
        candidate_k=10,
    )
    embedder: Embedder = FakeEmbedder()
    searcher = HybridSearcher.build_or_load(settings, embedder, force_rebuild=True)
    hits = searcher.search(
        "What should I do after HTTP 429?",
        embedder.encode_query("api 429 retry"),
        4,
        10,
    )

    assert hits
    assert any(hit.chunk.document_id == "technical-api-limits" for hit in hits)
    assert all(hit.chunk.title and hit.chunk.chunk_id for hit in hits)


def test_index_reloads_without_reembedding(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=PROJECT_ROOT / "data" / "knowledge_base",
        index_dir=tmp_path / "index",
    )
    embedder = FakeEmbedder()
    first = HybridSearcher.build_or_load(settings, embedder, force_rebuild=True)

    class NoEncodeEmbedder(FakeEmbedder):
        def encode_documents(self, texts: list[str]) -> np.ndarray:
            raise AssertionError("A valid persisted index should not be re-embedded")

    loaded = HybridSearcher.build_or_load(settings, NoEncodeEmbedder())
    assert loaded.chunks == first.chunks
    assert loaded.vector_index.ntotal == len(first.chunks)


def test_rebuilds_index_after_source_change(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "policy.md").write_text(
        "# Policy\n\nRefund requests need a receipt.", encoding="utf-8"
    )
    settings = Settings(data_dir=source_dir, index_dir=tmp_path / "index")
    embedder = FakeEmbedder()
    first = HybridSearcher.build_or_load(settings, embedder)
    (source_dir / "policy.md").write_text(
        "# Policy\n\nRefund requests need invoice references.", encoding="utf-8"
    )
    second = HybridSearcher.build_or_load(settings, embedder)
    assert first.chunks[0].text != second.chunks[0].text
