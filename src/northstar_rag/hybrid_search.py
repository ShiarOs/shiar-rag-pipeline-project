"""Persistent exact-vector and BM25 search with reciprocal-rank fusion."""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .documents import chunk_documents, load_documents
from .embeddings import Embedder
from .models import Chunk, SearchHit
from .settings import Settings

INDEX_FILENAME = "vectors.faiss"
METADATA_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1


def tokenize(text: str) -> list[str]:
    """Keep alphanumeric terms, including exact numeric identifiers."""

    return re.findall(r"[a-z0-9]+", text.casefold())


def source_fingerprint(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(data_dir.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def fuse_rankings(rankings: list[list[int]], constant: int = 60) -> list[int]:
    scores = fusion_scores(rankings, constant)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def fusion_scores(rankings: list[list[int]], constant: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0.0) + 1.0 / (constant + rank)
    return scores


class HybridSearcher:
    """Search one immutable chunk snapshot through dense and lexical rankings."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        vector_index,
        rrf_k: int,
    ) -> None:
        if not chunks or vectors.shape[0] != len(chunks):
            raise ValueError(
                "Chunk metadata and vector rows must be nonempty and aligned"
            )
        self.chunks = chunks
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.vector_index = vector_index
        self.rrf_k = rrf_k
        tokenized_corpus = [
            tokenize(f"{chunk.title} {chunk.section} {chunk.text}") for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def build_or_load(
        cls, settings: Settings, embedder: Embedder, force_rebuild: bool = False
    ) -> "HybridSearcher":
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = source_fingerprint(settings.data_dir)
        if not force_rebuild and cls._cache_matches(settings, embedder, fingerprint):
            return cls._load(settings, embedder, fingerprint)

        documents = load_documents(settings.data_dir)
        chunks = chunk_documents(
            documents,
            max_words=settings.max_chunk_words,
            overlap_words=settings.overlap_words,
        )
        embedding_texts = [
            f"{chunk.title}\n{chunk.section}\n{chunk.text}" for chunk in chunks
        ]
        vectors = embedder.encode_documents(embedding_texts)
        if vectors.shape != (len(chunks), embedder.dimension):
            raise ValueError("Expected one correctly sized vector for every chunk")
        index = faiss.IndexFlatIP(embedder.dimension)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        cls._save(settings, chunks, vectors, index, embedder, fingerprint)
        return cls(chunks, vectors, index, settings.rrf_k)

    @staticmethod
    def _cache_matches(
        settings: Settings, embedder: Embedder, fingerprint: str
    ) -> bool:
        manifest_path = settings.index_dir / MANIFEST_FILENAME
        vector_path = settings.index_dir / INDEX_FILENAME
        metadata_path = settings.index_dir / METADATA_FILENAME
        if not (
            manifest_path.is_file()
            and vector_path.is_file()
            and metadata_path.is_file()
        ):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return all(
            (
                manifest.get("schema_version") == SCHEMA_VERSION,
                manifest.get("source_fingerprint") == fingerprint,
                manifest.get("embedding_model") == settings.embedding_model,
                manifest.get("embedding_revision") == settings.embedding_revision,
                manifest.get("dimension") == embedder.dimension,
                manifest.get("max_chunk_words") == settings.max_chunk_words,
                manifest.get("overlap_words") == settings.overlap_words,
            )
        )

    @classmethod
    def _load(
        cls, settings: Settings, embedder: Embedder, fingerprint: str
    ) -> "HybridSearcher":
        manifest = json.loads(
            (settings.index_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        chunks = [
            Chunk(**json.loads(line))
            for line in (settings.index_dir / METADATA_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        index = faiss.read_index(str(settings.index_dir / INDEX_FILENAME))
        if index.ntotal != len(chunks) or index.d != embedder.dimension:
            raise ValueError("Cached vector index and metadata are out of sync")
        if manifest.get("source_fingerprint") != fingerprint:
            raise ValueError("Cached index source fingerprint does not match")
        vectors = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
        return cls(chunks, vectors, index, settings.rrf_k)

    @staticmethod
    def _save(
        settings: Settings,
        chunks: list[Chunk],
        vectors: np.ndarray,
        index,
        embedder: Embedder,
        fingerprint: str,
    ) -> None:
        directory = settings.index_dir
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary_name:
            temporary = Path(temporary_name)
            vector_tmp = temporary / INDEX_FILENAME
            metadata_tmp = temporary / METADATA_FILENAME
            manifest_tmp = temporary / MANIFEST_FILENAME
            faiss.write_index(index, str(vector_tmp))
            with metadata_tmp.open("w", encoding="utf-8", newline="\n") as output:
                for chunk in chunks:
                    output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "embedding_model": settings.embedding_model,
                "embedding_revision": settings.embedding_revision,
                "dimension": embedder.dimension,
                "index_type": "IndexFlatIP",
                "distance": "inner_product_on_normalized_vectors",
                "max_chunk_words": settings.max_chunk_words,
                "overlap_words": settings.overlap_words,
                "chunk_count": len(chunks),
                "source_fingerprint": fingerprint,
            }
            manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(vector_tmp, directory / INDEX_FILENAME)
            os.replace(metadata_tmp, directory / METADATA_FILENAME)
            os.replace(manifest_tmp, directory / MANIFEST_FILENAME)

    def search(
        self, query: str, query_vector: np.ndarray, top_k: int, candidate_k: int
    ) -> list[SearchHit]:
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Require candidate_k >= top_k >= 1")
        limit = min(candidate_k, len(self.chunks))
        vector = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.vector_index.d:
            raise ValueError("Query and indexed vectors have different dimensions")

        dense_values, dense_positions = self.vector_index.search(vector, limit)
        dense_ranking = [int(i) for i in dense_positions[0] if i >= 0]
        dense_scores = {
            int(i): float(score)
            for score, i in zip(dense_values[0], dense_positions[0], strict=True)
            if i >= 0
        }

        lexical_values = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        lexical_ranking = sorted(
            range(len(self.chunks)),
            key=lambda i: (lexical_values[i], -i),
            reverse=True,
        )[:limit]
        lexical_max = float(lexical_values.max(initial=0.0))
        lexical_scores = {
            i: float(lexical_values[i] / lexical_max) if lexical_max > 0 else 0.0
            for i in lexical_ranking
        }

        fused_scores = fusion_scores([dense_ranking, lexical_ranking], self.rrf_k)
        fused = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
        hits = []
        for index in fused[:top_k]:
            hits.append(
                SearchHit(
                    chunk=self.chunks[index],
                    dense_score=dense_scores.get(index, 0.0),
                    lexical_score=lexical_scores.get(index, 0.0),
                    fused_score=fused_scores[index],
                )
            )
        return hits
