"""Local sentence-transformer embeddings with a small injectable interface."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """CPU-first embedder; model files download on the first build only."""

    def __init__(self, model_name: str, revision: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            device="cpu",
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None or dimension < 1:
            raise ValueError("Embedding model must expose a fixed vector dimension")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(texts), self.dimension):
            raise ValueError("Embedding model returned an unexpected matrix shape")
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding model returned non-finite values")
        return matrix

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]
