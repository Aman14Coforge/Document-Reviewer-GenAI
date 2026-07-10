"""Chroma embedding function for Model Garden API."""

from __future__ import annotations

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from src.embedding_client import EMBED_MODEL, EXPECTED_DIM
from src.embedding_client import embed_documents as api_embed_documents
from src.embedding_client import embed_query as api_embed_query


class ModelGardenEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma wrapper around Model Garden embedding API."""

    @staticmethod
    def name() -> str:
        dim = EXPECTED_DIM or "auto"
        model = EMBED_MODEL or "unknown"
        return f"model_garden_{model}_{dim}"

    def __call__(self, input: Documents) -> Embeddings:
        return api_embed_documents(list(input))

    def embed_query(self, input: Documents) -> Embeddings:
        """Embed search queries — required by Chroma on collection.query()."""
        if len(input) == 1:
            return [api_embed_query(input[0])]
        return api_embed_documents(list(input))
