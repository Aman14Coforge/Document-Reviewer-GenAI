"""Chroma embedding function for Model Garden API."""

from __future__ import annotations

from src.embedding_client import EMBED_MODEL, EXPECTED_DIM, embed_documents


class ModelGardenEmbeddingFunction:
    """Chroma-compatible wrapper around embedding_client.embed_documents."""

    def name(self) -> str:
        dim = EXPECTED_DIM or "auto"
        model = EMBED_MODEL or "unknown"
        return f"model_garden_{model}_{dim}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return embed_documents(input)