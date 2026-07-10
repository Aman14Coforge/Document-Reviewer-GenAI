"""Model Garden embedding API client (HTTP)."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

EMBEDDING_URL = os.getenv("EMBEDDING_API_URL", "").strip()
API_KEY = os.getenv("MODEL_GARDEN_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
EXPECTED_DIM = int(os.getenv("EMBEDDING_DIM", "0") or "0")

_http_client: httpx.Client | None = None


def get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        if not API_KEY:
            raise RuntimeError(
                "MODEL_GARDEN_API_KEY is not set. Required for model_garden embeddings."
            )
        _http_client = httpx.Client(
            headers={
                "X-API-KEY": API_KEY,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0),
        )
    return _http_client


def _parse_embedding_response(data: object, texts: list[str]) -> list[list[float]]:
    if isinstance(data, dict) and "embeddings" in data:
        emb = data["embeddings"]
        if emb and isinstance(emb[0], (int, float)):
            if len(texts) > 1:
                return [embed_query(text) for text in texts]
            return [list(emb)]
        if emb and isinstance(emb[0], list):
            return [list(vector) for vector in emb]

    if isinstance(data, dict) and "data" in data:
        return [list(item["embedding"]) for item in data["data"]]

    if isinstance(data, list) and data and isinstance(data[0], list):
        return [list(vector) for vector in data]

    raise RuntimeError(f"Unexpected embedding API response format: {data!r}")


def _validate_vectors(vectors: list[list[float]], *, context: str) -> list[list[float]]:
    if not vectors:
        raise RuntimeError(f"Embedding API returned no vectors ({context}).")

    for index, vector in enumerate(vectors):
        if not vector:
            raise RuntimeError(f"Empty embedding vector at index {index} ({context}).")
        if EXPECTED_DIM and len(vector) != EXPECTED_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch at index {index}: "
                f"got {len(vector)}, expected {EXPECTED_DIM} ({context})."
            )
    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not EMBEDDING_URL:
        raise RuntimeError("EMBEDDING_API_URL is not set.")
    if not EMBED_MODEL:
        raise RuntimeError("EMBEDDING_MODEL is not set.")

    client = get_http_client()
    payload = {"model": EMBED_MODEL, "input": texts}
    response = client.post(EMBEDDING_URL, json=payload)
    response.raise_for_status()
    vectors = _parse_embedding_response(response.json(), texts)
    return _validate_vectors(vectors, context=f"batch_size={len(texts)}")


def embed_query(text: str) -> list[float]:
    vectors = embed_documents([text])
    return vectors[0]


def get_model_garden_embed_config() -> dict:
    return {
        "url": EMBEDDING_URL,
        "model": EMBED_MODEL,
        "expected_dim": EXPECTED_DIM or None,
    }


if __name__ == "__main__":
    print("Testing Model Garden embeddings...")
    print(get_model_garden_embed_config())
    texts = ["Aman Gupta AI project", "RAG pipeline"]
    vectors = embed_documents(texts)
    print(f"Vectors: {len(vectors)}, dim: {len(vectors[0])}")
    print(f"Sample: {vectors[0][:10]}")
