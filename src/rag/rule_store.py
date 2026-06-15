"""
ChromaDB rule store with local or Model Garden embeddings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "rules" / "rules.json"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
CONFIG_PATH = PROJECT_ROOT / "config" / "compliance.json"

DEFAULT_LOCAL_MODEL_DIRS = (
    PROJECT_ROOT / "model",
    PROJECT_ROOT / "models" / "all-MiniLM-L6-v2",
)

load_dotenv(PROJECT_ROOT / ".env")


def load_compliance_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def get_embedding_provider() -> str:
    env_provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    config = load_compliance_config()
    return str(config.get("embedding_provider", "local")).strip().lower()


def get_embedding_model_name() -> str:
    config = load_compliance_config()

    if get_embedding_provider() == "model_garden":
        from src.embedding_client import EMBED_MODEL
        return EMBED_MODEL or config.get("embedding_model", "model_garden")

    return config.get("embedding_model", "all-MiniLM-L6-v2")


def _is_valid_local_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def resolve_embedding_model_path(raw_path: str) -> Path:
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_embedding_model_path() -> Path | None:
    if get_embedding_provider() == "model_garden":
        return None

    env_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if env_path:
        path = resolve_embedding_model_path(env_path)
        if _is_valid_local_model_dir(path):
            return path

        raise FileNotFoundError(
            f"EMBEDDING_MODEL_PATH folder invalid (need config.json + weights): {env_path}"
        )

    config = load_compliance_config()
    config_path = config.get("embedding_model_path")

    if config_path:
        path = resolve_embedding_model_path(str(config_path))
        if _is_valid_local_model_dir(path):
            return path

    for candidate in DEFAULT_LOCAL_MODEL_DIRS:
        if _is_valid_local_model_dir(candidate):
            return candidate

    return None


def get_embedding_model_source() -> str:
    if get_embedding_provider() == "model_garden":
        from src.embedding_client import EMBED_MODEL
        return EMBED_MODEL or "model_garden"

    local_path = get_embedding_model_path()
    if local_path is not None:
        return str(local_path)

    return get_embedding_model_name()


def get_collection_name() -> str:
    config = load_compliance_config()
    return config.get("chroma_collection", "gdp_rules")


def build_rule_document(rule: dict) -> str:
    return "\n".join([
        f"Rule ID: {rule.get('rule_id', '')}",
        f"Title: {rule.get('title', '')}",
        f"Category: {rule.get('category', '')}",
        f"Criteria: {rule.get('verifiable_criteria', '')}",
    ])


def get_embedding_function():
    provider = get_embedding_provider()

    if provider == "model_garden":
        from src.rag.model_garden_embedding import ModelGardenEmbeddingFunction
        return ModelGardenEmbeddingFunction()

    if provider != "local":
        raise ValueError(
            f"Unknown embedding_provider '{provider}'. Use 'local' or 'model_garden'."
        )

    from chromadb.utils import embedding_functions

    if get_embedding_model_path() is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=get_embedding_model_source()
    )


def get_client():
    import chromadb

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection(*, create: bool = True):
    client = get_client()
    name = get_collection_name()
    embedding_function = get_embedding_function()

    if create:
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    return client.get_collection(name=name, embedding_function=embedding_function)


def load_rules(rules_path: Path | None = None) -> list[dict]:
    path = rules_path or DEFAULT_RULES_PATH
    data = json.loads(path.read_text(encoding="utf-8"))

    rules = data.get("rules", data)
    if not isinstance(rules, list):
        raise ValueError("Rules file must contain a top-level 'rules' array.")

    return rules


def embed_rules(rules_path: Path | None = None, *, rebuild: bool = False) -> dict:
    rules = load_rules(rules_path)
    client = get_client()
    name = get_collection_name()
    provider = get_embedding_provider()

    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    collection = get_collection(create=True)

    ids = [rule["rule_id"] for rule in rules]
    documents = [build_rule_document(rule) for rule in rules]
    metadatas = [
        {
            "rule_id": rule["rule_id"],
            "title": rule.get("title", ""),
            "category": rule.get("category", ""),
            "severity": rule.get("severity", ""),
        }
        for rule in rules
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    embedding_function = get_embedding_function()
    sample_vectors = embedding_function([documents[0]])
    local_path = get_embedding_model_path()

    result = {
        "collection": name,
        "chroma_path": str(CHROMA_PATH),
        "embedding_provider": provider,
        "embedding_model": get_embedding_model_name(),
        "embedding_model_source": get_embedding_model_source(),
        "embedding_model_path": str(local_path) if local_path else None,
        "embedding_dimension": len(sample_vectors[0]),
        "rule_count": len(rules),
        "rule_ids": ids,
        "rebuild": rebuild,
    }

    if provider == "model_garden":
        from src.embedding_client import get_model_garden_embed_config
        result["model_garden_config"] = get_model_garden_embed_config()

    return result


def retrieve_rule_ids_for_text(text: str, top_k: int = 5) -> list[str]:
    if not text.strip():
        return []

    collection = get_collection(create=True)

    if collection.count() == 0:
        raise RuntimeError(
            "Rule vector store is empty. Run scripts/embed_rules.py first."
        )

    result = collection.query(
        query_texts=[text],
        n_results=min(top_k, collection.count())
    )

    ids = result.get("ids", [[]])[0]
    return list(ids)


def resolve_rules_by_ids(rule_ids: list[str], all_rules: list[dict]) -> list[dict]:
    rule_map = {rule["rule_id"]: rule for rule in all_rules}

    resolved = []
    seen = set()

    for rule_id in rule_ids:
        if rule_id in rule_map and rule_id not in seen:
            resolved.append(rule_map[rule_id])
            seen.add(rule_id)

    return resolved