"""ChromaDB rule store with local or Model Garden embeddings.

Rule documents embed keywords, typical phrases, and criteria for better RAG retrieval.
Retrieval reranks vector hits by section overlap and keyword matches (see chunk_rag config).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.logging_config import get_logger

logger = get_logger("rag.store")

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


def _join_list(values: list | None) -> str:
    if not values:
        return ""
    return ", ".join(str(value).strip() for value in values if str(value).strip())


def build_rule_document(rule: dict) -> str:
    """Build rich text for embedding — keywords/phrases improve vector + keyword rerank."""
    lines = [
        f"Rule ID: {rule.get('rule_id', '')}",
        f"Title: {rule.get('title', '')}",
        f"Category: {rule.get('category', '')}",
        f"Rule type: {rule.get('rule_type', '')}",
    ]

    sections = rule.get("applies_to_sections", [])
    if sections:
        lines.append(f"Applies to sections: {_join_list(sections)}")

    keywords = rule.get("keywords", [])
    if keywords:
        lines.append(f"Keywords: {_join_list(keywords)}")

    phrases = rule.get("typical_phrases", [])
    if phrases:
        lines.append(f"Typical phrases: {_join_list(phrases)}")

    lines.append(f"Criteria: {rule.get('verifiable_criteria', '')}")

    recommendation = rule.get("recommendation", "")
    if recommendation:
        lines.append(f"Recommendation: {recommendation}")

    intent = rule.get("validation_intent", "")
    if intent:
        lines.append(f"Validation intent: {intent}")

    return "\n".join(lines)


def build_rule_metadata(rule: dict) -> dict:
    return {
        "rule_id": rule["rule_id"],
        "title": rule.get("title", ""),
        "category": rule.get("category", ""),
        "severity": rule.get("severity", ""),
        "rule_type": rule.get("rule_type", "semantic"),
        "validation_intent": rule.get("validation_intent", "") or "",
        "keywords": _join_list(rule.get("keywords", [])),
        "typical_phrases": _join_list(rule.get("typical_phrases", [])),
        "applies_to_sections": _join_list(rule.get("applies_to_sections", [])),
        "recommendation": rule.get("recommendation", "") or "",
    }


def _metadata_terms(metadata: dict, *keys: str) -> list[str]:
    terms: list[str] = []
    for key in keys:
        raw = metadata.get(key, "")
        if not raw:
            continue
        terms.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return terms


def keyword_overlap_score(text: str, metadata: dict) -> int:
    if not text.strip():
        return 0
    upper = text.upper()
    score = 0
    for term in _metadata_terms(metadata, "keywords", "typical_phrases"):
        if term.upper() in upper:
            score += 1
    return score


def section_overlap_score(section_type: str | None, metadata: dict) -> int:
    if not section_type:
        return 0
    section = section_type.strip().lower()
    sections = [part.strip().lower() for part in metadata.get("applies_to_sections", "").split(",") if part.strip()]
    if not sections:
        return 0
    if section in sections or "full" in sections:
        return 2
    return 0


def load_rag_retrieval_config() -> dict:
    config = load_compliance_config()
    return config.get(
        "chunk_rag",
        {
            "keyword_rerank": True,
            "keyword_filter_min_hits": 0,
            "retrieve_candidate_multiplier": 3,
            "rag_rule_types": ["semantic"],
        },
    )


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
    metadatas = [build_rule_metadata(rule) for rule in rules]

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

    logger.info(
        "Embedded %s rules into collection=%s provider=%s rebuild=%s",
        len(rules),
        name,
        provider,
        rebuild,
    )
    return result


def retrieve_rule_ids_for_text(
    text: str,
    top_k: int = 5,
    *,
    section_type: str | None = None,
) -> list[str]:
    if not text.strip():
        return []

    collection = get_collection(create=True)
    if collection.count() == 0:
        raise RuntimeError("Rule vector store is empty. Run scripts/embed_rules.py first.")

    rag_config = load_rag_retrieval_config()
    keyword_rerank = rag_config.get("keyword_rerank", True)
    min_keyword_hits = int(rag_config.get("keyword_filter_min_hits", 0))
    candidate_multiplier = max(1, int(rag_config.get("retrieve_candidate_multiplier", 3)))
    allowed_rule_types = {
        str(rule_type).strip().lower()
        for rule_type in rag_config.get("rag_rule_types", ["semantic"])
        if str(rule_type).strip()
    }

    fetch_k = min(top_k, collection.count())
    if keyword_rerank or min_keyword_hits > 0 or allowed_rule_types:
        fetch_k = min(max(top_k, top_k * candidate_multiplier), collection.count())

    result = collection.query(
        query_texts=[text],
        n_results=fetch_k,
        include=["metadatas", "distances"],
    )
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []

    ranked: list[tuple[str, dict, float, int, int]] = []
    for index, rule_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}
        rule_type = str(metadata.get("rule_type", "semantic")).lower()
        if allowed_rule_types and rule_type not in allowed_rule_types:
            continue

        keyword_score = keyword_overlap_score(text, metadata)
        if min_keyword_hits > 0 and keyword_score < min_keyword_hits:
            continue

        section_score = section_overlap_score(section_type, metadata)
        distance = float(distances[index]) if index < len(distances) else 1.0
        ranked.append((rule_id, metadata, distance, keyword_score, section_score))

    if keyword_rerank and ranked:
        ranked.sort(
            key=lambda item: (-item[4], -item[3], item[2]),
        )

    selected = [rule_id for rule_id, _, _, _, _ in ranked[:top_k]]
    logger.debug(
        "RAG retrieve top_k=%s section=%s candidates=%s selected=%s",
        top_k,
        section_type,
        len(ids),
        selected,
    )
    return selected


def resolve_rules_by_ids(rule_ids: list[str], all_rules: list[dict]) -> list[dict]:
    rule_map = {rule["rule_id"]: rule for rule in all_rules}
    resolved = []
    seen = set()
    for rule_id in rule_ids:
        if rule_id in rule_map and rule_id not in seen:
            resolved.append(rule_map[rule_id])
            seen.add(rule_id)
    return resolved
