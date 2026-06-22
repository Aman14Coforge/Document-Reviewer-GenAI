from sentence_transformers import CrossEncoder

# Load once (VERY important for performance)
_model = CrossEncoder("BAAI/bge-reranker-base")


def rerank_rules(chunk_text: str, rule_ids: list[str], rule_map: dict, top_k: int = 5):
    """
    Rerank rules using cross-encoder
    Input:
        chunk_text → text of chunk
        rule_ids → retrieved rule IDs (top 20)
        rule_map → rule_id → rule data
    Output:
        top_k reranked rule IDs
    """

    # Prepare pairs
    pairs = []
    valid_rule_ids = []

    for rid in rule_ids:
        if rid in rule_map:
            rule_text = rule_map[rid].get("description", "")
            pairs.append((chunk_text, rule_text))
            valid_rule_ids.append(rid)

    if not pairs:
        return []

    # Get scores
    scores = _model.predict(pairs)

    # Combine and sort
    scored = list(zip(valid_rule_ids, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top_k
    return [rid for rid, _ in scored[:top_k]]