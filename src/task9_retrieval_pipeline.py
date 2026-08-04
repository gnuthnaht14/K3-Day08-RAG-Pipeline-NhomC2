"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Calibrate threshold này bằng cách tự đo điểm cosine của semantic_search
# cho câu hỏi liên quan vs câu hỏi lạc đề (xem ghi chú ở trên) — ĐỪNG copy nguyên
# giá trị mẫu, mỗi corpus/embedding model sẽ cho khoảng điểm khác nhau.
SCORE_THRESHOLD = 0.3   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_pageindex: bool = True,
    use_dense: bool = True,
    use_bm25: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không
        use_pageindex: Có cho phép gọi PageIndex fallback hay không
        use_dense: Có chạy semantic/vector search hay không
        use_bm25: Có chạy lexical/BM25 search hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid', 'hybrid-no-rerank', 'dense', 'bm25' hoặc 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    if not use_dense and not use_bm25:
        return []

    # Fetch a wider candidate pool so RRF has enough documents from enabled
    # retrievers. Dense and sparse scores deliberately remain independent:
    # cosine similarity and BM25 scores are not directly comparable.
    candidate_k = top_k * 3
    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    if use_dense:
        try:
            dense_results = semantic_search(query, top_k=candidate_k) or []
        except (FileNotFoundError, ValueError, RuntimeError):
            # Before Task 4 has created Chroma, BM25 can still serve the query.
            dense_results = []

    if use_bm25:
        try:
            sparse_results = lexical_search(query, top_k=candidate_k) or []
        except (FileNotFoundError, ValueError, RuntimeError):
            sparse_results = []

    # Fallback confidence MUST use the original dense cosine score. RRF is a
    # rank-fusion score, not a measure of semantic confidence.
    best_dense_score = (
        float(dense_results[0].get("score", 0.0)) if dense_results else 0.0
    )
    if use_pageindex and use_dense and best_dense_score < score_threshold:
        try:
            fallback_results = pageindex_search(query, top_k=top_k) or []
        except Exception:
            # PageIndex is optional in the UI demo. If it has not been
            # configured, has stale document IDs, or is out of quota, keep the
            # available dense/BM25 results instead of failing the whole query.
            fallback_results = []
        if fallback_results:
            return _normalize_results(fallback_results, "pageindex", top_k)

    ranked_lists = [results for results in (dense_results, sparse_results) if results]

    if use_reranking and len(ranked_lists) > 1:
        # RRF consumes the two original ranked lists. Calling
        # rerank(..., method="rrf") would be incorrect because rerank receives
        # one flat candidate list in the agreed Task 7 contract.
        final_results = rerank_rrf(
            ranked_lists,
            top_k=top_k,
        )
        result_source = "hybrid"
    elif use_dense and use_bm25:
        # Comparison mode: preserve each retriever's native order and avoid
        # applying any score normalization when reranking is disabled.
        final_results = _merge_without_reranking(
            [dense_results, sparse_results], top_k=top_k
        )
        # Keep the historical ``hybrid`` label when one enabled retriever is
        # empty (for example, an unavailable vector index); use the explicit
        # comparison label only when both sources contributed results.
        result_source = (
            "hybrid-no-rerank"
            if dense_results and sparse_results
            else "hybrid"
        )
    elif use_dense:
        final_results = dense_results[:top_k]
        result_source = "dense"
    else:
        final_results = sparse_results[:top_k]
        result_source = "bm25"

    return _normalize_results(
        final_results,
        result_source,
        top_k,
        dense_results=dense_results,
        sparse_results=sparse_results,
    )


def _merge_without_reranking(
    ranked_lists: list[list[dict]], top_k: int
) -> list[dict]:
    """Concatenate enabled ranked lists while preserving native rank order."""
    merged: list[dict] = []
    seen: set[str] = set()
    for ranked_list in ranked_lists:
        for result in ranked_list:
            content = str(result.get("content") or "")
            if not content or content in seen:
                continue
            seen.add(content)
            merged.append(result)
            if len(merged) >= top_k:
                return merged
    return merged


def _normalize_results(
    results: list[dict],
    source: str,
    top_k: int,
    *,
    dense_results: list[dict] | None = None,
    sparse_results: list[dict] | None = None,
) -> list[dict]:
    """Return the stable Task 9 schema without mutating upstream results."""
    def unique_rank_map(results: list[dict] | None) -> dict[str, tuple[int, dict]]:
        mapping: dict[str, tuple[int, dict]] = {}
        seen: set[str] = set()
        unique_rank = 0
        for result in results or []:
            content = str(result.get("content") or "")
            if not content or content in seen:
                continue
            seen.add(content)
            unique_rank += 1
            mapping[content] = (unique_rank, result)
        return mapping

    dense_by_content = unique_rank_map(dense_results)
    sparse_by_content = unique_rank_map(sparse_results)
    normalized = []
    for result in results[:top_k]:
        item = result.copy()
        item["metadata"] = item.get("metadata") or {}
        item["source"] = source

        content = str(item.get("content") or "")
        component_scores = {}
        component_ranks = {}
        for name, lookup in (
            ("dense", dense_by_content),
            ("bm25", sparse_by_content),
        ):
            if content in lookup:
                rank, component = lookup[content]
                component_ranks[name] = rank
                component_scores[name] = float(component.get("score", 0.0))
        if component_scores:
            item["component_scores"] = component_scores
            item["component_ranks"] = component_ranks

        if source == "hybrid" and len(component_scores) > 1:
            item["score_type"] = "rrf"
        elif source == "dense" or (
            source == "hybrid" and "dense" in component_scores
        ):
            item["score_type"] = "cosine"
        elif source == "bm25":
            item["score_type"] = "bm25"
        elif source == "pageindex":
            item["score_type"] = "pageindex"
        else:
            item["score_type"] = "native"
        normalized.append(item)
    return normalized


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
