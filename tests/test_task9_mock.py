import sys
import types
from unittest.mock import Mock

# Stub dependency modules trước khi import Task 9.
dense_module = types.ModuleType("src.task5_semantic_search")
sparse_module = types.ModuleType("src.task6_lexical_search")
rerank_module = types.ModuleType("src.task7_reranking")
pageindex_module = types.ModuleType("src.task8_pageindex_vectorless")

dense_module.semantic_search = Mock()
sparse_module.lexical_search = Mock()
rerank_module.rerank_rrf = Mock()
pageindex_module.pageindex_search = Mock()

sys.modules[dense_module.__name__] = dense_module
sys.modules[sparse_module.__name__] = sparse_module
sys.modules[rerank_module.__name__] = rerank_module
sys.modules[pageindex_module.__name__] = pageindex_module

import src.task9_retrieval_pipeline as pipeline


def setup_function():
    pipeline.semantic_search.reset_mock()
    pipeline.lexical_search.reset_mock()
    pipeline.rerank_rrf.reset_mock()
    pipeline.pageindex_search.reset_mock()


def test_hybrid_retrieval():
    dense = [
        {
            "content": "Tuition fee information",
            "score": 0.82,
            "metadata": {"id": "dense-1"},
        }
    ]
    sparse = [
        {
            "content": "Tuition payment policy",
            "score": 7.2,
            "metadata": {"id": "sparse-1"},
        }
    ]
    fused = [
        {
            "content": "Tuition fee information",
            "score": 0.032,
            "metadata": {"id": "dense-1"},
        }
    ]

    pipeline.semantic_search.return_value = dense
    pipeline.lexical_search.return_value = sparse
    pipeline.rerank_rrf.return_value = fused

    results = pipeline.retrieve(
        "tuition fee",
        top_k=2,
        score_threshold=0.48,
    )

    pipeline.semantic_search.assert_called_once_with(
        "tuition fee", top_k=6
    )
    pipeline.lexical_search.assert_called_once_with(
        "tuition fee", top_k=6
    )
    pipeline.rerank_rrf.assert_called_once_with(
        [dense, sparse], top_k=2
    )
    pipeline.pageindex_search.assert_not_called()

    assert len(results) == 1
    assert results[0]["source"] == "hybrid"


def test_low_dense_score_uses_pageindex():
    pipeline.semantic_search.return_value = [
        {
            "content": "Weak result",
            "score": 0.15,
            "metadata": {},
        }
    ]
    pipeline.lexical_search.return_value = []
    pipeline.pageindex_search.return_value = [
        {
            "content": "PageIndex result",
            "score": 1.0,
            "metadata": {"section": "Tuition"},
        }
    ]

    results = pipeline.retrieve(
        "unknown question",
        top_k=3,
        score_threshold=0.48,
    )

    pipeline.pageindex_search.assert_called_once_with(
        "unknown question", top_k=3
    )
    pipeline.rerank_rrf.assert_not_called()

    assert results[0]["source"] == "pageindex"


def test_empty_dense_uses_pageindex():
    pipeline.semantic_search.return_value = []
    pipeline.lexical_search.return_value = []
    pipeline.pageindex_search.return_value = [
        {
            "content": "Fallback result",
            "score": 1.0,
            "metadata": {},
        }
    ]

    results = pipeline.retrieve("nonsense", top_k=2)

    assert results[0]["source"] == "pageindex"


def test_dense_only_respects_top_k():
    pipeline.semantic_search.return_value = [
        {"content": "A", "score": 0.9, "metadata": {}},
        {"content": "B", "score": 0.8, "metadata": {}},
    ]
    pipeline.lexical_search.return_value = []

    results = pipeline.retrieve(
        "tuition",
        top_k=1,
        use_reranking=False,
    )

    assert len(results) == 1
    assert results[0]["content"] == "A"
    assert results[0]["source"] == "hybrid"
    pipeline.rerank_rrf.assert_not_called()


def test_zero_top_k_short_circuits():
    results = pipeline.retrieve("anything", top_k=0)

    assert results == []
    pipeline.semantic_search.assert_not_called()
    pipeline.lexical_search.assert_not_called()