"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "university_services_docs"

# Corpus toàn cục: List of {'content': str, 'metadata': dict}
CORPUS: list[dict] = []
_bm25_index: BM25Okapi | None = None


def tokenize(text: str) -> list[str]:
    """Tách từ: Viết thường, loại bỏ dấu câu và split theo khoảng trắng."""
    if not text:
        return []
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return cleaned.split()


def load_corpus() -> list[dict]:
    """
    Load corpus từ ChromaDB (ưu tiên) hoặc từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    # 1. Thử load từ ChromaDB nếu đã được index ở Task 4
    if CHROMA_DIR.exists():
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(name=COLLECTION_NAME)
            data = collection.get(include=["documents", "metadatas"])
            if data and data.get("documents"):
                docs = data["documents"]
                metas = data.get("metadatas") or [{}] * len(docs)
                corpus = [
                    {"content": d, "metadata": m if m is not None else {}}
                    for d, m in zip(docs, metas)
                ]
                if corpus:
                    return corpus
        except Exception:
            pass

    # 2. Fallback: Read và chunk từ data/standardized/
    corpus = []
    if STANDARDIZED_DIR.exists():
        md_files = list(STANDARDIZED_DIR.rglob("*.md"))
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            doc_type = "legal" if "legal" in str(md_file) else "news"
            metadata = {"source": md_file.name, "type": doc_type}

            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                )
                splits = splitter.split_text(content)
                for i, split_text in enumerate(splits):
                    corpus.append(
                        {
                            "content": split_text,
                            "metadata": {**metadata, "chunk_index": i},
                        }
                    )
            except ImportError:
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for i, p in enumerate(paragraphs):
                    corpus.append(
                        {
                            "content": p,
                            "metadata": {**metadata, "chunk_index": i},
                        }
                    )

    return corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi | None:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def get_bm25_state():
    """Lazy load CORPUS và BM25 Index."""
    global CORPUS, _bm25_index
    if not CORPUS:
        CORPUS = load_corpus()
        _bm25_index = build_bm25_index(CORPUS)
    elif _bm25_index is None and CORPUS:
        _bm25_index = build_bm25_index(CORPUS)
    return CORPUS, _bm25_index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or top_k <= 0:
        return []

    corpus, bm25_index = get_bm25_state()
    if not corpus or bm25_index is None:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25_index.get_scores(tokenized_query)

    results = []
    for idx, score in enumerate(scores):
        if score > 0:
            results.append(
                {
                    "content": corpus[idx]["content"],
                    "score": float(round(score, 4)),
                    "metadata": corpus[idx].get("metadata", {}),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

