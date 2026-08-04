"""
Task 6 — Lexical Search Module (BM25).

CƠ CHẾ VÀ LÝ DO CHỌN BM25 TRONG DỰ ÁN NÀY:
--------------------------------------------------------------------------------
1. BM25 vs TF-IDF:
   - BM25 là bản cải tiến vượt trội của TF-IDF. TF-IDF bị hạn chế khi một từ xuất
     hiện quá nhiều trong tài liệu dài (điểm số tăng tuyến tính gây méo lệch).
   - BM25 khắc phục bằng 2 tham số:
     + Term Saturation (k1=1.5): Điểm số bão hòa dần khi tần suất từ đạt ngưỡng.
     + Document Length Normalization (b=0.75): Phạt bớt tài liệu quá dài để tránh
       ưu tiên không hợp lý.

2. BM25 vs Elasticsearch:
   - Elasticsearch là hệ thống full-text search chuẩn công nghiệp (Production),
     tuy nhiên rất nặng vì đòi hỏi chạy Server/Docker riêng biệt.
   - Với quy mô bài lab/dự án nhỏ (vài trăm chunks), rank-bm25 chạy hoàn toàn
     In-Memory trong Python, không tốn tài nguyên và dễ dàng triển khai.

3. BM25 vs Weaviate (BM25 Built-in):
   - Weaviate hỗ trợ sẵn cả Vector & Lexical search trong cùng 1 database. Tuy nhiên,
     do dự án đã chọn ChromaDB làm Vector Store ở Task 4 & 5, việc tích hợp BM25
     thuần qua rank-bm25 giúp duy trì tính độc lập, đơn giản và không phải đổi DB.

TỪ KHÓA & TÁCH TỪ (TOKENIZATION):
   - Sử dụng thư viện `underthesea` (word_tokenize) để tách từ ghép Tiếng Việt
     (ví dụ: "học phí" -> "học_phí"), giúp khớp từ khóa chính xác theo ngữ nghĩa.
"""

import re
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "university_services_docs"

# Corpus toàn cục: List of {'content': str, 'metadata': dict}
CORPUS: list[dict] = []
_bm25_index: BM25Okapi | None = None


def tokenize(text: str) -> list[str]:
    """
    Tách từ cho BM25:
    - Sử dụng underthesea để tách từ ghép tiếng Việt (format='text' -> ví dụ: 'học_phí').
    - Fallback: loại bỏ dấu câu, chuyển chữ thường và split theo khoảng trắng.
    """
    if not text:
        return []
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    if word_tokenize is not None:
        try:
            tokenized_str = word_tokenize(cleaned, format="text")
            return tokenized_str.split()
        except Exception:
            pass
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
