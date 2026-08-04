"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k + use_hyde (mặc định False)
    - Output: danh sách chunks có score, sorted descending
    - Sử dụng OpenAI embedding model: text-embedding-3-small
    - Hỗ trợ HyDE (Hypothetical Document Embeddings) bằng gpt-4o-mini khi use_hyde=True
"""

import os
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Re-use configuration from Task 4
try:
    from .task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        get_embedding_model,
    )
except ImportError:
    CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
    COLLECTION_NAME = "university_services_docs"
    EMBEDDING_MODEL = "text-embedding-3-small"

    def get_embedding_model():
        raise ImportError("Task 4 embedding backend is unavailable.")

_openai_client = None
_collection = None


def get_openai_client() -> OpenAI:
    """Lazy load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            raise ValueError(
                "Không tìm thấy OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong môi trường hoặc file .env."
            )
        _openai_client = OpenAI(api_key=api_key, base_url=base_url)
    return _openai_client


def get_collection() -> chromadb.Collection:
    """Lazy load ChromaDB collection. Báo lỗi nếu collection chưa tồn tại."""
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Thư mục ChromaDB không tồn tại tại '{CHROMA_DIR}'. Vui lòng chạy Task 4 để index dữ liệu trước."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            _collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' chưa tồn tại trong ChromaDB. Vui lòng chạy Task 4 để index dữ liệu trước."
            ) from e
    return _collection


def generate_hypothetical_document(query: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Sinh một câu trả lời giả định (hypothetical document) cho câu truy vấn bằng LLM (gpt-4o-mini).
    """
    client = get_openai_client()
    target_model = model_name
    if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY") and not target_model.startswith("openai/"):
        target_model = f"openai/{target_model}"

    prompt = f"Please write a passage that directly answers the following question:\nQuestion: {query}\nPassage:"

    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates hypothetical answer passages to improve semantic search retrieval.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def get_query_embedding(text: str) -> list[float]:
    """Embed query with the exact same backend used by Task 4 indexing."""
    return get_embedding_model().embed_query(text)


def semantic_search(
    query: str, top_k: int = 10, use_hyde: bool = False
) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity với model text-embedding-3-small.
    Hỗ trợ HyDE (Hypothetical Document Embeddings) nếu use_hyde=True.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: Nếu True, sinh câu trả lời giả định bằng gpt-4o-mini trước khi embed

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or top_k <= 0:
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    if use_hyde:
        search_text = generate_hypothetical_document(query, model_name="gpt-4o-mini")
    else:
        search_text = query

    query_vector = get_query_embedding(search_text)

    n_results = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            output.append({
                "content": doc,
                "score": round(score, 4),
                "metadata": meta if meta is not None else {}
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
