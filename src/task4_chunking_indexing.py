"""ask 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs_task4"
MOCK_MODE = os.getenv("TASK4_MOCK_MODE", "").lower() in {"1", "true", "yes"}


def load_documents() -> list[dict]:
    """Load every non-empty Markdown file under data/standardized."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative = md_file.relative_to(STANDARDIZED_DIR)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": relative.as_posix(),
                    "type": relative.parts[0]
                    if len(relative.parts) > 1
                    else "unknown",
                },
            }
        )
    return documents


class _MockEmbeddings:
    """Deterministic mock embedding used only for isolated smoke tests."""

    def _embed(self, text: str) -> list[float]:
        seed = text.encode("utf-8")
        values = []
        for index in range(EMBEDDING_DIM):
            digest = hashlib.sha256(seed + index.to_bytes(4, "big")).digest()
            values.append((int.from_bytes(digest[:4], "big") / 2**32) * 2 - 1)
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class _OpenAIEmbeddings:
    """Small LangChain-compatible adapter around OpenAI embeddings API."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=EMBEDDING_DIM,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_EMBEDDING_INSTANCE = None


def get_embedding_model():
    """Return OpenAI embeddings, or the isolated mock when no key is present."""
    global _EMBEDDING_INSTANCE
    if _EMBEDDING_INSTANCE is None:
        if MOCK_MODE or not os.getenv("OPENAI_API_KEY"):
            _EMBEDDING_INSTANCE = _MockEmbeddings()
        else:
            _EMBEDDING_INSTANCE = _OpenAIEmbeddings()
    return _EMBEDDING_INSTANCE


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents with RecursiveCharacterTextSplitter or SemanticChunker if available."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = None
    if CHUNKING_METHOD == "semantic":
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            splitter = SemanticChunker(
                get_embedding_model(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75,
            )
        except ImportError:
            print("⚠️ langchain_experimental chưa được cài đặt, dùng RecursiveCharacterTextSplitter...")

    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    size_limiter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for document in documents:
        chunk_index = 0
        for semantic_text in splitter.split_text(document["content"]):
            # Semantic boundaries are preferred, but a hard size limit keeps
            # prompt size predictable and satisfies the shared chunk contract.
            limited_texts = (
                [semantic_text]
                if len(semantic_text) <= CHUNK_SIZE
                else size_limiter.split_text(semantic_text)
            )
            for text in limited_texts:
                text = text.strip()
                if not text:
                    continue
                chunks.append(
                    {
                        "content": text,
                        "metadata": {
                            **document["metadata"],
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach 1536-dimensional embeddings to each chunk."""
    if not chunks:
        return chunks
    vectors = get_embedding_model().embed_documents(
        [chunk["content"] for chunk in chunks]
    )
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = [float(value) for value in vector]
    return chunks


def get_collection():
    """Return the persistent Chroma collection owned by Task 4."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Install chromadb to use the Task 4 vector store.") from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


def reset_task4_collection() -> None:
    """Remove only Task 4's collection before a fresh reindex."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Install chromadb to reset the Task 4 collection.") from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass


def index_to_vectorstore(chunks: list[dict]):
    """Upsert chunks into the isolated Task 4 Chroma collection."""
    collection = get_collection()
    if not chunks:
        return collection

    ids = []
    for chunk in chunks:
        source = chunk["metadata"]["source"].replace("/", "__")
        ids.append(f"{source}__chunk_{chunk['metadata']['chunk_index']}")
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Search Chroma and return the shared retrieval contract."""
    if not query or top_k <= 0:
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[get_embedding_model().embed_query(query)],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for content, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        output.append(
            {
                "content": content,
                "score": round(max(0.0, 1.0 - float(distance)), 6),
                "metadata": metadata or {},
            }
        )
    return sorted(output, key=lambda item: item["score"], reverse=True)


def run_pipeline() -> None:
    """Run load -> semantic chunk -> embed -> isolated Chroma index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} Markdown documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} semantic chunks")
    embedded_chunks = embed_chunks(chunks)
    reset_task4_collection()
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks in {COLLECTION_NAME}")


if __name__ == "__main__":
    run_pipeline()
