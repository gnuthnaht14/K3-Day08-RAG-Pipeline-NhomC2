"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
import re
from collections.abc import Callable

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# TODO: Chọn LLM model (OpenRouter model ID)
LLM_MODEL = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit

INSUFFICIENT_EVIDENCE_MESSAGE = "I cannot verify this information"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source") or f"Source {index}")
        year = str(metadata.get("year") or metadata.get("date") or "n.d.")
        doc_type = str(metadata.get("type") or "unknown")
        content = str(chunk.get("content") or "").strip()

        if not content:
            continue

        context_parts.append(
            f"[Document {index} | Source: {source} | Year: {year} | "
            f"Type: {doc_type} | Citation: [{source}, {year}]]\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    *,
    chunks: list[dict] | None = None,
    answer_generator: Callable[[str, str], str] | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks tối đa đưa vào context
        chunks: Mock/pre-retrieved chunks. Nếu None, gọi retrieve() của Task 9
        answer_generator: Hàm mock nhận (system_prompt, user_message) và trả answer.
            Nếu None, gọi LLM thật qua OpenAI/OpenRouter.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip()
    if not query or top_k <= 0:
        return _generation_result(INSUFFICIENT_EVIDENCE_MESSAGE, [])

    retrieved_chunks = list(chunks) if chunks is not None else retrieve(query, top_k=top_k)
    retrieved_chunks = retrieved_chunks[:top_k]

    # Empty chunks, or chunks with no usable text, are insufficient evidence.
    usable_chunks = [
        chunk
        for chunk in retrieved_chunks
        if str(chunk.get("content") or "").strip()
    ]
    if not usable_chunks:
        return _generation_result(INSUFFICIENT_EVIDENCE_MESSAGE, [])

    reordered = reorder_for_llm(usable_chunks)
    context = format_context(reordered)
    user_message = (
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        f"Question: {query}\n\n"
        "Answer only from the context and cite claims using the supplied "
        "[Source, Year] citation labels."
    )

    if answer_generator is not None:
        answer = answer_generator(SYSTEM_PROMPT, user_message)
    else:
        answer = _call_llm(SYSTEM_PROMPT, user_message)

    answer = str(answer or "").strip()
    if not answer or not _contains_citation(answer):
        answer = INSUFFICIENT_EVIDENCE_MESSAGE

    return _generation_result(answer, usable_chunks)


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Call OpenRouter or OpenAI; imported lazily so mock tests need no SDK."""
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        model = os.getenv("LLM_MODEL", LLM_MODEL)
    elif openai_key:
        client = OpenAI(api_key=openai_key)
        configured_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        model = configured_model.removeprefix("openai/")
    else:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY, or pass "
            "answer_generator for an offline mock test."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def _contains_citation(answer: str) -> bool:
    """Accept a non-empty bracket citation containing source and year/date."""
    return bool(re.search(r"\[[^\[\],]+,\s*[^\[\]]+\]", answer))


def _generation_result(answer: str, chunks: list[dict]) -> dict:
    """Build the stable output schema consumed by app.py and evaluation."""
    retrieval_source = (
        str(chunks[0].get("source") or "hybrid") if chunks else "none"
    )
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Offline demo: thay chunks và answer_generator bằng Task 9 + LLM thật khi
    # các module upstream và API key đã sẵn sàng.
    mock_chunks = [
        {
            "content": "Học phí được công bố theo từng chương trình và kỳ học.",
            "score": 0.91,
            "metadata": {
                "source": "RMIT Tuition Fees",
                "year": 2026,
                "type": "legal",
            },
            "source": "hybrid",
        },
        {
            "content": "Sinh viên cần kiểm tra hóa đơn trước hạn thanh toán.",
            "score": 0.82,
            "metadata": {
                "source": "RMIT Payment Guide",
                "year": 2026,
                "type": "guide",
            },
            "source": "hybrid",
        },
    ]

    def mock_answer_generator(_system_prompt: str, _user_message: str) -> str:
        return (
            "Học phí phụ thuộc vào chương trình và kỳ học; sinh viên nên kiểm "
            "tra hóa đơn trước hạn thanh toán [RMIT Tuition Fees, 2026]."
        )

    question = "Học phí và thời hạn thanh toán được xác định như thế nào?"
    result = generate_with_citation(
        question,
        chunks=mock_chunks,
        answer_generator=mock_answer_generator,
    )
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(
        f"[Sources: {len(result['sources'])} chunks | "
        f"via {result['retrieval_source']}]"
    )
