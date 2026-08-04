"""Streamlit demo UI for the University Services RAG pipeline.

Run with:
    streamlit run app.py

The normal mode connects Task 10 -> Task 9. An explicitly labelled offline
demo mode keeps the presentation usable when the vector index or API is not
available; it uses extractive search over the repository's Markdown corpus.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
PAGEINDEX_MAP = PROJECT_ROOT / "data" / "pageindex_doc_ids.json"

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="UniGuide AI · University Services",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --brand: #4f46e5;
        --brand-soft: #eef2ff;
        --mint: #0f766e;
        --surface: rgba(255,255,255,.82);
        --line: rgba(148,163,184,.24);
    }
    .stApp {
        background:
          radial-gradient(circle at 10% 0%, rgba(99,102,241,.11), transparent 28rem),
          radial-gradient(circle at 95% 12%, rgba(13,148,136,.09), transparent 24rem),
          #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,.86);
        border-right: 1px solid var(--line);
    }
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 5rem;}
    .hero {
        padding: 1.55rem 1.7rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(125deg, rgba(255,255,255,.96), rgba(238,242,255,.82));
        box-shadow: 0 18px 50px rgba(51,65,85,.08);
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: var(--brand); font-size: .76rem; font-weight: 800;
        letter-spacing: .13em; text-transform: uppercase; margin-bottom: .45rem;
    }
    .hero h1 {color: var(--ink); font-size: 2.25rem; line-height: 1.1; margin: 0 0 .55rem;}
    .hero p {color: var(--muted); margin: 0; max-width: 760px; line-height: 1.65;}
    .status-pill {
        display: inline-flex; align-items: center; gap: .4rem; padding: .32rem .68rem;
        border-radius: 999px; background: #ecfdf5; color: #047857;
        font-size: .78rem; font-weight: 700; margin-top: .85rem;
    }
    .status-pill.demo {background: #fff7ed; color: #c2410c;}
    .suggestion-label {color: var(--muted); font-size: .85rem; margin: 1.15rem 0 .35rem;}
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line); border-radius: 18px;
        background: var(--surface); padding: .35rem .45rem;
        box-shadow: 0 8px 28px rgba(51,65,85,.04);
    }
    [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {margin-top: .65rem;}
    .answer-meta {color: var(--muted); font-size: .78rem; margin-top: .65rem;}
    .source-title {font-weight: 750; color: var(--ink);}
    .source-meta {font-size: .78rem; color: var(--muted); margin-bottom: .45rem;}
    .source-content {color: #334155; font-size: .9rem; line-height: 1.55;}
    .pipeline {
        font-size: .79rem; color: var(--muted); line-height: 1.8;
        padding: .75rem .85rem; border-radius: 14px; background: #f8fafc;
        border: 1px solid var(--line);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.7); border: 1px solid var(--line);
        padding: .72rem .85rem; border-radius: 16px;
    }
    .stButton > button {border-radius: 12px;}
    [data-testid="stChatInput"] {border-radius: 16px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


SUGGESTIONS = [
    ("💳", "Học phí tại RMIT Vietnam là bao nhiêu?"),
    ("🏅", "Điều kiện xin học bổng Academic Achievement?"),
    ("🗓️", "Cách đăng ký học phần qua myRMIT?"),
]

DATASET_SUGGESTIONS = [
    "Sinh viên cần lưu ý gì khi đăng ký học phần?",
    "Thủ tục xin nộp học phí muộn gồm những gì?",
    "Quy định khen thưởng sinh viên xuất sắc như thế nào?",
]

FOLLOW_UP_MARKERS = (
    "còn",
    "thế còn",
    "trường hợp đó",
    "điều này",
    "nó ",
    "vậy ",
    "bao lâu",
    "ở đâu",
    "như thế nào",
)


def _init_state() -> None:
    defaults = {
        "messages": [],
        "pending_query": None,
        "last_route": "—",
        "last_latency": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _queue_query(question: str) -> None:
    st.session_state.pending_query = question


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_query = None
    st.session_state.last_route = "—"
    st.session_state.last_latency = None


@st.cache_data(show_spinner=False, ttl=5)
def _health_snapshot() -> dict[str, Any]:
    markdown_count = len(list(STANDARDIZED_DIR.rglob("*.md")))
    has_api_key = bool(
        os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    )
    vector_chunks = 0
    if CHROMA_DIR.exists():
        try:
            import chromadb

            from src.task4_chunking_indexing import COLLECTION_NAME

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            vector_chunks = client.get_collection(COLLECTION_NAME).count()
        except Exception:
            vector_chunks = 0
    return {
        "documents": markdown_count,
        "has_index": vector_chunks > 0,
        "vector_chunks": vector_chunks,
        "has_api_key": has_api_key,
        "has_pageindex": PAGEINDEX_MAP.exists(),
    }


def _contextualize_query(query: str, messages: list[dict]) -> str:
    """Attach the previous user turn only when the new question looks elliptical."""
    normalized = query.lower().strip()
    looks_like_follow_up = len(query.split()) <= 8 or any(
        marker in normalized for marker in FOLLOW_UP_MARKERS
    )
    if not looks_like_follow_up:
        return query

    previous_questions = [
        message["content"]
        for message in messages
        if message.get("role") == "user"
    ]
    if not previous_questions:
        return query
    return (
        f"Câu hỏi trước: {previous_questions[-1]}\n"
        f"Câu hỏi tiếp theo: {query}"
    )


@st.cache_data(show_spinner=False)
def _offline_corpus() -> list[dict]:
    chunks: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [
            re.sub(r"\s+", " ", part).strip(" #\t")
            for part in re.split(r"\n\s*\n|(?=^#{1,3}\s)", content, flags=re.MULTILINE)
        ]
        relative = md_file.relative_to(STANDARDIZED_DIR)
        for index, paragraph in enumerate(paragraphs):
            if len(paragraph) < 70:
                continue
            year_match = re.search(r"\b(20\d{2})\b", paragraph)
            chunks.append(
                {
                    "content": paragraph[:1600],
                    "metadata": {
                        "source": relative.as_posix(),
                        "type": relative.parts[0] if len(relative.parts) > 1 else "document",
                        "year": year_match.group(1) if year_match else "n.d.",
                        "chunk_index": index,
                    },
                    "source": "offline-demo",
                }
            )
    return chunks


def _tokens(text: str) -> set[str]:
    stop_words = {
        "và", "là", "có", "của", "cho", "tại", "như", "thế", "nào",
        "được", "trong", "về", "một", "các", "theo", "tôi", "xin",
    }
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stop_words
    }


def _token_sequence(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _offline_demo_response(query: str, top_k: int) -> dict:
    """Small extractive fallback for UI rehearsals; never presented as real RAG."""
    query_tokens = _tokens(query)
    normalized_query = query.lower()
    required_entities = [
        entity
        for entity in ("rmit", "myrmit", "academic achievement")
        if entity in normalized_query
    ]
    query_sequence = _token_sequence(query)
    query_bigrams = set(zip(query_sequence, query_sequence[1:]))
    ranked: list[dict] = []
    for chunk in _offline_corpus():
        searchable_text = (
            f"{chunk['content']} {chunk['metadata'].get('source', '')}"
        ).lower()
        if required_entities and not all(
            entity in searchable_text for entity in required_entities
        ):
            continue
        overlap = query_tokens & _tokens(chunk["content"])
        # One generic shared word (for example "học") is not enough evidence.
        if len(overlap) < 2:
            continue
        content_sequence = _token_sequence(chunk["content"])
        content_bigrams = set(zip(content_sequence, content_sequence[1:]))
        phrase_matches = len(query_bigrams & content_bigrams)
        item = chunk.copy()
        item["score"] = round(
            len(overlap) / max(len(query_tokens), 1) + phrase_matches * 0.15,
            4,
        )
        ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    sources = ranked[:top_k]

    if not sources:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "offline-demo",
        }

    excerpts = []
    for source in sources[:2]:
        metadata = source["metadata"]
        excerpt = source["content"][:420].rstrip()
        excerpts.append(
            f"{excerpt}… [{metadata['source']}, {metadata['year']}]"
        )
    return {
        "answer": "\n\n".join(excerpts),
        "sources": sources,
        "retrieval_source": "offline-demo",
    }


def _run_pipeline(
    query: str,
    top_k: int,
    offline_mode: bool,
    use_pageindex: bool,
) -> dict:
    if offline_mode:
        return _offline_demo_response(query, top_k)

    from src.task10_generation import generate_with_citation

    return generate_with_citation(
        query,
        top_k=top_k,
        use_pageindex=use_pageindex,
    )


def _source_name(source: dict, index: int) -> str:
    metadata = source.get("metadata") or {}
    return str(
        metadata.get("source")
        or metadata.get("title")
        or metadata.get("section")
        or f"Tài liệu {index}"
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"📚 Xem {len(sources)} đoạn nguồn được sử dụng"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            name = _source_name(source, index)
            doc_type = metadata.get("type", "document")
            year = metadata.get("year") or metadata.get("date") or "n.d."
            route = source.get("source", "hybrid")
            try:
                score = f"{float(source.get('score', 0)):.4f}"
            except (TypeError, ValueError):
                score = "n/a"

            with st.container(border=True):
                st.markdown(
                    f"<div class='source-title'>{index}. {html.escape(name)}</div>"
                    f"<div class='source-meta'>{html.escape(str(doc_type))} · "
                    f"{html.escape(str(year))} · {html.escape(str(route))} · "
                    f"score {score}</div>",
                    unsafe_allow_html=True,
                )
                content = str(source.get("content") or "").strip()
                st.markdown(
                    f"<div class='source-content'>{html.escape(content[:650])}"
                    f"{'…' if len(content) > 650 else ''}</div>",
                    unsafe_allow_html=True,
                )


def _render_message(message: dict) -> None:
    avatar = "🎓" if message["role"] == "assistant" else "🙂"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            latency = message.get("latency")
            route = message.get("route", "unknown")
            meta = f"Tuyến truy xuất: `{route}`"
            if latency is not None:
                meta += f" · `{latency:.2f}s`"
            st.markdown(f"<div class='answer-meta'>{meta}</div>", unsafe_allow_html=True)
            _render_sources(message.get("sources") or [])


_init_state()
health = _health_snapshot()
default_offline = not (health["has_index"] and health["has_api_key"])


with st.sidebar:
    st.markdown("## 🎓 UniGuide AI")
    st.caption("Trợ lý chính sách & dịch vụ đại học")
    st.divider()

    offline_mode = st.toggle(
        "Chế độ demo offline",
        value=default_offline,
        help="Dùng tìm kiếm trích đoạn cục bộ, không gọi vector DB hoặc LLM.",
    )
    use_pageindex = st.toggle(
        "Cho phép PageIndex fallback",
        value=False,
        disabled=offline_mode or not health["has_pageindex"],
        help=(
            "Mỗi retrieval có thể tiêu tốn PageIndex credit. Chỉ bật khi "
            "muốn demo fallback."
        ),
    )
    top_k = st.slider(
        "Số đoạn nguồn",
        min_value=3,
        max_value=8,
        value=5,
        help="Số chunks tối đa được đưa vào context của LLM.",
    )

    st.markdown("### Trạng thái")
    st.write("✅ Dữ liệu Markdown" if health["documents"] else "❌ Chưa có dữ liệu")
    st.write(
        f"✅ Vector index ({health['vector_chunks']} chunks)"
        if health["has_index"]
        else "⚠️ Chưa có vector index"
    )
    st.write("✅ API key" if health["has_api_key"] else "⚠️ Chưa có API key")
    st.write(
        "✅ PageIndex đã cấu hình (chưa kiểm tra credit)"
        if health["has_pageindex"]
        else "○ PageIndex chưa cấu hình"
    )

    st.markdown("### Pipeline")
    st.markdown(
        "<div class='pipeline'>Query<br>↳ Semantic + BM25<br>↳ RRF fusion"
        "<br>↳ PageIndex fallback<br>↳ LLM + citations</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    col_clear, col_export = st.columns(2)
    with col_clear:
        st.button("Xóa chat", on_click=_clear_chat, use_container_width=True)
    with col_export:
        transcript = json.dumps(
            st.session_state.messages,
            ensure_ascii=False,
            indent=2,
        )
        st.download_button(
            "Tải chat",
            data=transcript,
            file_name="uniguide-chat.json",
            mime="application/json",
            use_container_width=True,
        )


mode_class = "demo" if offline_mode else ""
mode_label = "Offline rehearsal" if offline_mode else "RAG pipeline online"
st.markdown(
    f"""
    <section class="hero">
      <div class="eyebrow">University services · grounded answers</div>
      <h1>Hỏi chính sách, nhận câu trả lời có nguồn.</h1>
      <p>UniGuide kết hợp tìm kiếm ngữ nghĩa và từ khóa để trả lời về học phí,
      học bổng, đăng ký học phần và các dịch vụ sinh viên.</p>
      <span class="status-pill {mode_class}">● {mode_label}</span>
    </section>
    """,
    unsafe_allow_html=True,
)

metric_docs, metric_route, metric_latency = st.columns(3)
metric_docs.metric("Tài liệu sẵn sàng", health["documents"])
metric_route.metric("Tuyến gần nhất", st.session_state.last_route)
latency_value = (
    f"{st.session_state.last_latency:.2f}s"
    if st.session_state.last_latency is not None
    else "—"
)
metric_latency.metric("Thời gian phản hồi", latency_value)


if not st.session_state.messages:
    st.markdown("<div class='suggestion-label'>Câu hỏi mẫu theo yêu cầu bài lab</div>", unsafe_allow_html=True)
    suggestion_columns = st.columns(3)
    for column, (icon, question) in zip(suggestion_columns, SUGGESTIONS):
        with column:
            st.button(
                f"{icon}  {question}",
                key=f"suggestion_{question}",
                on_click=_queue_query,
                args=(question,),
                use_container_width=True,
            )

    st.markdown("<div class='suggestion-label'>Câu hỏi phù hợp bộ dữ liệu hiện có</div>", unsafe_allow_html=True)
    dataset_columns = st.columns(3)
    for column, question in zip(dataset_columns, DATASET_SUGGESTIONS):
        with column:
            st.button(
                question,
                key=f"dataset_suggestion_{question}",
                on_click=_queue_query,
                args=(question,),
                use_container_width=True,
            )


for chat_message in st.session_state.messages:
    _render_message(chat_message)


typed_query = st.chat_input(
    "Hỏi về học phí, học bổng, đăng ký học phần…",
)
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    previous_messages = list(st.session_state.messages)
    effective_query = _contextualize_query(query, previous_messages)

    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant", avatar="🎓"):
        started_at = time.perf_counter()
        try:
            with st.status("Đang truy xuất và kiểm chứng nguồn…", expanded=True) as status:
                st.write("Tìm kiếm semantic và BM25")
                response = _run_pipeline(
                    effective_query,
                    top_k,
                    offline_mode,
                    use_pageindex,
                )
                st.write("Sắp xếp lại context và kiểm tra citation")
                status.update(label="Đã hoàn tất", state="complete", expanded=False)

            latency = time.perf_counter() - started_at
            answer = str(response.get("answer") or "I cannot verify this information")
            sources = list(response.get("sources") or [])
            route = str(response.get("retrieval_source") or "unknown")

            st.markdown(answer)
            st.markdown(
                f"<div class='answer-meta'>Tuyến truy xuất: `{route}` · `{latency:.2f}s`</div>",
                unsafe_allow_html=True,
            )
            _render_sources(sources)
        except Exception as exc:
            latency = time.perf_counter() - started_at
            route = "error"
            sources = []
            answer = (
                "Mình chưa thể hoàn tất truy vấn này. Hãy kiểm tra vector index, "
                "API key hoặc bật **Chế độ demo offline** trong thanh bên."
            )
            st.error(answer)
            with st.expander("Chi tiết kỹ thuật"):
                st.code(f"{type(exc).__name__}: {exc}")

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "route": route,
        "latency": latency,
    }
    st.session_state.messages.append(assistant_message)
    st.session_state.last_route = route
    st.session_state.last_latency = latency
