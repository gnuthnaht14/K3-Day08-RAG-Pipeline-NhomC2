"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Lưu mapping tên file -> doc_id để không phải upload lại mỗi lần chạy
DOC_ID_MAP_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"
# Thư mục tạm chứa các PDF convert từ markdown trước khi upload
_TMP_PDF_DIR = Path(__file__).parent.parent / "data" / "_pageindex_tmp_pdf"


def _load_doc_id_map() -> dict:
    if DOC_ID_MAP_PATH.exists():
        return json.loads(DOC_ID_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_id_map(doc_id_map: dict) -> None:
    DOC_ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_ID_MAP_PATH.write_text(
        json.dumps(doc_id_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """Convert 1 file markdown sang PDF đơn giản (chỉ giữ text thô) bằng fpdf2,
    vì PageIndex nhận PDF chứ không nhận .md trực tiếp."""
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Built-in Helvetica is Latin-1 only and fails on Vietnamese text. Use a
    # Unicode TTF available on the current OS instead.
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    unicode_font = next((font for font in font_candidates if font.exists()), None)
    if unicode_font is None:
        raise FileNotFoundError(
            "Không tìm thấy font Unicode để chuyển Markdown tiếng Việt sang PDF."
        )
    pdf.add_font("DocumentUnicode", fname=str(unicode_font))
    pdf.set_font("DocumentUnicode", size=11)

    for line in text.splitlines():
        # multi_cell tự động xuống dòng, tránh lỗi text quá dài trên 1 dòng
        line = line.expandtabs(4)
        line = re.sub(r"[\ue000-\uf8ff]", "•", line)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, line if line.strip() else " ")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def _sync_remote_doc_ids(client, md_files: list[Path], doc_id_map: dict) -> dict:
    """Recover local cache entries for documents already uploaded remotely."""
    response = client.list_documents()
    remote_documents = response.get("documents", [])
    remote_ids = {
        remote.get("doc_id") or remote.get("id")
        for remote in remote_documents
        if remote.get("doc_id") or remote.get("id")
    }

    # An API-key/account change makes the old cached IDs inaccessible. Remove
    # them before matching/uploading so setdefault cannot preserve stale IDs.
    stale_keys = [
        key for key, doc_id in doc_id_map.items() if doc_id not in remote_ids
    ]
    for key in stale_keys:
        doc_id_map.pop(key, None)
    if stale_keys:
        print(
            f"  ↻ Removed {len(stale_keys)} stale cached document ID(s) "
            "after API key/account change"
        )

    markdown_by_pdf_name: dict[str, list[Path]] = {}
    for md_file in md_files:
        markdown_by_pdf_name.setdefault(f"{md_file.stem}.pdf", []).append(md_file)

    for remote in remote_documents:
        remote_name = (
            remote.get("name")
            or remote.get("file_name")
            or remote.get("filename")
        )
        doc_id = remote.get("doc_id") or remote.get("id")
        matches = markdown_by_pdf_name.get(str(remote_name), [])
        if doc_id and len(matches) == 1:
            key = str(matches[0].relative_to(STANDARDIZED_DIR))
            doc_id_map.setdefault(key, doc_id)

    print(
        f"  Found {response.get('total', len(remote_documents))} remote "
        f"PageIndex document(s)"
    )
    return doc_id_map


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong file .env.")

    from pageindex import PageIndexClient
    from pageindex.client import PageIndexAPIError

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_id_map = _load_doc_id_map()

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"  ⚠ Không tìm thấy file .md nào trong {STANDARDIZED_DIR}")
        return doc_id_map

    doc_id_map = _sync_remote_doc_ids(client, md_files, doc_id_map)
    _save_doc_id_map(doc_id_map)

    for md_file in md_files:
        key = str(md_file.relative_to(STANDARDIZED_DIR))

        if key in doc_id_map:
            print(f"  ↷ Bỏ qua (đã upload trước đó): {md_file.name}")
            continue

        pdf_path = _TMP_PDF_DIR / (md_file.stem + ".pdf")
        _markdown_to_pdf(md_file, pdf_path)

        try:
            resp = client.submit_document(str(pdf_path))
        except PageIndexAPIError as exc:
            if "LimitReached" in str(exc):
                print(
                    "  ⚠ Đã đạt giới hạn tài liệu PageIndex. Giữ các tài liệu "
                    "đã upload và bỏ qua phần còn lại."
                )
                break
            raise
        doc_id = resp.get("doc_id") or resp.get("id")
        doc_id_map[key] = doc_id
        # Persist after every successful upload so an API/quota error cannot
        # cause the next run to upload the same document again.
        _save_doc_id_map(doc_id_map)
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    _save_doc_id_map(doc_id_map)
    return doc_id_map


def wait_until_documents_ready(
    doc_id_map: dict, timeout_seconds: int = 300, poll_seconds: int = 5
) -> bool:
    """Wait for all cached PageIndex documents to finish cloud processing."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong file .env.")

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    pending = dict(doc_id_map)
    deadline = time.monotonic() + timeout_seconds

    while pending and time.monotonic() < deadline:
        for name, doc_id in list(pending.items()):
            status = str(client.get_document(doc_id).get("status", "unknown"))
            if status == "completed":
                print(f"  ✓ Processed: {name}")
                pending.pop(name)
            elif status in {"failed", "error"}:
                print(f"  ✗ Processing failed: {name}")
                pending.pop(name)
        if pending:
            print(f"  … Waiting for {len(pending)} document(s)")
            time.sleep(poll_seconds)

    if pending:
        print(
            "  ⚠ Hết thời gian chờ. Kiểm tra trạng thái tài liệu trên "
            "PageIndex dashboard rồi chạy lại."
        )
        return False
    return True


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong file .env.")

    from pageindex import PageIndexClient
    from pageindex.client import PageIndexAPIError

    doc_id_map = _load_doc_id_map()
    if not doc_id_map:
        raise RuntimeError(
            "Chưa có doc_id nào trong PageIndex — hãy gọi upload_documents() trước."
        )

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    retrieval_ids: dict[str, str] = {}
    for doc_id in doc_id_map.values():
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
        except PageIndexAPIError as exc:
            error_message = str(exc)
            if (
                "Insufficient credits" in error_message
                or "LimitReached" in error_message
                or "Access denied to document" in error_message
            ):
                print(
                    "  ⚠ PageIndex fallback không khả dụng (credit hoặc quyền "
                    "truy cập tài liệu); giữ kết quả hybrid."
                )
                return []
            raise
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        if retrieval_id:
            retrieval_ids[retrieval_id] = doc_id

    # Poll all submitted retrievals in one shared window. Polling each document
    # sequentially could block the Streamlit request for N * 60 seconds.
    completed_retrievals: list[dict] = []
    pending = dict(retrieval_ids)
    deadline = time.monotonic() + 60
    while pending and time.monotonic() < deadline:
        for retrieval_id in list(pending):
            try:
                retrieval = client.get_retrieval(retrieval_id)
            except PageIndexAPIError as exc:
                error_message = str(exc)
                if (
                    "Insufficient credits" in error_message
                    or "LimitReached" in error_message
                    or "Access denied" in error_message
                ):
                    pending.pop(retrieval_id, None)
                    continue
                raise

            status = retrieval.get("status")
            if status == "completed":
                completed_retrievals.append(retrieval)
                pending.pop(retrieval_id, None)
            elif status in {"failed", "error"}:
                pending.pop(retrieval_id, None)

        if pending:
            time.sleep(2)

    results: list[dict] = []
    rank = 0
    for retrieval in completed_retrievals:
        # API /retrieval is legacy and does not return a comparable relevance
        # score. Assign a deterministic global reciprocal-rank score.
        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    content = item.get("relevant_content", "").strip()
                    if not content:
                        continue
                    rank += 1
                    results.append(
                        {
                            "content": content,
                            "score": 1.0 / rank,
                            "metadata": {
                                "section": item.get("section_title"),
                            },
                            "source": "pageindex",
                        }
                    )

    # Chuẩn hoá theo contract chung: sort giảm dần theo score trước khi cắt top_k
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        uploaded_documents = upload_documents()

        print("\nWaiting for PageIndex processing...")
        if wait_until_documents_ready(uploaded_documents):
            print("\nTest query:")
            results = pageindex_search("student tuition payment", top_k=3)
            for r in results:
                print(f"[{r['score']:.3f}] {r['content'][:100]}...")
