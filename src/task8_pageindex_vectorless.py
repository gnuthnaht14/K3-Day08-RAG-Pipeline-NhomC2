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
    pdf.set_font("Helvetica", size=11)

    for line in text.splitlines():
        # multi_cell tự động xuống dòng, tránh lỗi text quá dài trên 1 dòng
        pdf.multi_cell(0, 6, line if line.strip() else " ")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_id_map = _load_doc_id_map()

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"  ⚠ Không tìm thấy file .md nào trong {STANDARDIZED_DIR}")
        return doc_id_map

    for md_file in md_files:
        key = str(md_file.relative_to(STANDARDIZED_DIR))

        if key in doc_id_map:
            print(f"  ↷ Bỏ qua (đã upload trước đó): {md_file.name}")
            continue

        pdf_path = _TMP_PDF_DIR / (md_file.stem + ".pdf")
        _markdown_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        doc_id_map[key] = doc_id
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    _save_doc_id_map(doc_id_map)
    return doc_id_map


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
    from pageindex.client import PageIndexClient

    doc_id_map = _load_doc_id_map()
    if not doc_id_map:
        raise RuntimeError(
            "Chưa có doc_id nào trong PageIndex — hãy gọi upload_documents() trước."
        )

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    results: list[dict] = []

    for doc_id in doc_id_map.values():
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")

        # Poll cho đến khi status == "completed" (API /retrieval là async)
        retrieval = None
        for _ in range(30):  # tối đa ~60s (30 * 2s)
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") == "completed":
                break
            time.sleep(2)

        if retrieval is None or retrieval.get("status") != "completed":
            continue

        # Lưu ý: API /retrieval đã deprecated (response có field "deprecation").
        # In json.dumps(retrieval) ra 1 lần trước khi tin cấu trúc parse dưới đây,
        # vì schema thật có thể khác pseudocode tham khảo.
        #
        # PageIndex không trả score trực tiếp -> tự gán theo rank (1/rank), dùng
        # rank GLOBAL trong retrieval của 1 doc (không reset về 1 ở mỗi group) để
        # điểm số vẫn phản ánh đúng thứ tự ưu tiên PageIndex trả về.
        rank = 0
        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    rank += 1
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": 1.0 / rank,
                        "metadata": {"section": item.get("section_title")},
                        "source": "pageindex",
                    })

    # Chuẩn hoá theo contract chung: sort giảm dần theo score trước khi cắt top_k
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")