"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    if not legal_dir.exists():
        print(f"Không tìm thấy thư mục nguồn: {legal_dir}")
        return

    md = MarkItDown()

    # Duyệt đệ quy và giữ nguyên phần cấu trúc con dưới data/landing/legal.
    for filepath in sorted(legal_dir.rglob("*")):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            relative_path = filepath.relative_to(legal_dir)
            output_path = (OUTPUT_DIR / "legal" / relative_path).with_suffix(".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Converting: {relative_path}")
            result = md.convert(str(filepath))
            content = result.text_content.strip()
            if not content:
                print(
                    "  [SKIP] PDF không có lớp text; cần OCR trước khi convert "
                    f"({relative_path})"
                )
                continue

            output_path.write_text(content + "\n", encoding="utf-8")
            print(f"  [OK] Saved: {output_path.relative_to(OUTPUT_DIR)}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    if not news_dir.exists():
        print(f"Không tìm thấy thư mục nguồn: {news_dir}")
        return

    for filepath in sorted(news_dir.rglob("*.json")):
        if filepath.suffix.lower() == ".json":
            relative_path = filepath.relative_to(news_dir)
            output_path = (OUTPUT_DIR / "news" / relative_path).with_suffix(".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Converting: {relative_path}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            title = data.get("title") or filepath.stem
            header = (
                f"# {title}\n\n"
                f"**Source:** {data.get('url', 'N/A')}\n"
                f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n"
                "---\n\n"
            )
            content = str(data.get("content_markdown") or "").strip()
            output_path.write_text(header + content + "\n", encoding="utf-8")
            print(f"  [OK] Saved: {output_path.relative_to(OUTPUT_DIR)}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
