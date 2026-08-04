"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://uet.vnu.edu.vn/mo-cong-dang-ky-ho-so-xet-tuyen-dai-hoc-chinh-quy-nam-2026/",
    "https://uet.vnu.edu.vn/cac-nganh-tuyen-sinh-dai-hoc-2026-tai-truong-dai-hoc-cong-nghe-thuoc-danh-muc-cac-nganh-dao-tao-duoc-ap-dung-chinh-sach-hoc-bong-theo-nghi-dinh-so-179-2026-nd-cp/",
    "https://uet.vnu.edu.vn/nam-2026-truong-dh-cong-nghe-dhqg-ha-noi-tuyen-sinh-dai-hoc-co-gi-moi-ho-tro-sinh-vien-ra-sao/",
    "https://uet.vnu.edu.vn/canh-giac-voi-cac-thong-bao-giay-bao-gia-mao-gui-toi-thi-sinh-phu-huynh-trong-ky-tuyen-sinh-dai-hoc-nam-2025/",
    "https://uet.vnu.edu.vn/le-trao-hoc-bong-truong-dai-hoc-cong-nghe-nam-hoc-2024-2025/",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = "Unknown"
            if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                title = result.metadata.get("title") or result.metadata.get("og:title") or "Unknown"
            elif hasattr(result, "title") and result.title:
                title = result.title

            markdown_content = getattr(result, "markdown", "") or ""
            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": markdown_content,
            }
    except Exception as e:
        print(f"  ⚠️ Crawl4AI error ({e}), dùng fallback requests...")
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title else "Unknown"
        text = soup.get_text(separator="\n\n")
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": text,
        }



async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
