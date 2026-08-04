# Hướng Dẫn Bài Lab Ngày 8 — K3 (University Services RAG)

---

## 1. Bài này là gì?

**K3-Day08-RAG-Pipeline-Starter** là bài lab Ngày 8 — bạn xây dựng một **RAG pipeline end-to-end** (từ thu thập dữ liệu đến chatbot trả lời có trích dẫn nguồn).

Chủ đề của K3: **Dịch vụ & chính sách đại học** — chatbot trả lời câu hỏi về học phí, học bổng, ký túc xá, đăng ký học phần, thư viện, hỗ trợ sinh viên...

Dữ liệu mẫu trong repo được lấy từ trang công khai **RMIT Vietnam** (`rmit.edu.vn`).

Bài này **nối tiếp Ngày 7 (K3 Variant)** — cùng domain "University Services", nên pipeline bạn làm hôm nay sẽ nhất quán với những gì đã học ở lab trước.

---

## 2. Cách làm bài: Nhóm tự chia task + Report cá nhân theo phần mình phụ trách

> **Có, cách này hoàn toàn ổn** — và thực ra **đúng tinh thần bài lab** hơn là bắt 4–6 người làm lại y hệt 10 task.

### Mô hình chung

| Ai | Làm gì | Nộp gì (cá nhân) |
|----|--------|------------------|
| **Cả nhóm** | **Tự họp chia task**, dùng **1 repo chung**, ghép pipeline chạy được | Demo nhóm + chatbot/eval |
| **Mỗi bạn** | **Implement sâu** phần task được giao | **Report cá nhân** + pytest **đúng task của mình** |
| **Không bắt buộc** | Mỗi người tự tay code hết Task 1→10 | — |

Pipeline 10 bước vẫn phải **đủ và chạy được ở cấp nhóm**. Điểm cá nhân căn vào **phần bạn chịu trách nhiệm**, không phải “ai cũng pass 35/35 test”.

### Nguyên tắc coach hay nhắc

1. **Owner rõ ràng** — mỗi file task (`task5_...py`, `task9_...py`...) có **1 người chính**; ghi tên trong `group_project/README.md`.
2. **Hiểu toàn pipeline** — bạn không code Task 7 vẫn phải giải thích được Task 7 làm gì trong demo (ít nhất 2–3 câu).
3. **Report cá nhân ≠ copy code nhóm** — giải thích *lý do kỹ thuật*, *trade-off*, *lỗi đã gặp* ở phần mình làm.
4. **Pytest theo phần mình** — chạy test **đúng task bạn owner**, không cần pass hết 10 task.

### Các bạn tự chia role — không có bảng cố định

**Coach không gán sẵn R1/R2/R3...** Nhóm **tự họp 5–10 phút đầu buổi**, thống nhất ai làm gì dựa trên:
- Sở thích / kinh nghiệm (ai thích crawl data, ai thích UI, ai thích thuật toán...)
- Khối lượng công việc **cân bằng** — tránh 1 người ôm 7 task, 3 người kia nhàn
- Thứ tự phụ thuộc pipeline (data phải xong trước khi index; search xong trước khi ghép Task 9)

**Việc nhóm cần làm ngay:**
1. Mở `group_project/README.md` → điền bảng phân công (tên, MSSV, **task số mấy**, file nào)
2. Chọn 1 người **điều phối** (không nhất thiết = người code nhiều nhất) — nhắc deadline từng giai đoạn
3. Thống nhất **1 repo chung** + cách merge code (ai push branch nào, ai review)

**Gợi ý chia theo “khối”, không bắt buộc:**

| Khối công việc | Task liên quan | Ai thường hợp? |
|----------------|----------------|----------------|
| Data | 1, 2, 3 | Bạn thích crawl, xử lý file |
| Index & search | 4, 5, 6 | Bạn thích ML / embedding |
| Retrieval nâng cao | 7, 8, 9 | Bạn thích ghép pipeline, thuật toán |
| Product | 10, `app.py`, eval | Bạn thích UI, demo, viết báo cáo |

Nhóm 4 người → mỗi người ~2–3 task (+ eval chia chung). Nhóm 6 người → có thể tách eval riêng 1 người. **Cách chia do các bạn quyết**, miễn **đủ 10 task có owner** và **không trùng**.

**Ví dụ thực tế (chỉ tham khảo, không copy):**
- Bạn A: Task 1, 2, 3  
- Bạn B: Task 4, 5  
- Bạn C: Task 6, 7, 8  
- Bạn D: Task 9, 10 + điều phối  
- Bạn E: `app.py`, golden dataset, RAGAS  

Nhóm khác có thể chia hoàn toàn khác — **miễn ghi rõ trong README**.

### Report cá nhân — nộp những gì?

Mỗi bạn nộp **1 file report ngắn** (PDF/MD, 1–2 trang), gồm:

1. **Task mình được nhóm giao** (vd: Task 4, 5 — không cần tên role R1/R2)
2. **Bạn đã implement gì** — file nào, hàm nào, tham số chọn (chunk size, model...)
3. **Vì sao chọn cách đó** — 3–5 câu giải thích kỹ thuật
4. **Kết quả kiểm tra** — screenshot/log pytest task của mình
5. **1 lỗi đã gặp + cách fix** (nếu có)
6. **Cách phần mình nối với task khác trong pipeline** (vd: Task 5 output đi đâu trước Task 9)

**Không cần** viết lại toàn bộ 10 task nếu bạn không làm.

### Chấm điểm cá nhân (theo phần được giao)

| Tiêu chí | Gợi ý trọng số |
|----------|----------------|
| Task mình owner **chạy đúng** (pytest pass) | ~60% |
| **Report cá nhân** rõ ràng, có giải thích kỹ thuật | ~30% |
| Tham gia demo — trả lời được câu hỏi về **phần mình** + hiểu sơ pipeline | ~10% |

Repo starter ghi “35/35 test = 50đ cá nhân” — đó là **mốc nếu làm solo**. Làm theo nhóm: map điểm theo **task bạn owner** (Task 1 = 3đ, Task 4 = 7đ...).

### Lệnh pytest — chạy đúng task của bạn

```bash
# Thay X bằng số task bạn owner, vd Task 5:
pytest tests/test_individual.py::TestTask5 -v

# Nếu owner nhiều task:
pytest tests/test_individual.py::TestTask4 tests/test_individual.py::TestTask5 -v
```

### Checklist nhóm (trước khi demo)

- [ ] Đã **tự họp chia task** + ghi vào `group_project/README.md` (tên + MSSV + task số)
- [ ] Mỗi task 1–10 **có đúng 1 owner** (eval có thể 1–2 người phụ)
- [ ] **Cả nhóm:** `pytest tests/test_individual.py -v` pass trên repo chung
- [ ] Mỗi người có **report cá nhân** riêng
- [ ] Demo: ai cũng nói được 1–2 phút về **phần mình** + 1 câu về pipeline tổng

---

## 3. Toàn cảnh pipeline — 10 Task (tham khảo cho cả nhóm)

Hãy tưởng tượng cả nhóm đang xây một **thủ thư AI** cho sinh viên:

```
Thu thập tài liệu → Chuẩn hóa → Cắt nhỏ → Lưu vector
    → Tìm kiếm (semantic + BM25) → Xếp hạng lại → Fallback nếu yếu
        → LLM trả lời kèm nguồn trích dẫn
```

### Task 1 — Tải văn bản chính sách (≥ 3 file)

**Nhiệm vụ:** Tìm và tải **ít nhất 3** file PDF/DOCX về chính sách đại học, lưu vào `data/landing/legal/`.

**Gợi ý nguồn (RMIT):**
- Học phí & thanh toán (Tuition Fees)
- Học bổng (Scholarship)
- Ký túc xá / chỗ ở (Accommodation)
- Đăng ký học phần (Course Registration)

**Đặt tên file rõ ràng**, ví dụ: `tuition-fees-rmit.pdf`, `academic-achievement-scholarship-rmit.pdf`

**Pass khi:** Có ≥ 3 file trong `data/landing/legal/` → pytest `test_task1_*`

---

### Task 2 — Crawl bài viết / thông báo (≥ 5 bài)

**Nhiệm vụ:** Crawl **ít nhất 5** bài về tin tức, sự kiện, thư viện, hỗ trợ sinh viên... Lưu JSON vào `data/landing/news/`.

**Công cụ:** Crawl4AI + Playwright Chromium

```bash
pip install crawl4ai
playwright install chromium
```

Mỗi file JSON cần có: `url`, `title`, `date_crawled`, `content_markdown`.

**Lưu ý:** Một số trang trường chặn bot (HTTP 403) — không phải lỗi của bạn. Đổi nguồn khác hoặc dùng dữ liệu mẫu có sẵn trong repo.

**Pass khi:** Có ≥ 5 file trong `data/landing/news/` → pytest `test_task2_*`

---

### Task 3 — Convert sang Markdown

**Nhiệm vụ:** Dùng **MarkItDown** (Microsoft) chuyển toàn bộ file trong `data/landing/` sang `.md` trong `data/standardized/`.

```bash
pip install "markitdown[pdf]"
python src/task3_convert_markdown.py
```

**Vì sao cần bước này?** PDF có nhiều "rác" định dạng. Markdown giữ cấu trúc tiêu đề (`#`, `##`) sạch sẽ, dễ chunk và dễ cho LLM đọc.

**Pass khi:** Có file `.md` tương ứng trong `data/standardized/` → pytest `test_task3_*`

---

### Task 4 — Chunking & Indexing (ChromaDB)

**Nhiệm vụ:**
1. Đọc markdown từ `data/standardized/`
2. Chọn chiến lược chunk (RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter...)
3. Embed bằng model (khuyến nghị `BAAI/bge-m3`)
4. Lưu vào ChromaDB tại `chroma_db/`, collection `university_services_docs`

**Trong code, bạn phải giải thích (comment):**
- Chunk size / overlap bao nhiêu? Vì sao?
- Embedding model nào? Dimension bao nhiêu?

**Mẹo:** Đổi dữ liệu → xóa `chroma_db/` cũ trước khi index lại, tránh kết quả rác.

**Pass khi:** Vector store có data → pytest `test_task4_*`

---

### Task 5 — Semantic Search (Dense Retrieval)

**Nhiệm vụ:** Viết hàm `semantic_search(query, top_k)` trả về danh sách chunks kèm **cosine score**, sắp xếp giảm dần.

```python
# Output mẫu
[{"content": "...", "score": 0.82, "metadata": {...}}, ...]
```

**Ý nghĩa:** Tìm theo **ngữ nghĩa** — "học phí" vẫn match "chi phí đào tạo".

**Bonus (+5đ):** Implement HyDE (sinh câu trả lời giả định rồi embed để search).

**Pass khi:** Format đúng, sorted → pytest `test_task5_*`

---

### Task 6 — Lexical Search (BM25)

**Nhiệm vụ:** Viết hàm `lexical_search(query, top_k)` dùng **BM25**.

**Ý nghĩa:** Tìm theo **từ khóa chính xác** — tên chương trình, mã quy định, tên riêng.

**Bonus (+5đ):** Dùng TF-IDF / Elasticsearch / Weaviate và giải thích trong demo.

**Pass khi:** Format đúng → pytest `test_task6_*`

---

### Task 7 — Reranking

**Nhiệm vụ:** Viết hàm `rerank(query, candidates, top_k)` để chấm lại độ liên quan.

**Lựa chọn phổ biến trong lab:**
- RRF (Reciprocal Rank Fusion) — gộp thứ hạng Semantic + BM25
- Cross-encoder (Jina, Qwen...)
- MMR (tăng diversity)

**Pass khi:** Output được sắp xếp lại → pytest `test_task7_*`

---

### Task 8 — PageIndex (Vectorless Fallback)

**Nhiệm vụ:** Đăng ký [PageIndex](https://pageindex.ai/), upload tài liệu, viết `pageindex_search(query, top_k)`.

**Vì sao cần?** Khi câu hỏi mang tính **tổng hợp cả chương/mục** (vd: "Tóm tắt quy trình xin học bổng?"), chunk 800 ký tự có thể mất bức tranh toàn cảnh. PageIndex đọc theo cấu trúc mục lục.

**Pass khi:** Query trả về kết quả → pytest `test_task8_*`

---

### Task 9 — Retrieval Pipeline hoàn chỉnh

**Nhiệm vụ:** Viết `retrieve(query, top_k, score_threshold)` nối tất cả module:

```
Query
 ├─ Semantic Search ──┐
 ├─ Lexical Search  ──┼─→ Merge (RRF) → Rerank → Kết quả
 └─ Nếu cosine score top-1 < threshold → PageIndex Fallback
```

**BẪY QUAN TRỌNG (hay mất điểm):**

Điểm RRF sau khi fuse chỉ ~0.016 (phụ thuộc thứ hạng, không phản ánh độ liên quan thật).  
→ So `score_threshold` với **cosine score gốc** từ `semantic_search`, **KHÔNG** so với điểm RRF.

```python
# ĐÚNG
if dense_results[0]["score"] < score_threshold:
    # fallback PageIndex

# SAI — fallback gần như không bao giờ chạy
if rrf_results[0]["score"] < 0.48:
    ...
```

**Pass khi:** Pipeline + fallback hoạt động → pytest `test_task9_*`

---

### Task 10 — Generation có Citation

**Nhiệm vụ:**
1. **Reorder chunks** (`front + back[::-1]`) — chống *lost in the middle*
2. Inject context vào prompt
3. Gọi LLM, bắt buộc trích dẫn `[Nguồn, Năm]`
4. Không đủ evidence → trả `"I cannot verify this information"`

**Pass khi:** Có citation + reorder → pytest `test_task10_*`

---

## 4. Chấm điểm tổng thể

| Thành phần | Điểm | Ghi chú |
|-----------|------|---------|
| **Cá nhân (theo role)** | **50%** | Pytest task được giao + report cá nhân |
| **Bài nhóm** | **30%** | Chatbot demo + eval pipeline |
| **Bonus** | **20%** | HyDE, deploy, UI... (cả nhóm hoặc cá nhân) |

- **Solo (tự học):** pass **35/35 test** ≈ full 50đ cá nhân.
- **Theo nhóm (khuyến nghị):** pass test **đúng task role** + report → map vào 50đ (vd: owner Task 4+5 ≈ 7+6 = 13đ task + report).

Test mẫu domain K3: `"university tuition policy"`, `"What is the tuition fee at RMIT Vietnam?"`

---

## 5. Bài nhóm

Nhóm 4–6 người chọn **1 trong 2**:

### Option A: RAG Chatbot (Streamlit)

- Giao diện chat trong `app.py` (starter có sẵn 🎓 University Services)
- Kết nối Task 9 + Task 10
- Hiển thị source documents, hỗ trợ follow-up

**Câu hỏi gợi ý có sẵn trong app:**
- "Học phí tại RMIT Vietnam là bao nhiêu?"
- "Điều kiện xin học bổng Academic Achievement?"
- "Cách đăng ký học phần qua myRMIT?"

### Option B: RAG Evaluation Pipeline

- Tạo `golden_dataset.json` (≥ 15 cặp Q&A)
- Chạy DeepEval / RAGAS / TruLens
- So sánh A/B (có rerank vs không, hybrid vs dense-only)
- Viết `results.md`

**Ví dụ câu hỏi golden dataset K3:**
- "Học phí hàng năm chương trình Business tại RMIT Vietnam?"
- "Trường có ký túc xá trong khuôn viên không?"

---

## 6. Lộ trình thời gian

| Giai đoạn | Thời gian | Mục tiêu nhóm | Gợi ý ai làm trước |
|-----------|-----------|---------------|---------------------|
| **Setup + chia task** | 0:00–0:15 | Họp chia việc, ghi README, cài môi trường | Cả nhóm |
| **Data** | 0:15–0:45 | Task 1–3 xong, có `.md` trong `standardized/` | Owner Task 1–3 |
| **Index & search** | 0:45–1:30 | Task 4–6, có `chroma_db/`, search chạy được | Owner Task 4–6 (song song khi data xong) |
| **Retrieval nâng cao** | 1:30–2:00 | Task 7–8 | Owner Task 7–8 |
| **Ghép pipeline** | 2:00–2:20 | Task 9 (+ Task 10 nếu chưa ai làm) | Owner Task 9 |
| **Product & eval** | 2:20–2:40 | `app.py`, golden dataset, report cá nhân | Owner Task 10 / eval |
| **Demo** | 2:40+ | Thuyết trình, Q&A | Cả nhóm — mỗi người nói phần mình |

Các bạn **tự quyết** ai nhận giai đoạn nào — bảng trên chỉ là **thứ tự kỹ thuật**, không phải gán role.

---

## 7. Lỗi thường gặp

| Lỗi | Cách sửa |
|-----|----------|
| `MissingDependencyException` khi convert PDF | `pip install "markitdown[pdf]"` |
| `Executable doesn't exist` khi crawl | `playwright install chromium` |
| `UnicodeEncodeError` trên Windows | `$env:PYTHONIOENCODING="utf-8"` |
| Fallback không bao giờ chạy | So cosine gốc, không so RRF |
| Kết quả search lẫn dữ liệu cũ | Xóa `chroma_db/` rồi chạy lại Task 4 |
| Rate limit khi chạy RAGAS | Giảm số câu trong golden dataset khi test |

---

## 8. Lưu ý cuối

Nếu bạn học **track 3 giai đoạn 2**, repo này sẽ được phát triển tiếp lên **Knowledge Graph** để xử lý câu hỏi phức tạp hơn. Đừng xóa sau khi nộp bài.

---

## 9. Checklist trước khi nộp

**Cá nhân (mỗi bạn):**
- [ ] Đã **cùng nhóm thống nhất** task mình owner (ghi trong README)
- [ ] Pytest **task của mình** pass
- [ ] Đã nộp **report cá nhân** (1–2 trang)
- [ ] Demo được: giải thích phần mình + 1 câu pipeline tổng

**Cả nhóm:**
- [ ] `group_project/README.md` có bảng phân công
- [ ] `data/` + `chroma_db/` đủ (owner data/vector lo)
- [ ] `pytest tests/test_individual.py -v` pass trên repo chung
- [ ] `app.py` + eval chạy được
- [ ] `.env` / API keys team dùng chung (không commit secret)

---

**Chúc các bạn làm lab vui! Tự chia task cho hợp lý, report rõ phần mình — không cần ôm hết 10 task.**

```bash
# Chạy test đúng task bạn phụ trách, vd Task 5:
pytest tests/test_individual.py::TestTask5 -v
```
