# Thành viên nhóm

## Kiến trúc hệ thống

```mermaid
flowchart LR
    U[Người dùng] --> UI[Streamlit UI<br/>app.py]
    UI --> T10[Task 10<br/>Generation + Citation]
    T10 --> T9[Task 9<br/>Retrieval Pipeline]

    T9 --> T5[Task 5<br/>Semantic Search]
    T9 --> T6[Task 6<br/>BM25 Search]
    T5 --> T7[Task 7<br/>RRF / Reranking]
    T6 --> T7
    T9 -. điểm semantic thấp .-> T8[Task 8<br/>PageIndex Fallback]

    T5 --> V[(ChromaDB<br/>Vector Index)]
    T6 --> D[(Markdown<br/>Corpus)]
    T8 --> P[(PageIndex)]
    T7 --> T10
    T10 --> LLM[OpenAI / OpenRouter LLM]
    LLM --> A[Câu trả lời<br/>+ nguồn trích dẫn]
    T10 --> A
```

## Phân công thành viên

| Thành viên | Mã học viên | Nhiệm vụ |
|---|---|---|
| Nhữ Trọng Thành | 2A202601977 | Task 9, 10 |
| Mai Hồng Sơn | 2A202601921 | Task 5, 6 |
| Lê Thị Linh | 2A202601441 | Task 3, 4 |
| Vũ Thu Huyền | 2A202601583 | Task 1, 2 |
| Lường Thị Hảo | 2A202601637 | Task 7, 8 |
