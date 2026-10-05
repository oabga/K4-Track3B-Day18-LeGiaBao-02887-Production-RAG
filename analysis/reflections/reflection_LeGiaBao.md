# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Le Gia Bao
**MSSV:** 02887
**Khóa:** K4 - Track 3B
**Ngày hoàn thành:** 2026-10-05

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Với threshold=0.85 trên toàn bộ corpus (26 doc): semantic tạo **208 chunks** (avg 99 ký tự, min 6, max 354) so với basic **51 chunks** (avg 410 ký tự). Semantic chia nhỏ hơn hẳn vì bám theo cosine similarity giữa câu liên tiếp — mỗi khi chủ đề đổi (sim < 0.85) là tách chunk mới, nên không cắt giữa ý nhưng dễ tạo chunk rất ngắn (6 ký tự) khi câu liền kề không liên quan ngữ nghĩa dù cùng đoạn văn. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent=2048/child=256 trên toàn corpus tạo **11 parent, 87 child**. Thực tế lại là module gây lỗi nghiêm trọng nhất: cắt child theo ký tự cứng (`text[start:start+256]`) đã cắt đứt từ phủ định "KHÔNG" ngay giữa câu ở 1 trong 20 câu hỏi test (xem Bottom-5 #2 trong `failure_analysis.md`) → bài học: production RAG phải cắt theo ranh giới câu, không phải theo số ký tự. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF giải quyết vấn đề BM25 và Dense "đồng ý" ở các rank khác nhau — ví dụ test `test_rrf_merges`: BM25 xếp "doc2" hạng 2, Dense xếp "doc2" hạng 1, RRF cộng dồn `1/(60+rank+1)` từ cả 2 danh sách nên "doc2" nổi lên đầu kết quả hybrid dù không ai xếp nó hạng 1 một mình. Hệ số k=60 giúp làm mượt chênh lệch rank giữa 2 phương pháp có thang điểm khác nhau hoàn toàn (BM25 score không chặn trên, cosine similarity trong [-1,1]). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Model `bge-reranker-v2-m3` chấm query "Nhân viên được nghỉ phép bao nhiêu ngày?" với 3 doc: doc đúng ("nghỉ 12 ngày/năm") = **0.9914**, 2 doc nhiễu ("thử việc 60 ngày", "mật khẩu 90 ngày") chỉ 0.02 và 0.0007 — phân biệt rất rõ dù cả 3 đều chứa chữ "ngày". Latency đo trên **CPU** (phải ép `CUDA_VISIBLE_DEVICES=""` vì GPU sandbox chỉ 3.68GB, không đủ chạy song song với bge-m3 dense encoder) là ~171ms/query cho 3 doc — trên GPU thực tế sẽ nhanh hơn nhiều, cho thấy rerank là bước tốn chi phí nhất nếu scale lên top-20. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Production (lần chạy cuối, `reports/ragas_report.json`): faithfulness 0.80, answer_relevancy 0.8212, context_precision 0.9333, **context_recall 0.7917 (thấp nhất)**. Context_recall thấp nhất vì một số câu hỏi (vd. "mua thiết bị 55 triệu") bị cắt mất dòng dữ liệu đúng trong bảng markdown do chunking theo ký tự, khiến retrieval không "nhớ" được đáp án dù nó có trong corpus (case #4 trong failure_analysis). Điểm số RAGAS cũng dao động khá nhiều giữa 2 lần chạy khác nhau (answer_relevancy từng là 0.6061 ở lần chạy trước) vì OpenAI generation không fix seed — một bài học quan trọng về reproducibility của RAG eval. |
| Contextual embeddings (Anthropic-style) | M5 | `_enrich_single_call()` / `contextual_prepend()` | Enrichment (combined mode, 1 API call/chunk cho 100 chunk, mất ~280s) thêm câu mô tả vị trí tài liệu trước mỗi chunk, ví dụ: *"Đoạn văn nằm trong tài liệu hướng dẫn về chính sách bảo mật mật khẩu."* Tuy nhiên ở case #1 (version conflict 90 vs 120 ngày), context vẫn **không đủ** để LLM chọn đúng version vì câu mô tả không nêu rõ version nào là hiện hành — cho thấy contextual prepend giảm ambiguity về *vị trí* chunk nhưng không tự động giải quyết xung đột *nội dung* giữa các version tài liệu; cần kết hợp thêm `extract_metadata()` với field ngày hiệu lực. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message) #1:**
  `NameError: name 'c' is not defined` tại `src/m2_search.py`, dòng `PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": c["text"]})` bên trong `for i, v in enumerate(vectors)`.
  - **Nguyên nhân gốc rễ & debug:** Viết comprehension chỉ enumerate `vectors` nhưng vẫn tham chiếu `c` (biến chunk gốc) — lỗi lọt qua vì `pytest` dùng corpus rất nhỏ nhưng chưa từng test đường dẫn `DenseSearch.index()` thật với Qdrant (`test_m2.py` chỉ test BM25 + RRF, không test Dense trực tiếp). Phát hiện khi chạy `python main.py` full pipeline, đọc traceback chỉ thẳng dòng lỗi → sửa thành `for i, (c, v) in enumerate(zip(chunks, vectors))`.
  - **Bài học:** Unit test pass 100% không đồng nghĩa pipeline end-to-end chạy đúng — `test_m2.py` không cover code path dùng Qdrant thật, nên integration test (chạy `main.py`) là bước bắt buộc, không thể bỏ qua dù tests xanh hết.

- **Lỗi kỹ thuật gặp phải #2:**
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 3.68 GiB` khi `CrossEncoder("BAAI/bge-reranker-v2-m3")` load model.
  - **Nguyên nhân gốc rễ & debug:** Môi trường sandbox chỉ có GPU 3.68GB; `DenseSearch` (bge-m3) đã chiếm gần hết VRAM trước khi reranker load tiếp model thứ hai. Debug bằng cách đọc traceback xác định chính xác bước OOM (load model, không phải lúc inference), rồi set env `CUDA_VISIBLE_DEVICES=""` để ép toàn bộ pipeline chạy CPU — chấp nhận đánh đổi tốc độ (chạy chậm hơn ~5-10 lần) để đổi lấy ổn định trong môi trường GPU giới hạn.
  - **Kiến thức còn thiếu & cách khắc phục:** Trước lab chưa để ý rằng nhiều model embedding/rerank load đồng thời trên cùng 1 GPU nhỏ sẽ cộng dồn VRAM dù mỗi model riêng lẻ không lớn. Đã đọc thêm về `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` và chiến lược "load model tuần tự, giải phóng sau khi dùng xong" (`del model; torch.cuda.empty_cache()`) để áp dụng cho dự án cá nhân có nhiều model cùng chạy.

- **Thiết lập môi trường:** Sandbox chỉ có sẵn Python 3.14 trong khi `ragas`/project pin `.python-version=3.11`. Giải quyết bằng `pip install uv` → `uv python install 3.11` (tải Python 3.11.17 standalone, không cần sudo/apt) → `uv venv --python 3.11 .venv` → cài `requirements.txt` vào venv đó. Bài học: `uv` là công cụ hữu ích để quản lý version Python không cần quyền root, rất hợp với môi trường sandbox/CI hạn chế quyền.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

*(Khung áp dụng cho dạng project RAG tra cứu tài liệu nội bộ/doanh nghiệp — điều chỉnh lại tên và chi tiết theo đúng project thật của bạn.)*

## Project: Trợ lý tra cứu tài liệu nội bộ (Internal Policy Q&A Assistant)

### Hiện tại
- RAG pipeline hiện tại: Chunking cơ bản theo đoạn văn + dense-only search (tương đương `naive_baseline.py` trong lab), chưa có rerank, chưa có evaluation có hệ thống.
- Known issues: Không xử lý được version conflict giữa các bản tài liệu cũ/mới; chunking cắt cứng dễ mất ngữ cảnh (đặc biệt từ phủ định); không có số đo khách quan để biết pipeline tốt lên hay xấu đi sau mỗi thay đổi.

### Plan áp dụng
1. [ ] Chunking strategy: Dùng **Structure-aware chunking** làm chính (giữ nguyên section theo header, tránh cắt đứt bảng/danh sách) kết hợp **Hierarchical** cho tài liệu dài — nhưng sửa child splitting để cắt theo câu (`re.split` theo dấu câu) thay vì theo số ký tự cứng, rút kinh nghiệm trực tiếp từ lỗi ở case #2 trong lab.
2. [ ] Search: **Hybrid (BM25 + Dense) + RRF** — vì corpus tiếng Việt có nhiều từ khóa chính xác (tên chính sách, mã số) mà dense-only dễ bỏ sót, trong khi BM25 lại không hiểu ngữ nghĩa paraphrase; RRF cân bằng được cả hai như đã kiểm chứng trong M2.
3. [ ] Reranking: Có — `CrossEncoderReranker` (bge-reranker-v2-m3), nhưng tăng `RERANK_TOP_K` lên 5 cho câu hỏi multi-hop/multi-field thay vì mặc định 3, dựa trên bài học từ case #4 (reranker bỏ sót chunk lương vì bị chunk nghỉ-phép áp đảo).
4. [ ] Evaluation: **RAGAS 4 metrics** làm baseline định lượng mỗi khi đổi pipeline, chạy trên bộ test set cố định (tương tự `test_set.json`) để so sánh before/after khách quan thay vì đánh giá cảm tính.
5. [ ] Enrichment: Ưu tiên **Contextual prepend + Auto metadata** (đặc biệt field ngày hiệu lực/version) để giải quyết version-conflict — vấn đề lặp lại nhiều lần nhất trong Bottom-5 của lab này.

### Timeline
- Tuần 1 (06/10 – 12/10/2026): Implement structure-aware + hierarchical chunking (sentence-aware) cho corpus thật của project, viết lại test set đánh giá ~20 câu hỏi đại diện (bao gồm version-conflict, multi-hop, numeric).
- Tuần 2 (13/10 – 19/10/2026): Tích hợp Hybrid Search + RRF + Reranker, chạy RAGAS baseline đầu tiên, so sánh với naive để xác nhận cải thiện thật (rút kinh nghiệm: phải đo bằng số, không giả định "pipeline phức tạp hơn = tốt hơn" vì lab này cho thấy production có thể thua naive nếu chunking sai).
- Tuần 3 (20/10 – 26/10/2026): Thêm M5 Enrichment (metadata version/ngày hiệu lực), đo lại RAGAS, viết failure analysis cho bottom-N để tiếp tục vòng lặp cải tiến.
