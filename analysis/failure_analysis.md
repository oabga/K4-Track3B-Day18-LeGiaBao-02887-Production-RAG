# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Le Gia Bao
**MSSV:** 02887
**Khóa:** K4 - Track 3B

*(Số liệu trong file này khớp với `reports/ragas_report.json` — kết quả từ lần chạy `python src/pipeline.py` cuối cùng.)*

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8008 | 0.8000 | -0.0008 |
| Answer Relevancy | 0.7611 | 0.8212 | +0.0601 |
| Context Precision | 0.9250 | 0.9333 | +0.0083 |
| Context Recall | 0.9250 | 0.7917 | -0.1333 |

**Nhận xét tổng quan:** Production (hybrid search + rerank + enrichment) thắng naive ở 2/4 metric, hoà ở faithfulness, và thua rõ nhất ở `context_recall` (-0.1333). Cả 4 metric production đều ≥0.79 (≥0.70 lẫn ≥0.75). Điểm RAGAS dao động khá nhiều giữa các lần chạy (đã quan sát 2 lần chạy pipeline với answer_relevancy lệch từ 0.61 → 0.82) vì OpenAI chat completion không fix seed/temperature=0 — bản thân đây cũng là một insight quan trọng về độ ổn định (reproducibility) của RAG pipeline dùng LLM generation, không phải lỗi code.

---

## Bottom-5 Failures

*(Trích từ `reports/ragas_report.json` → `failures`. Context được xác minh lại bằng cách scroll trực tiếp collection Qdrant `lab18_production` đã index — BM25/Dense/Rerank đều deterministic nên context tái hiện chính xác 100%; riêng câu trả lời LLM có thể lệch nhẹ so với lần được RAGAS chấm điểm gốc do temperature > 0, được ghi chú rõ ở từng case.)*

### #1
- **Question:** "Bao lâu phải đổi mật khẩu một lần?"
- **Expected:** "Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế."
- **Got (tái hiện):** "Mật khẩu phải được thay đổi mỗi 90 ngày." *(chọn nhầm version cũ)*
- **Worst metric:** faithfulness = 0.0
- **Error Tree:**
  - Output sai? → Có — trả lời theo chính sách **đã bị thay thế** (v1, 90 ngày) thay vì hiện hành (v2.0, 120 ngày).
  - Context đúng? → Có đủ dữ kiện nhưng **mâu thuẫn**: 3 chunk trả về gồm cả bản ghi "90 ngày" (v1), "120 ngày" (v2.0), và 1 chunk ghi chú "...đã được thay thế bởi Chính sách mật khẩu v2.0 từ ngày 01/07/2024" — nhưng chunk ghi chú này KHÔNG nói rõ "90 ngày" chính là con số bị thay thế, nên LLM không nối được 2 sự kiện lại với nhau.
  - Query OK? → Có
  - → **Root cause:** Giống hệt pattern đã thấy ở lần chạy trước (xem thêm Case Study) — hệ thống retrieve cả tài liệu cũ lẫn mới mà không có tín hiệu metadata rõ ràng để loại bỏ version cũ, nên LLM đôi khi chọn nhầm (lần này), đôi khi từ chối trả lời (lần chạy trước) — **cả 2 lần chạy đều fail ở đúng câu hỏi này**, chứng tỏ đây là lỗi thiết kế ổn định (systematic), không phải nhiễu ngẫu nhiên.
- **Suggested fix:** Thêm metadata `superseded_by` / `effective_date` ở bước M5 enrichment cho các cặp tài liệu v1/v2 (mat_khau_v1.md, mat_khau_v2.md, nghi_phep_nam_v2023.md, v2024.md...), lọc bỏ chunk version cũ khỏi context trước khi đưa vào prompt sinh câu trả lời.

### #2
- **Question:** "Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?"
- **Expected:** "KHÔNG. Nhân viên tuyệt đối không được tự ý xử lý malware. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT. Tự ý xử lý bị coi là vi phạm nghiêm trọng."
- **Got (tái hiện):** "Không, nhân viên không nên tự ý xử lý malware hoặc tìm cách khắc phục mà không có sự hướng dẫn của đội CNTT."
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:**
  - Output sai? → Không hẳn — ý chính ("KHÔNG tự xử lý") **đúng và có căn cứ** trong context.
  - Context đúng? → Có, chunk `bao_mat_su_co.md` chứa đủ thông tin, kể cả chi tiết "báo cáo trong 1 giờ" (nằm trong context nhưng answer không nhắc tới).
  - Query OK? → Có
  - → **Root cause:** Giống case #3 trong phân tích lần trước — answer đúng nhưng **thiếu chi tiết phụ** (thời hạn báo cáo, kênh báo cáo). RAGAS `answer_relevancy` sinh câu hỏi ngược từ answer rồi so embedding với câu hỏi gốc; answer càng rút gọn thì càng dễ bị chấm thấp dù nội dung không sai. Đây là điểm yếu đã biết của metric này với câu trả lời ngắn/cụt, không phải lỗi retrieval.
- **Suggested fix:** Prompt LLM yêu cầu liệt kê **toàn bộ** hành động/thời hạn cụ thể xuất hiện trong context (không chỉ kết luận đúng/sai), ví dụ thêm hướng dẫn "Nếu context có số liệu/thời hạn cụ thể, phải trích dẫn đầy đủ trong câu trả lời."

### #3
- **Question:** "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** "Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày)."
- **Got (tái hiện, nhất quán với lần chạy trước):** "...2% của 15.000.000 VNĐ = 300.000 VNĐ. Nhân viên sẽ bị phạt 300.000 VNĐ."
- **Worst metric:** faithfulness = 0.4
- **Error Tree:**
  - Output sai? → Có một phần — tính đúng công thức "2% × 15 triệu" nhưng áp dụng **nguyên 1 tháng phí** cho 5 ngày trễ thay vì pro-rata (300.000 × 5/30 ≈ 50.000 VNĐ).
  - Context đúng? → Có, chunk `tam_ung.md` nêu "quá hạn 15 ngày → phí 2%/tháng trên số tiền chưa hoàn ứng" nhưng **không nói rõ** cách tính khi trễ chưa đủ 1 tháng.
  - Query OK? → Có
  - → **Root cause:** Context thiếu rule pro-rata tường minh → LLM suy luận theo hướng đơn giản nhất (tính tròn tháng). Đây là lỗi **lặp lại giống hệt ở cả 2 lần chạy pipeline** (context tái hiện y hệt vì retrieval deterministic), chứng tỏ là lỗi hệ thống ở nguồn dữ liệu/prompt, không phải ngẫu nhiên.
- **Suggested fix:** Bổ sung rule pro-rata rõ ràng ngay trong `tam_ung.md` (ví dụ: "phí tính theo số ngày trễ / 30 ngày"), hoặc thêm field `calculation_rule` khi enrichment, đồng thời thêm few-shot ví dụ tính phí trễ hạn theo ngày vào system prompt cho câu hỏi dạng numeric.

### #4
- **Question:** "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"
- **Expected:** "Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt."
- **Got (tái hiện):** "Cần thêm phê duyệt của Kế toán trưởng." *(sai — lấy nhầm từ bảng phê duyệt của chính sách tạm ứng)*
- **Worst metric:** context_recall = 0.0
- **Error Tree:**
  - Output sai? → Có, trả lời sai hoàn toàn người phê duyệt.
  - Context đúng? → **Sai** — context #1 là bảng phê duyệt `mua_sam.md` nhưng bị cắt chunk ngay sau dòng "Dưới 5.000.000 VNĐ", **không có dòng tương ứng mức >50 triệu (CEO)**; context #3 lại là bảng phê duyệt của `tam_ung.md` ("Từ 5.000.000 VNĐ trở lên: cần thêm phê duyệt Kế toán trưởng") — một bảng **hoàn toàn khác chính sách**, nhưng do hình thức giống nhau (đều là bảng ngưỡng tiền → người duyệt) nên bị reranker xếp vào top-3 và LLM nhầm lẫn lấy nhầm nguồn.
  - Query OK? → Có
  - → **Root cause:** (1) Bảng markdown nhiều dòng trong `mua_sam.md` bị cắt ngang bởi chunking theo ký tự, mất đúng dòng chứa đáp án (>50 triệu → CEO); (2) reranker đánh giá theo similarity bề mặt ("ngưỡng tiền → phê duyệt") nên lẫn giữa 2 tài liệu khác nhau có cấu trúc câu tương tự.
- **Suggested fix:** Với bảng markdown, nên chunk theo **toàn bộ bảng** (không cắt giữa các dòng của cùng 1 table) — có thể detect bảng bằng regex `^\|.*\|$` và giữ nguyên cả khối; đồng thời thêm `source` vào câu trả lời cuối cùng (citation) để dễ phát hiện khi LLM trộn nhầm nguồn giữa 2 tài liệu.

### #5
- **Question:** "Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"
- **Expected:** "Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng."
- **Got (tái hiện):** "Lương thử việc của nhân viên Junior (P1-P2) là 85% của mức lương cao nhất là 20.000.000 VNĐ, tức là 17.000.000 VNĐ." *(đúng!)*
- **Worst metric:** faithfulness = 0.0 (theo report gốc)
- **Error Tree:**
  - Output sai? → **Không rõ** — khi tái hiện lại với context y hệt (deterministic), câu trả lời tái hiện lại **đúng hoàn toàn**, khớp ground truth. Điều này cho thấy câu trả lời bị chấm faithfulness=0.0 trong lần eval gốc nhiều khả năng là một lần sinh câu trả lời sai/hallucinate ngẫu nhiên (ví dụ nhầm con số) do OpenAI API không set `temperature=0`/seed cố định, chứ không phải lỗi retrieval — context luôn đúng và đủ (chunk `bang_luong_2024.md` có rõ cả bảng lương và công thức 85%).
  - Context đúng? → Có, đầy đủ.
  - Query OK? → Có
  - → **Root cause:** Giới hạn **reproducibility** của sinh câu trả lời bằng LLM có temperature mặc định — cùng 1 context, 1 câu hỏi, có thể ra câu trả lời khác nhau giữa các lần gọi API.
- **Suggested fix:** Set `temperature=0` (hoặc thấp, ví dụ 0.1) khi gọi `client.chat.completions.create()` trong `run_query()` (`src/pipeline.py`) cho tác vụ tra cứu chính sách — nơi cần độ chính xác/nhất quán cao hơn là tính sáng tạo, giúp giảm biến thiên điểm RAGAS giữa các lần chạy.

---

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Bao lâu phải đổi mật khẩu một lần?" (version-conflict, fail ở **cả 2 lần chạy pipeline khác nhau** — refuse ở lần 1, chọn nhầm version ở lần 2)

**Error Tree walkthrough:**
1. Output đúng? → Sai ở cả 2 lần chạy (lần 1: "Không tìm thấy" dù có đáp án; lần 2: chọn nhầm 90 ngày thay vì 120 ngày)
2. Context đúng? → Chunk đúng tài liệu nguồn, cả 90 ngày lẫn 120 ngày đều có mặt trong top-3, kèm 1 chunk ghi chú version đã bị thay thế — nhưng **không đủ rõ ràng** để LLM tự tin chọn đúng version hiện hành.
3. Query rewrite OK? → Có, không có vấn đề ở bước retrieval-query.
4. Fix ở bước: **M5 Enrichment / M1 Metadata** — cần gắn rõ field version/ngày hiệu lực ngay từ bước enrichment, lọc chunk version cũ *trước khi* đưa vào context, thay vì để LLM tự suy luận từ văn bản mô tả chung chung.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm metadata `superseded_by`/`effective_date` ở `extract_metadata()`/`_enrich_single_call()`, lọc version cũ khỏi context trước khi sinh câu trả lời — giải quyết dứt điểm nhóm lỗi version-conflict (case #1, lặp lại ổn định ở cả 2 lần chạy).
- Set `temperature=0` cho bước generate câu trả lời cuối (`run_query()`) để giảm biến thiên ngẫu nhiên giữa các lần chạy (case #5) và giúp kết quả RAGAS ổn định, dễ so sánh hơn khi thử nghiệm thay đổi pipeline.
- Chunk các bảng markdown (approval table) theo nguyên khối bảng thay vì cắt theo ký tự, tránh mất dòng dữ liệu quan trọng và tránh nhầm lẫn giữa 2 tài liệu có cấu trúc bảng giống nhau (case #4).
- Prompt yêu cầu LLM trích dẫn đầy đủ mọi chi tiết/số liệu phụ xuất hiện trong context liên quan câu hỏi, giảm answer bị "rút gọn" khiến answer_relevancy bị chấm thấp oan (case #2).
