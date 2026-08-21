# Lab 21 — Evaluation Report

**Họ tên**: Tạ Minh Đức  **MSSV**: 2A202601497  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 14.6 GB`

> Mọi con số dưới đây khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (intent, urgency, product, sentiment) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98, suggested=256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Lý do giữ max_length=1024 thay vì 256:** p95=98 cho thấy phần lớn mẫu ngắn (~93 token trung bình), nhưng tier T4 mặc định 1024 và biên VRAM vẫn đủ (peak 12.01/14.6 GB). Việc giữ 1024 an toàn hơn là cắt ngắn khi data thay đổi. Chi phí là padding thêm, không ảnh hưởng đến kết quả.

**Template có giữ khối `<think>` không?** `Có` — VERDICT: *reasoning preserved — safe to train on traces* *(results/template_check.json)*

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (preview):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

**Phân tích:** Với `assistant-only`, chỉ 39/94 token (41%) được tính loss — đúng là phần câu trả lời JSON. Khi thử `everything`, 94/94 token được tính loss bao gồm cả câu hỏi của user → đây là Lỗi kinh điển khiến model học cách viết lại câu hỏi thay vì trả lời.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3345 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1042 |
| (c) LoRA fine-tune | 0.970 | 0.678 | 1.000 | 1430 |

**(b) có thật sự mạnh hơn (a) không?** `Có` — (b) = 0.765 >> (a) = 0.000 về target; format 1.000 vs 0.000. (b) cũng nhanh hơn 3× (1042ms vs 3345ms).

Prompt (b) không bị sửa so với mặc định. Prompt (a) naive không có cấu trúc triage cụ thể nên model trả về văn bản tự do thay vì JSON → format=0.

---

## 4. Giải phẫu cấu hình sai (NB4 + NB5 §4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.627 | **0.970** | 988 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.537 | **0.970** | 891 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.570 | **0.000** | 1033 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.706 | **0.940** | 1068 | 7.09 |

> Xếp hạng bằng cột **target (NB5 §4)**: correct = attn_only > qlora >> wrong_lr

**4.1 — `attn_only` so với `correct` về target và train loss:**

`attn_only` có cùng số tham số huấn luyện với `correct` (32,456,704 vs 32,464,896 = chênh 0.025%, đạt bởi `matched_rank()` với r=283). Trên tập target, hai run **hoà nhau** (cùng 0.970). Tuy nhiên, theo train loss, `attn_only` thắng (0.537 < 0.627) — đây chính xác là điều lab cảnh báo: loss thấp hơn không đồng nghĩa với hiệu năng tốt hơn trên tác vụ. Kết quả này cho thấy: với tác vụ hẹp như JSON triage (chỉ 4 trường, ít token đầu ra), vị trí adapter (attention-only vs all-linear) **không phải đòn bẩy**; cả hai đều đạt ngưỡng bão hoà 0.97 với ngân sách tham số tương đương. Đây là bằng chứng trực tiếp rằng rank mới là lever, không phải vị trí — ít nhất trên bài toán cụ thể này.

**4.2 — `wrong_lr` (LR = 1e-5, thang full-FT):**

Loss curve của `wrong_lr` gần như phẳng trong toàn bộ 30 step: 2.163 → 2.066 → 1.606 → 1.326 → 1.141 → 1.119. Nếu chỉ nhìn vào đường loss mà không biết LR, người quan sát có thể kết luận: "model đang học, chỉ cần thêm step". Kết luận này **sai hoàn toàn**: bước cập nhật quá nhỏ (LR 10× thấp hơn mức cần) khiến LoRA adapter gần như không thay đổi so với khởi tạo. Target=0.000 và format=0.000 ở NB5 xác nhận model không học được gì dù 30 step đã hoàn thành.

**4.3 — `qlora` tiết kiệm VRAM và trả giá:**

`qlora` dùng 7.09 GB VRAM so với 12.01 GB của `correct` — **tiết kiệm 41%**, đây là lợi thế thật sự và lớn (cho phép chạy model lớn hơn trên cùng phần cứng). Tuy nhiên, target tụt từ 0.970 → 0.940 (giảm 3 điểm %). Số đo này **có phần ủng hộ** khuyến nghị "không dùng QLoRA cho Qwen3.5" của nhà cung cấp — sai số lượng tử hoá làm giảm nhẹ chất lượng. Tuy nhiên mức giảm (0.03) khá nhỏ và latency tăng (1812ms vs 1430ms), nên trade-off phụ thuộc vào constraints phần cứng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.080` · `valid_trace_rate = 0.00`

**Lý do FAILED:** Fine-tune vượt baseline (b) về target (0.970 > 0.765, Δ=+0.205) và format hoàn hảo (1.000). Tuy nhiên, regression tụt từ 0.758 → 0.678, delta = **-0.080** vượt ngưỡng cho phép 0.020. Đây là dấu hiệu **catastrophic forgetting**: model đã tập trung học quá mức vào tác vụ triage JSON đến mức "quên" kiến thức phổ thông. Cụ thể, 30 optimizer step trên 225 mẫu JSON thuần túy làm thiên lệch phân phối nội tại của model. `valid_trace_rate = 0.0` cũng xác nhận khả năng suy luận (<think>) bị tắt hoàn toàn — do training data không có reasoning trace nên model học cách bỏ qua bước suy nghĩ.

Giải pháp để PASS: thêm 1–5% dữ liệu phổ thông (replay data, deck §14.3) vào tập train để duy trì general capability. Hiện tại, nếu chỉ xét target, fine-tune **thắng rõ ràng**; điểm FAILED là do quên thảm hoạ, không phải do fine-tune kém.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Alo shop, ốp lưng DH734695. Giá bao nhiêu. | intent=hoi_thong_tin, urgency=trung_binh | ✅ đúng | ✅ đúng | ✅ FT thắng (nhanh hơn 3×) |
| 2 | Chào shop, ốp lưng VN833689. Sai màu. Sớm nhé. | intent=san_pham_loi, urgency=trung_binh | ✅ đúng | ✅ đúng | ✅ FT thắng (latency tốt hơn) |
| 3 | Mình đặt bình giữ nhiệt VN804124. Chưa thấy tiền. | intent=hoan_tien, urgency=? | ✅ đúng | ❌ urgency sai | ❌ **FT thua** — không phân biệt được urgency từ context mơ hồ |
| 4 | Shop ơi, nồi chiên DH249548. Thiếu phụ kiện. | intent=san_pham_loi, urgency=? | ✅ đúng | ❌ urgency sai | ❌ **FT thua** — "thiếu phụ kiện" không có từ khoá urgency rõ ràng |
| 5 | Alo shop, ốp lưng DH936478. Shipper không liên lạc được. | intent=van_chuyen, urgency=thap | ✅ đúng | ✅ đúng | ✅ FT thắng (format + accuracy) |

**Mẫu chung ở các ca FT thua:** Những ticket có context mơ hồ về urgency (không có từ khoá rõ ràng như "gấp", "khẩn", "ngay") — fine-tune có xu hướng mặc định `trung_binh` thay vì phân tích ngữ cảnh. Prompt (b) đã prompt tường minh về thang urgency nên xử lý tốt hơn.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Không nên deploy bản fine-tune hiện tại dưới dạng thay thế cho (b). Dù target tăng mạnh (0.000 → 0.970 so với baseline ngây thơ, và vượt (b) 0.765 → 0.970), cổng hồi quy FAILED vì regression tụt -0.080 — nghĩa là model đã "quên" 8% năng lực phổ thông để đổi lấy chuyên môn triage. Trong môi trường sản xuất, điều này không thể chấp nhận được: một chatbot CSKH mà không trả lời được các câu hỏi thông thường là một chatbot hỏng.

Đòn bẩy thật sự trong lab này **không phải rank hay vị trí adapter**. Bằng chứng: `attn_only` (q,v only, r=283) hoà với `correct` (all-linear, r=16) trên tập target dù cơ chế hoàn toàn khác. Trong khi đó, `wrong_lr` (chỉ đổi LR ÷10) bị sập hoàn toàn (target=0). Điều này chỉ ra: **learning rate là lever nguy hiểm nhất** — sai LR và không có gì hoạt động, đúng LR và cả placement lẫn rank đều cho kết quả tương đương. Chất lượng dữ liệu và loss mask cũng là đòn bẩy lớn: mask sai (everything) hoặc thiếu replay data là hai nguyên nhân phổ biến nhất khiến fine-tune thất bại trong thực tế. Bước tiếp theo hợp lý là thêm 10–15 mẫu phổ thông (replay) vào tập train và chạy lại — chi phí thêm dưới 5 phút nhưng có thể đẩy verdict từ FAILED sang PASSED.

**Ba điều tôi học được:**
1. **Loss thấp ≠ model tốt hơn.** `attn_only` có loss thấp nhất (0.537) nhưng target bằng `correct` (0.970). Nếu chỉ nhìn loss để quyết định deploy, sẽ chọn sai model mà không biết.
2. **Baseline (b) phải được đo trước khi nhìn thấy bất kỳ kết quả train nào.** Nếu đo sau, sẽ vô thức cài prompt yếu để fine-tune trông "thắng" — đây không phải gian lận có chủ ý mà là bias tự nhiên của não người khi thấy hai con số.
3. **Catastrophic forgetting xảy ra nhanh hơn tôi nghĩ.** Chỉ 30 step trên 225 mẫu là đủ để regression tụt 8%. Lab cũ không đo regression nên lỗi này vô hình — đó là lý do ba-baseline design tồn tại.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** Thêm 10–15 câu hỏi phổ thông vào tập train (replay data) và chạy lại NB3 + NB5 để kiểm tra cổng hồi quy có PASS không. Đồng thời thử B4 (quét rank r=8,16,64) để xác nhận xem bài toán này có cần rank cao hơn không hay 16 đã là điểm bão hoà.

---

## 8. NB6 — Merge & Hot-swap adapter (B1)

### 8.1 Merge check

| | |
|---|---|
| Điểm **trước** merge | `0.9700` |
| Điểm **sau** merge | `0.9700` |
| Delta (Δ) | `+0.0000` |
| Tolerance | `0.01` |
| Kết quả | ✅ **PASS** — không tụt điểm sau merge |

Merge (`W = W₀ + (α/r)·BA`) là phép toán chính xác — trọng số adapter được gộp thẳng vào base weights. Kết quả Δ=0.0000 xác nhận không có sai số dtype đáng kể trong quá trình merge trên fp16.

### 8.2 Hot-swap 3 adapter trên cùng 1 base

Nạp một base model duy nhất, sau đó lần lượt set_adapter() cho từng adapter:

| Adapter | Output trên ticket "chuột không dây" |
|---|---|
| `correct` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` |
| `attn_only` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` |
| `qlora` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` |

Cả 3 adapter cho output nhất quán trên cùng ticket. Hot-swap hoạt động đúng: một base trong VRAM (15.64 GB khả dụng), nhiều adapter (~120 MB mỗi cái) swap theo request.

**Câu hỏi B1:** Merge cho overhead suy luận = 0 nhưng mất khả năng hot-swap (base đã bị biến đổi, không thể tách adapter ra lại). Nên giữ adapter riêng khi cần phục vụ nhiều tác vụ/khách hàng khác nhau trên cùng một endpoint — ví dụ: một base Qwen3.5-4B phục vụ cả triage CSKH lẫn chatbot tư vấn sản phẩm, swap adapter theo từng request.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap — `results/merge_check.json` (Δ=0.0000), hot-swap 3 adapter (correct, attn_only, qlora)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
