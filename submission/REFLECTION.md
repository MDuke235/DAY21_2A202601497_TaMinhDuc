# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng đảo ngược giữa loss huấn luyện và năng lực thực tế trên tác vụ: Run `attn_only` (chỉ gắn q,v với rank nâng lên r=283) đạt `train_loss = 0.537` — thấp hơn đáng kể so với `correct` (`train_loss = 0.627`), nhưng khi đưa lên tập test `target` ở NB5 §4 thì cả hai hoàn toàn hoà nhau (cùng đạt `0.970`). Nếu chỉ dừng lại ở loss đồ thị (như lab cũ), tôi sẽ kết luận sai rằng `attn_only` tốt hơn. Ngoài ra, việc chỉ train 30 optimizer steps trên 225 mẫu mà regression task đã lập tức tụt -0.080 (quên thảm hoạ) cũng là một bất ngờ lớn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở NB4 (3 run đối chứng mất ~53 phút) và bước lưu merged model ở NB6 (mất ~17 phút để ghi 9.32 GB checkpoint). Điều này đúng như dự đoán về thời gian tính toán của GPU T4 chia sẻ trên Colab. Tuy nhiên, chỗ phát sinh thời gian ngoài dự kiến là xử lý phân mảnh VRAM sau khi merge ở NB6 khiến Accelerate tự offload model lên meta device khi load lại base cho phần hot-swap.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin 3 điều sai lầm:
1. *Cứ training loss giảm mượt là model đã giỏi lên*: Giờ tôi hiểu loss thấp có thể chỉ là ghi nhớ (memorization), phải đo trên task accuracy thực tế.
2. *Fine-tune luôn áp đảo prompt*: Baseline (b) được prompt kỹ đã đạt 0.765 target, 1.000 format và latency chỉ 1042ms mà không tốn một giây train nào.
3. *Vị trí và rank là đòn bẩy quan trọng nhất*: Kết quả cho thấy Learning Rate (thang 1e-4 LoRA vs 1e-5 full-FT) mới là yếu tố quyết định sống còn — sai LR là model chết hẳn (`target=0.000`), còn đúng LR thì `attn_only` hay `text-linear` đều chạm ngưỡng bão hoà 0.970.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích luồng hoạt động từ NB1 đến NB5, đối chiếu rubric, giải thích cơ chế `matched_rank`, và khắc phục lỗi `ValueError: offload_dir` do tràn VRAM sau khi merge model ở NB6.
Chỗ AI hay mắc bẫy hoặc code tutorial cũ hay sai: Các đoạn code mẫu thường mặc định `bf16=True` cho kiến trúc mới, nhưng trên GPU T4 (sm_75) không hỗ trợ bf16 phần cứng, nếu không có hàm `align_trainable_precision` recast LoRA weights về fp32 thì `GradScaler` của fp16 sẽ nổ lỗi ở step 0.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Tôi sẽ **đóng băng tập đánh giá và đo baseline (b) trước tiên** (kỹ thuật Prompt Engineering tối ưu + đo trên cả tập mục tiêu và tập hồi quy chung). Nếu baseline (b) đã đáp ứng đủ SLA của khách hàng, tôi sẽ khuyến nghị dùng prompt thay vì tốn chi phí hạ tầng huấn luyện và duy trì adapter. Nếu bắt buộc fine-tune, bước kỹ thuật đầu tiên luôn là **chứng minh loss mask (`mask_proof`) bằng cách giải mã ngược**, tuyệt đối không tin tưởng mù quáng vào cờ của thư viện trước khi bấm train.
