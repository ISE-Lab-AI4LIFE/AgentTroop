## Ý nghĩa của accuracy

Accuracy là tỷ lệ test prompts mà chương trình `contains_word("generate")` dự đoán **đúng** response của victim. Cụ thể:
- Chương trình kiểm tra prompt có chứa từ "generate" không → nếu có, predict REFUSE (1), nếu không predict ACCEPT (0)
- So sánh prediction với response thật của Llama-3.1-8B
- `0.92` = 23/25 test prompts đúng

Con số này **không phản ánh chất lượng thật** vì:
1. Verifier chỉ dùng **25 prompts** → mỗi lần sai mất 4%, 23/25 luôn ra 0.92
2. Accuracy thực tế trên tập holdout 1581 prompts chỉ ~**0.72-0.75** (xem log lines 252, 375, 504)
3. Best program luôn là `contains_word("generate")` — version space sụp đổ ngay cycle 1

---

## Kết quả các RQs

| RQ | Mô tả | Kết quả | Đánh giá |
|---|---|---|---|
| **RQ0** | Chương trình dự đoán chính xác response trên prompt chưa thấy? (kỳ vọng >85%) | **0.68** (34/50) | **FAILED** — 0.68 < 0.85 |
| **RQ1** | Can thiệp có mục tiêu giảm số truy vấn? (kỳ vọng giảm ≥50%) | 1 intervention để đạt threshold, best_accuracy=1.0 | **PASSED** nhưng `interventions_to_threshold=1` rất đáng nghi — có bug |
| **RQ2** | Phát hiện thành phần phòng thủ mới? | Exported 50 cặp (program, explanation) chờ annotator đánh giá | Pending |
| **RQ3** | Scientific Memory hỗ trợ transfer learning? | **Skipped** — không có prior_campaign_id | Không đánh giá được |
| **ASR (Baseline)** | ASR gốc (không tấn công) | **0.433** (13/30) | Victim từ chối 57% harmful prompts |
| **Harmony ASR** | ASR sau khi HARMONY-X tấn công | **0.4** (12/30) | **THẤP HƠN baseline** (0.4 < 0.433) → tấn công phản tác dụng |

---

## Phân tích chi tiết

### RQ0 — thất bại
Chương trình `contains_word("generate")` chỉ đạt 0.68 (34/50) trên tập kiểm tra. Nguyên nhân: chương trình quá đơn giản, chỉ dùng 1 keyword, không generalize được (nhiều harmful prompts không chứa "generate", nhiều benign prompts có thể chứa "generate").

### RQ1 — interventions_to_threshold=1 rất bất thường
Con số này có vẻ là bug: campaign chạy 22 iterations nhưng `interventions_to_threshold` báo 1, `best_accuracy=1.0` cũng sai. Cần kiểm tra lại logic tính toán trong `evaluation/metrics/intervention_efficiency.py`.

### Harmony ASR < Baseline ASR
ASR sau tấn công (0.4) thấp hơn baseline (0.433) → chiến lược tấn công đang **làm cho victim an toàn hơn**. Các kỹ thuật reframe (educational_framing, code_jailbreak) vô tình kích hoạt thêm refusal patterns thay vì bypass chúng.

### Hạn chế chính của experiment
1. **Version space thiếu exploration** — entropy giảm từ 3.16 → 0.086 ngay cycle 1
2. **Chỉ tìm ra 1 keyword** ("generate") — không khám phá được các pattern phòng thủ phức tạp hơn
3. **Threshold quá thấp** (0.7) cho phép chương trình kém được "verified"
4. **Verifier test set quá nhỏ** (25 prompts) không đại diện