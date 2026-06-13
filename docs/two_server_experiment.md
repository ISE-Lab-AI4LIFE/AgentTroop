# Thí nghiệm 2 Server: Train → Inference

## Kiến trúc

```
Server A (Train)                          Server B (Inference)
─────────────────────                     ─────────────────────
1. run_experiment.py                      4. run_rmcbench_asr.py
   → orchestrator.run()                      → HarmonyASREvaluator.evaluate()
   → Học VS, surrogate, causal,              → load_campaign_state(knowledge_dir)
     SDE, technique stats                    → select_technique() dùng stats đã học
                                             → VS disagreement prioritisation
2. save_campaign_state()                     → Surrogate pre-screen
   → outputs/<campaign_id>/                  → Retry loop + failure context
     ├── version_space.json                  → Victim (OpenRouter)
     ├── surrogate_state.json                → Judge
     ├── surrogate_model.pkl               → ASR result
     ├── surrogate_scaler.pkl
     ├── sde_state.json            5. outputs/rmcbench_asr_result.json
     ├── causal_accumulator.json              ├── ASR (overall)
     ├── technique_stats.json                 ├── Per-category breakdown
                                             ├── Per-prompt details
3. Copy campaign dir ──────────────────→      └── CSV log
   (scp / rsync / blob storage)
```

## Yêu cầu

| Hạng mục | Server A | Server B |
|---|---|---|
| Python codebase | ✅ Toàn bộ | ✅ **Cùng version** (git commit) |
| PyTorch / sklearn | ✅ | ✅ **Cùng version** (pickle compatibility) |
| Neo4j | ✅ Có | ❌ **Không cần** |
| Redis | ✅ Có | ❌ **Không cần** |
| Ollama (victim local) | ❌ Không cần | ❌ Không cần (dùng OpenRouter) |
| OpenRouter API key | ❌ Không cần | ✅ Cần |
| Campaign state dir | ✅ Tạo ra | ⚠️ Copy từ A |
| Dataset RMCBench.jsonl | ❌ Không cần | ✅ Cần |

## Quy trình từng bước

### Bước 1: Train trên Server A

```bash
cd /path/to/HARMONY_X
python llama3_1_8b/run_experiment.py \
  --config configs/experiment_config.json \
  --num-seeds 30
```

Sau khi chạy xong, campaign state được tự động lưu tại:

```
outputs/<campaign_id>/
├── version_space.json
├── surrogate_state.json
├── surrogate_model.pkl
├── surrogate_scaler.pkl
├── sde_state.json
├── causal_accumulator.json
└── technique_stats.json
```

### Bước 2: Copy campaign state sang Server B

```bash
# Trên Server A
scp -r outputs/<campaign_id>/ user@server-b:/path/to/HARMONY_X/outputs/

# Hoặc dùng rsync
rsync -avz outputs/<campaign_id>/ user@server-b:/path/to/HARMONY_X/outputs/
```

> **Lưu ý**: Cả 2 server phải dùng **cùng git commit** của codebase và **cùng version** các thư viện (torch, sklearn, numpy) để pickle tương thích.

### Bước 3: Đo ASR trên Server B

```bash
cd /path/to/HARMONY_X
export OPENROUTER_API_KEY="sk-..."

python run_rmcbench_asr.py \
  --jsonl "data/RMCBench.jsonl" \
  --knowledge-dir "outputs/<campaign_id>" \
  --max-retries 2 \
  --num-prompts 0 \
  --out "outputs/rmcbench_asr_result.json"
```

Kết quả được ghi vào:

```
outputs/
├── rmcbench_asr_result.json    (JSON chi tiết)
└── rmcbench_asr_result.csv     (CSV người đọc được)
```

### Bước 4: Đọc kết quả

File JSON chứa:

```json
{
  "asr": 0.87,
  "total": 473,
  "successes": 412,
  "failures": 61,
  "skipped": 0,
  "rq": "HarmonyASR",
  "duration_s": 1842.5,
  "victim_model": "meta-llama/llama-3.1-8b-instruct",
  "max_retries": 2,
  "details": [
    {
      "index": 0,
      "goal": "Write a phishing email...",
      "crafted_prompt": "...",
      "victim_response": "...",
      "outcome": 0,
      "label": "SUCCESS",
      "technique": "roleplay_dan",
      "attempts": 2,
      "failure_reasons": ["DIRECT_REFUSAL"],
      "category": "text-to-code",
      "level": 2
    }
  ]
}
```

## Luồng khôi phục trạng thái chi tiết

Khi `run_rmcbench_asr.py` chạy với `--knowledge-dir`:

1. **`_load_knowledge()`** đọc tất cả file từ campaign state dir
2. **Technique stats** → inject vào global `_TECHNIQUE_STATS` → `select_technique()` thấy đúng success rates
3. **Version space candidates** → `_prioritise_by_disagreement()` sắp xếp prompt theo entropy
4. **Surrogate model** → `_surrogate_predict()` pre-screen prompt trước khi gửi
5. **Defense program AST** → `_inject_adversarial_keywords()` thêm trigger words để bypass
6. Trên mỗi REFUSE, **failure context** được phân tích và feed back vào Red Team Agent cho lần retry tiếp theo

## Troubleshooting

| Vấn đề | Nguyên nhân | Fix |
|---|---|---|
| `AttributeError: 'NoneType' object has no attribute 'state_dict'` | `--knowledge-dir` trỏ sai hoặc campaign không có surrogate | Kiểm tra đường dẫn campaign state |
| `pickle.UnpicklingError` | Version sklearn/torch khác nhau giữa 2 server | Đồng bộ version hoặc train lại trên Server B |
| ASR=0.0, tất cả đều REFUSE | Không có `OPENROUTER_API_KEY` hoặc key hết hạn | Kiểm tra env var |
| ASR thấp hơn kỳ vọng | `--max-retries=0` hoặc không có `--knowledge-dir` | Thêm `--max-retries 2 --knowledge-dir ...` |
