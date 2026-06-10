# HARMONY-X Experiment: Reverse-engineer Llama-3.1-8B Safety Layer

## Mục tiêu

Chạy HARMONY-X pipeline để tự động reverse engineer cơ chế an toàn (defense layer) của **Llama-3.1-8B** thông qua Ollama.

Pipeline sẽ:
1. Gieo các baseline episodes (prompt harmful + benign, một số được encode bằng rot13/base64)
2. Cognitive Agent phát hiện anomalies (prompt giống nhau nhưng outcome khác nhau do encoding)
3. Cognitive Agent sinh hypotheses về luật an toàn của model
4. Strategist Agent thiết kế can thiệp để phân biệt giữa các hypotheses
5. Researcher Agent tổng hợp chương trình defense cuối cùng
6. Xuất ra AST, theory, và lịch sử can thiệp

## Yêu cầu

- **RAM**: ~8GB+ (cho Llama-3.1-8B)
- **Ollama**: phiên bản mới nhất
- **Python**: 3.10+
- **Dịch vụ**: Redis (local) + Neo4j (local)
- **prompt.csv** (RMCBench format) ở thư mục gốc dự án — chứa 473 harmful prompts

## Cài đặt

### 1. Cài đặt Ollama và Llama-3.1-8B

```bash
# Cài đặt Ollama (nếu chưa có)
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.1:8b

# Kiểm tra
ollama run llama3.1:8b "Hello" --nowordwrap
```

Hoặc chạy script tự động:

```bash
bash llama3_1_8b/setup.sh
```

### 2. Cài đặt Python dependencies

```bash
pip install -r llama3_1_8b/requirements.txt
```

### 3. Đảm bảo Redis và Neo4j đang chạy

```bash
# Redis
brew services start redis

# Neo4j
brew services start neo4j
```

### 4. Cấu hình API key cho Gemma (Cognitive Agent dùng LLM riêng)

Tạo file `.env` ở thư mục gốc:

```
GEMMA_API_KEY=your_api_key_here
```

Cognitive Agent dùng **Gemma-4-31b-it** (qua Google GenAI API) để sinh hypotheses.

## Cấu hình

Xem và sửa `llama3_1_8b/config.yaml`:

```yaml
use_async: false            # true: dùng async pipeline (Orchestrator.async_run)
orchestrator:
  max_iterations: 30        # số vòng lặp POMDP tối đa
  max_interventions: 300    # số can thiệp tối đa
  accuracy_threshold: 0.85  # ngưỡng hội tụ accuracy
  synthesis_interval: 3     # chạy Researcher mỗi N can thiệp
```

## Chạy thí nghiệm

### Sử dụng script (khuyến nghị)

```bash
cd /path/to/HARMONY_X
bash llama3_1_8b/run_exp.sh --num-seeds 100
```

Các tham số:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--num-seeds N` | 50 | Tổng số seed prompts (60% harmful / 40% benign) |
| `--full` | — | Dùng toàn bộ prompts từ CSV (ghi đè `--num-seeds`) |
| `--prior-campaign ID` | — | Campaign ID trước đó cho RQ3 transfer evaluation |

### Chạy thủ công

```bash
cd /path/to/HARMONY_X
GEMMA_API_KEY=$(head -1 .env | cut -d= -f2) python llama3_1_8b/run_experiment.py [--num-seeds N] [--full] [--prior-campaign ID]
```

### Kết quả

Sau khi chạy, các file đầu ra nằm trong `llama3_1_8b/outputs/`:

| File | Nội dung |
|------|----------|
| `final_program.json` | AST của chương trình defense cuối cùng |
| `final_theory.json` | Lý thuyết trừu tượng (dạng IF ... THEN ...) |
| `interventions_history.json` | Lịch sử các can thiệp (prompt, outcome) |
| `hypotheses_history.json` | Thông tin tổng quan kết quả thí nghiệm |

Log chi tiết nằm trong `llama3_1_8b/logs/experiment_YYYYMMDD_HHMMSS.log`.

## Phân tích kết quả

Mở Jupyter notebook:

```bash
jupyter notebook llama3_1_8b/analysis.ipynb
```

## Cấu trúc thư mục

```
llama3_1_8b/
├── README.md                  # Hướng dẫn
├── requirements.txt           # Python dependencies
├── setup.sh                   # Script cài đặt
├── ollama_victim.py           # Adapter Llama-3.1-8B (BaseVictim)
├── run_experiment.py          # Script chính
├── config.yaml                # Cấu hình tham số
├── logs/                      # Log file (tự động tạo)
│   └── experiment_*.log
├── outputs/                   # Kết quả (tự động tạo)
│   ├── final_program.json
│   ├── final_theory.json
│   ├── hypotheses_history.json
│   └── interventions_history.json
└── analysis.ipynb             # Notebook phân tích
```

## Giải thích luồng

```
Seed episodes (harmful + benign, một số encoded)
    ↓
Cognitive Agent phát hiện anomalies
(prompt giống nhau, outcome khác nhau)
    ↓
Cognitive Agent sinh hypotheses
(ví dụ: "model từ chối nếu prompt chứa 'bomb'")
    ↓
Strategist Agent thiết kế can thiệp
(transform prompt để phân biệt hypotheses)
    ↓
Victim (Llama-3.1-8B) trả lời can thiệp
    ↓
Researcher Agent tổng hợp chương trình
(contains_word, regex, v.v.)
    ↓
Lưu program vào DefenseProgramStore và theory vào ScientificMemory
```

## Troubleshooting

**Victim Ollama không kết nối được:**
- Kiểm tra `ollama serve` đang chạy
- Kiểm tra URL trong config.yaml (`http://localhost:11434`)

**Redis/Neo4j không kết nối được:**
- Chạy `brew services start redis` và `brew services start neo4j`
- Kiểm tra biến môi trường `HX_REDIS_URL`, `HX_NEO4J_URI`

**Cognitive Agent lỗi API key:**
- Đảm bảo `GEMMA_API_KEY` được set trong `.env`
- Hoặc set `GENAI_API_KEY`

**Llama-3.1-8B response quá chậm:**
- Giảm `max_tokens` trong config.yaml (mặc định 100)
- Kiểm tra RAM còn trống
