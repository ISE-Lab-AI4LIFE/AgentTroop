# Researcher Agent — Reverse Engineering Safety Programs

## Tổng quan

**Researcher Agent** là trung tâm reverse engineering của HARMONY-X. Nó thực thi pipeline 5 bước:

```
┌─────────────────────────────────────────────────────────────────┐
│  Researcher Agent                                                │
│                                                                  │
│  1. Đọc episodes từ Episodic Memory                              │
│  2. Tổng hợp chương trình bằng CVC5Synthesizer                   │
│  3. Kiểm chứng chương trình bằng ProgramVerifier                 │
│  4. Lưu chương trình vào Defense Program Store                    │
│  5. Trích xuất lý thuyết → Scientific Memory                     │
└─────────────────────────────────────────────────────────────────┘
```

Agent này **không sử dụng LLM** — chỉ dùng các module `synthesis` và `knowledge` đã có.

---

## Khởi tạo

```python
from agents.researcher import ResearcherAgent
from knowledge.episodic.episodic import EpisodicMemory
from knowledge.defense_store import DefenseProgramStore
from knowledge.scientific_memory import ScientificMemory

agent = ResearcherAgent(
    episodic_memory=EpisodicMemory(db_path="episodic.db"),
    defense_store=DefenseProgramStore(),
    scientific_memory=ScientificMemory(),
    ontology_memory=None,              # optional
    synthesizer=None,                   # auto-create CVC5Synthesizer
    executor=None,                      # auto-create ProgramExecutor
    default_model_family="GPT-4o",
)
```

### Constructor parameters

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `episodic_memory` | `EpisodicMemory` | **required** | Đọc intervention data |
| `defense_store` | `DefenseProgramStore` | **required** | Lưu chương trình đã verify |
| `scientific_memory` | `ScientificMemory` | **required** | Lưu abstract theory |
| `ontology_memory` | `OntologyMemory` | `None` | Optional, hỗ trợ synthesis |
| `synthesizer` | `CVC5Synthesizer` | `None` → auto | Dùng mặc định nếu không cấp |
| `executor` | `ProgramExecutor` | `None` → auto | Dùng `default_registry` |
| `default_model_family` | `str` | `"unknown"` | Gán nhãn cho theory |

### Default synthesizer params

Khi `synthesizer=None`, tự tạo:
```python
CVC5Synthesizer(max_depth=3, beam_width=200, timeout=30, use_cache=True)
```

---

## Phương thức

### `synthesize_from_campaign(campaign_id, experiment_id=None, allow_error_rate=0.0)`

Đọc episodes từ campaign, tổng hợp chương trình.

```python
program, stats = agent.synthesize_from_campaign(
    campaign_id="camp_001",
    experiment_id="exp_42",       # optional
    allow_error_rate=0.05,        # cho phép 5% sai số
)

if program is not None:
    print(f"Found program, depth={stats.depth_used}, tried={stats.programs_tried}")
else:
    print("No program found")
```

### `verify_program(program, victim, num_test_interventions=10, accuracy_threshold=0.9, exclude_prompts=None, verbose=False)`

Kiểm chứng chương trình trên victim.

```python
report = agent.verify_program(
    program,
    victim=my_victim,
    num_test_interventions=20,
    accuracy_threshold=0.95,
    verbose=True,
)

if report.verified:
    print(f"Verified! Accuracy: {report.accuracy:.2f}")
```

### `store_program(program, name, confidence, provenance, status="confirmed")`

Lưu chương trình vào Defense Program Store.

```python
pid = agent.store_program(
    program,
    name="keyword_filter_bomb",
    confidence=0.95,
    provenance=["exp_42", "ep_100"],
    status="confirmed",
)
```

### `abstract_theory(program, model_family=None, conditions=None, provenance=None)`

Trích xuất Theory từ chương trình.

```python
theory = agent.abstract_theory(
    program,
    model_family="GPT-4o",
    conditions={"accuracy": 0.95},
    provenance=["ep_1", "ep_2"],
)
```

### `store_theory(theory)`

Lưu Theory vào Scientific Memory.

```python
tid = agent.store_theory(theory)
```

### `run_reverse_engineering_pipeline(campaign_id, victim, program_name=None, allow_error_rate=0.0, num_test_interventions=10, accuracy_threshold=0.9, model_family=None, experiment_id=None, verbose=False)`

Pipeline end-to-end, trả về dict kết quả.

```python
result = agent.run_reverse_engineering_pipeline(
    campaign_id="camp_001",
    victim=my_victim,
    experiment_id="exp_42",
    allow_error_rate=0.05,
    num_test_interventions=20,
    accuracy_threshold=0.9,
)

print(result)
# {
#     "success": True,
#     "program_id": "prog_abc123",
#     "theory_id": "thr_def456",
#     "accuracy": 0.95,
#     "verified": True,
#     "stats": SynthesisStats(...),
#     "report": VerificationReport(...),
#     "error": None,
# }
```

---

## Pipeline flow chi tiết

```
run_reverse_engineering_pipeline()
│
├── 1. synthesize_from_campaign()
│      ├── filter_episodes(campaign_id, experiment_id)
│      ├── extract (prompt, outcome) pairs
│      └── synthesizer.synthesize_with_stats()
│
├── 2. verify_program()
│      └── ProgramVerifier.verify(victim, ...)
│
├── 3. store_program()
│      └── DefenseProgramStore.save(record)
│
├── 4. abstract_theory()
│      └── synthesizer.abstract_theory(program, ...)
│
└── 5. store_theory()
       └── ScientificMemory.save_theory(theory)
```

---

## Xử lý lỗi

- Nếu `synthesize_from_campaign` không tìm thấy episodes hoặc examples → trả về `(None, SynthesisStats())`
- Nếu synthesis không tìm được program → pipeline dừng, `success=False`
- Mọi exception trong pipeline được catch, log, và trả về dict `{success: False, error: "..."}`

---

## Dependencies

| Module | File | Vai trò |
|--------|------|---------|
| `EpisodicMemory` | `knowledge/episodic/episodic.py` | Đọc intervention data |
| `DefenseProgramStore` | `knowledge/defense_store.py` | Lưu program |
| `ScientificMemory` | `knowledge/scientific_memory.py` | Lưu theory |
| `OntologyMemory` | `knowledge/ontology_memory.py` | Optional primitive catalog |
| `CVC5Synthesizer` | `synthesis/cvc5_synthesizer.py` | Program synthesis |
| `ProgramVerifier` | `synthesis/verifier.py` | Verification |
| `ProgramExecutor` | `core/executor.py` | Program execution |

---

## Testing

17 tests, tất cả dùng `unittest.mock`:

```
tests/agents/test_researcher.py
├── TestSynthesizeFromCampaign  (4 tests)
├── TestVerifyProgram           (2 tests)
├── TestStoreProgram            (2 tests)
├── TestAbstractTheory          (2 tests)
├── TestStoreTheory             (1 test)
├── TestPipeline                (5 tests)
└── TestDefaultSynthesizer      (1 test)
```

Chạy: `python -m pytest tests/agents/test_researcher.py -v`
