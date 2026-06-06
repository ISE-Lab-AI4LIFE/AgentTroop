# Orchestrator V2 — KnowledgeManager + SessionMemory integration

Điều phối vòng lặp 6 phase HARMONY-X sử dụng `KnowledgeManager` cho
read/write permission-aware và `SessionMemory` cho checkpoint/state.

## Kiến trúc

```
Orchestrator
├── Phase 1-2: Cognitive Agent
│   ├── detect_anomalies(campaign_id, experiment_id)
│   └── generate_hypotheses(anomalies)
├── Phase 3-4: Strategist Agent (lặp)
│   ├── select_hypothesis_pair(hypotheses)
│   ├── design_intervention(h1, h2, campaign_id)
│   ├── execute_intervention(intervention, victim)
│   └── store_intervention(...) → Episodic Memory
├── Phase 5-6: Researcher Agent (định kỳ)
│   └── run_reverse_engineering_pipeline(campaign_id, victim, ...)
└── Session Memory (Redis) — checkpoint, state, hypothesis list
```

## Quick start

```python
from orchestration.orchestrator import Orchestrator
from knowledge.manager import KnowledgeManager
from knowledge.session_memory import SessionMemory

km = KnowledgeManager(redis_url="redis://localhost:6379/0")
sm = SessionMemory(redis_url="redis://localhost:6379/0")

orch = Orchestrator(
    cognitive_agent=cognitive_agent,
    strategist_agent=strategist_agent,
    researcher_agent=researcher_agent,
    knowledge_manager=km,
    session_memory=sm,
    victim=victim_llm,
    campaign_id="campaign_001",
    max_iterations=50,
    max_interventions=500,
    accuracy_threshold=0.9,
)

result = orch.run()
print(result["best_program_id"], result["best_accuracy"])

# Resume if interrupted
result2 = orch.resume()
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 50 | Số vòng lặp tối đa |
| `max_interventions` | 500 | Tổng số can thiệp tối đa |
| `accuracy_threshold` | 0.9 | Accuracy để dừng sớm |
| `allow_error_rate` | 0.0 | Noise tolerance |
| `synthesis_interval` | 5 | Chạy Researcher mỗi N interventions |

## Return value

```python
{
    "success": True/False,
    "campaign_id": str,
    "best_program_id": str or None,
    "best_accuracy": float,
    "total_interventions": int,
    "total_iterations": int,
    "error": str or None,
}
```

## Resume

Orchestrator tự động lưu checkpoint vào SessionMemory (Redis).
Gọi `resume()` để tiếp tục campaign bị gián đoạn.

## Testing

```bash
python -m pytest tests/orchestration/test_orchestrator_v2.py -v
```

Yêu cầu Docker (testcontainers tự động tạo Redis container).
