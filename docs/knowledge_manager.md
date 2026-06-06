# Knowledge Manager — Unified read/write + proposal queue

API thống nhất cho các agent đọc/ghi tri thức, đảm bảo **single writer ownership** qua proposal queue (Redis/in-memory).

## Kiến trúc

```
Agent (non-owner) → propose() → Queue (Redis/in-memory) → poll_proposals() → Owner → resolve()
Agent (owner)     → write()   → Store (trực tiếp)
```

### Target & Owners mapping

| Target | Owners | Non-owners được phép? |
|--------|--------|-----------------------|
| `episodic` | CognitiveAgent, StrategistAgent | write trực tiếp |
| `defense_program_store` | ResearcherAgent | propose |
| `scientific_memory` | ResearcherAgent | propose |
| `causal_graph` | ResearcherAgent | propose |
| `session_memory` | Orchestrator | propose |

## Yêu cầu

- Redis (optional) — nếu không có, fallback in-memory
- `pip install redis` (optional, cho production)

## Quick start

```python
from knowledge.manager import KnowledgeManager, Target

# In-memory (dev/test)
mgr = KnowledgeManager(use_redis=False)

# Hoặc Redis (production)
mgr = KnowledgeManager(redis_url="redis://localhost:6379/0", use_redis=True)

# Đăng ký store
mgr.register_store(Target.EPISODIC.value, my_episodic_instance)

# Owner write
mgr.write(Target.EPISODIC.value, data, "CognitiveAgent")

# Non-owner propose
pid = mgr.propose(Target.CAUSAL.value, data, "CognitiveAgent")

# Owner poll
proposals = mgr.poll_proposals("ResearcherAgent", Target.CAUSAL.value)

# Owner resolve
mgr.resolve_proposal(pid, accepted=True, result="ok")
```

## API

| Method | Mô tả |
|--------|-------|
| `register_store(target, store_instance)` | Đăng ký store module |
| `read(target, query)` | Đọc trực tiếp |
| `write(target, data, agent_id)` | Ghi nếu là owner |
| `propose(target, data, agent_id, action)` | Gửi proposal |
| `poll_proposals(agent_id, target, timeout)` | Lấy proposals (owner only) |
| `resolve_proposal(proposal_id, accepted, result, error)` | Đánh dấu kết quả |
| `get_proposal_status(proposal_id)` | Xem trạng thái |

## Backends

### Redis (production)
- Queue: Redis list `proposal_queue:<target>`
- Status: Redis hash `proposal:<proposal_id>`
- Connection pool, decode_responses=True

### In-memory (dev/test)
- `queue.Queue` per target
- Dict for proposal status
- Không cần external dependencies

## Testing

```bash
python -m pytest tests/knowledge/test_manager.py -v
```

Dùng **testcontainers** (Redis container tự động). Yêu cầu Docker.
