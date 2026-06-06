# Session Memory (L2) — Redis-backed campaign state cache

Lưu trạng thái chiến dịch reverse engineering trong Redis với TTL tự động.
Mỗi campaign là một hash `session:{campaign_id}`.

## Yêu cầu

- `pip install redis`
- Redis server đang chạy

## Quick start

```python
from knowledge.session_memory import SessionMemory

sm = SessionMemory("redis://localhost:6379/0", ttl=86400)

# Create
sm.create_session("campaign_1", "GPT-4o", {"env": "test"})

# Update
sm.increment_iteration("campaign_1")
sm.increment_intervention_count("campaign_1")
sm.set_best_program("campaign_1", "prog_abc", 0.95)
sm.add_hypothesis("campaign_1", "hyp_001")
sm.set_status("campaign_1", "completed")

# Read
session = sm.get_session("campaign_1")
print(session["target_model"], session["iteration"], session["status"])

# List active
sessions = sm.list_active_sessions()

# Delete
sm.delete_session("campaign_1")
```

## API

| Method | Mô tả |
|--------|-------|
| `create_session(campaign_id, target_model, metadata)` | Tạo session mới |
| `get_session(campaign_id)` → dict/None | Lấy toàn bộ session |
| `update_session(campaign_id, updates)` | Cập nhật field |
| `delete_session(campaign_id)` | Xoá session + hypothesis list |
| `increment_iteration(campaign_id)` → int | Atomic ++ |
| `increment_intervention_count(campaign_id, delta)` → int | Atomic += delta |
| `set_best_program(campaign_id, program_id, accuracy)` | Ghi nhận best program |
| `add_hypothesis(campaign_id, hyp_id)` / `remove_hypothesis` | Quản lý hypothesis list |
| `list_hypotheses(campaign_id)` → List[str] | Danh sách hypothesis IDs |
| `set_status(campaign_id, status)` | running / completed / failed |
| `list_active_sessions()` → List[str] | Quét Redis keys |
| `session_exists(campaign_id)` → bool | Kiểm tra tồn tại |

## Session fields

| Field | Type | Description |
|-------|------|-------------|
| `campaign_id` | string | ID chiến dịch |
| `target_model` | string | Model mục tiêu |
| `current_best_program_id` | string | Program tốt nhất |
| `current_best_accuracy` | float | 0..1 |
| `hypothesis_ids` | List[str] | Đọc từ Redis list riêng |
| `iteration` | int | Số vòng lặp |
| `intervention_count` | int | Số can thiệp |
| `status` | string | running / completed / failed |
| `started_at` | float | Timestamp |
| `updated_at` | float | Timestamp |
| `metadata` | dict | Mở rộng |

## Testing

```bash
python -m pytest tests/knowledge/test_session_memory.py -v
```

Yêu cầu Docker (testcontainers tự động tạo Redis container).
