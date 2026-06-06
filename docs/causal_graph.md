# Causal Graph — Neo4j-backed causal relationship store

Lưu trữ đồ thị nhân quả đã được kiểm chứng bằng can thiệp, dùng Neo4j làm backend. Hỗ trợ CRUD node/edge, truy vấn cause/effect, export/import.

## Yêu cầu

- Neo4j 5.x đang chạy (Docker: `docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5`)
- `pip install neo4j`

## Quick start

```python
from knowledge.causal_graph import CausalGraph

with CausalGraph() as g:
    nid1 = g.add_node("ROT13", "transform", {"desc": "ROT13 cipher"})
    nid2 = g.add_node("REFUSE", "outcome")
    g.add_edge(nid1, nid2, strength=0.85, p_value=0.01, intervention_ids=["int_001"])

    causes = g.find_causes(nid2, min_strength=0.5)
    for edge, node in causes:
        print(f"{node.name} -> {edge.strength}")
```

## API

### Nodes
- `add_node(name, node_type, metadata) -> str`
- `get_node(node_id) -> Optional[CausalNode]`
- `get_or_create_node(name, node_type, metadata) -> str`
- `find_nodes_by_name(name) -> List[CausalNode]`
- `find_nodes_by_type(node_type) -> List[CausalNode]`
- `update_node(node_id, name, metadata) -> bool`
- `delete_node(node_id) -> bool`

### Edges
- `add_edge(source_id, target_id, strength, p_value, intervention_ids) -> bool`
- `get_edge(source_id, target_id) -> Optional[CausalEdge]`
- `delete_edge(source_id, target_id) -> bool`

### Queries
- `find_causes(target_id, min_strength) -> List[Tuple[CausalEdge, CausalNode]]`
- `find_effects(source_id, min_strength) -> List[Tuple[CausalEdge, CausalNode]]`
- `get_all_nodes() -> List[CausalNode]`
- `get_all_edges() -> List[CausalEdge]`

### Admin
- `clear() -> int` — xoá toàn bộ
- `export(file_path)` — JSON
- `import_(file_path) -> int` — số node+edge đã import

## Dataclasses

```python
@dataclass
class CausalNode:
    id: str          # auto: cnd_<hex>
    name: str
    type: str        # primitive | defense_component | outcome
    metadata: Dict
    created_at: float

@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    strength: float   # 0..1, clamped
    p_value: float    # <0.05 ideal, clamped 0..1
    intervention_ids: List[str]
    created_at: float
```

## Constraints

- `CausalNode.id` — UNIQUE
- `CAUSAL_EDGE(source_id, target_id)` — composite UNIQUE
- Indexes: `type`, `name`

## Testing

```bash
python -m pytest tests/knowledge/test_causal_graph.py -v
```

Dùng **testcontainers** (Neo4j container tự động). Yêu cầu Docker.
