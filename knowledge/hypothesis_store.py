import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HypothesisRecord:
    id: str = ""
    description: str = ""
    condition: str = ""
    confidence: float = 0.0
    supporting_anomaly_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    campaign_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"hyp_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "condition": self.condition,
            "confidence": self.confidence,
            "supporting_anomaly_ids": self.supporting_anomaly_ids,
            "created_at": self.created_at,
            "campaign_id": self.campaign_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypothesisRecord":
        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            condition=data.get("condition", ""),
            confidence=float(data.get("confidence", 0.0)),
            supporting_anomaly_ids=list(data.get("supporting_anomaly_ids", [])),
            created_at=float(data.get("created_at", time.time())),
            campaign_id=data.get("campaign_id", ""),
            metadata=dict(data.get("metadata", {})),
        )


class HypothesisStore:
    def __init__(self, db_path: str = "hypothesis_store.db"):
        self._db_path = db_path
        self._records: Dict[str, HypothesisRecord] = {}

    def save(self, hypothesis: HypothesisRecord) -> str:
        self._records[hypothesis.id] = hypothesis
        logger.debug("Saved hypothesis %s", hypothesis.id)
        return hypothesis.id

    def get(self, hypothesis_id: str) -> Optional[HypothesisRecord]:
        return self._records.get(hypothesis_id)

    def find(
        self,
        campaign_id: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> List[HypothesisRecord]:
        results: List[HypothesisRecord] = []
        for rec in self._records.values():
            if campaign_id and rec.campaign_id != campaign_id:
                continue
            if rec.confidence < min_confidence:
                continue
            results.append(rec)
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    def delete(self, hypothesis_id: str) -> bool:
        if hypothesis_id in self._records:
            del self._records[hypothesis_id]
            return True
        return False

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
