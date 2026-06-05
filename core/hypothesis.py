from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from .program import Program
from .types import Confidence, HypothesisID, ObservationID, PolicyID, Timestamp


@dataclass
class Hypothesis:
    id: HypothesisID
    statement: str
    program: Optional[Program] = None
    belief: float = 0.5
    confidence: Confidence = 0.0
    supporting_observations: List[ObservationID] = field(default_factory=list)
    opposing_observations: List[ObservationID] = field(default_factory=list)
    related_policies: List[PolicyID] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Timestamp = field(default_factory=time)
    updated_at: Timestamp = field(default_factory=time)
    status: str = "PENDING"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"hyp_{int(self.created_at * 1000)}"
        if self.updated_at < self.created_at:
            self.updated_at = self.created_at

    def add_provenance(self, source: str, details: Dict[str, Any]) -> None:
        self.provenance.append({"source": source, "details": details, "timestamp": time()})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "program": self.program.to_dict() if self.program is not None else None,
            "belief": self.belief,
            "confidence": self.confidence,
            "supporting_observations": self.supporting_observations,
            "opposing_observations": self.opposing_observations,
            "related_policies": self.related_policies,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        program_data = data.get("program")
        return Hypothesis(
            id=data.get("id", f"hyp_{int(time() * 1000)}"),
            statement=data["statement"],
            program=Program.from_dict(program_data) if program_data is not None else None,
            belief=float(data.get("belief", 0.5)),
            confidence=float(data.get("confidence", 0.0)),
            supporting_observations=list(data.get("supporting_observations", [])),
            opposing_observations=list(data.get("opposing_observations", [])),
            related_policies=list(data.get("related_policies", [])),
            provenance=list(data.get("provenance", [])),
            metadata=data.get("metadata", {}),
            created_at=float(data.get("created_at", time())),
            updated_at=float(data.get("updated_at", data.get("created_at", time()))),
            status=data.get("status", "PENDING"),
        )
