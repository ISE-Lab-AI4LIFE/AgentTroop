from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional

from .types import ObservationID, InterventionID, Outcome, Timestamp


@dataclass
class Observation:
    intervention_id: InterventionID
    outcome: Outcome
    victim_id: Optional[str] = None
    campaign_id: Optional[str] = None
    experiment_id: Optional[str] = None
    raw_response: Optional[str] = None
    latency: Optional[float] = None
    token_usage: Optional[int] = None
    environment_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Timestamp = field(default_factory=time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: ObservationID = field(default_factory=lambda: "")

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"obs_{int(self.timestamp * 1000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "intervention_id": self.intervention_id,
            "victim_id": self.victim_id,
            "campaign_id": self.campaign_id,
            "experiment_id": self.experiment_id,
            "outcome": self.outcome,
            "raw_response": self.raw_response,
            "latency": self.latency,
            "token_usage": self.token_usage,
            "environment_metadata": self.environment_metadata,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return Observation(
            id=data.get("id", ""),
            intervention_id=data["intervention_id"],
            victim_id=data.get("victim_id"),
            campaign_id=data.get("campaign_id"),
            experiment_id=data.get("experiment_id"),
            outcome=int(data["outcome"]),
            raw_response=data.get("raw_response"),
            latency=data.get("latency"),
            token_usage=data.get("token_usage"),
            environment_metadata=data.get("environment_metadata", {}),
            timestamp=float(data.get("timestamp", 0.0)),
            metadata=data.get("metadata", {}),
        )
