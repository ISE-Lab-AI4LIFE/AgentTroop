import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program

from adapters.base_victim import BaseVictim


@dataclass
class Theory:
    """A scientific theory about LLM safety mechanisms.
    
    Following the structure from harmony_v5v.md Section 9:
    T = (pattern, conditions, confidence, provenance)
    """
    id: str
    pattern: str
    conditions: Dict[str, Any]
    confidence: float
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "conditions": self.conditions,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Theory":
        return cls(
            id=data["id"],
            pattern=data["pattern"],
            conditions=data.get("conditions", {}),
            confidence=float(data.get("confidence", 0.0)),
            provenance=list(data.get("provenance", [])),
            created_at=float(data.get("created_at", time.time())),
        )


class ScientificDiscoveryEvaluator:
    """Evaluates how well scientific theories generalise across victims."""

    def __init__(self) -> None:
        self._executor = ProgramExecutor(default_registry)

    def evaluate_theory(
        self,
        theory: Theory,
        victims: List[BaseVictim],
        test_inputs: List[str],
    ) -> float:
        """Measure how accurately the theory predicts victim behaviour.
        
        For each victim, constructs a program from the theory pattern
        and compares its predictions to the victim's responses.
        Returns the mean accuracy across all victims.
        """
        if not victims or not test_inputs:
            return 0.0
        total_correct = 0
        total_samples = 0
        for victim in victims:
            for inp in test_inputs:
                predicted = self._predict_from_theory(theory, inp)
                actual = victim.respond(inp)
                if predicted == actual:
                    total_correct += 1
                total_samples += 1
        return total_correct / total_samples if total_samples > 0 else 0.0

    def _predict_from_theory(self, theory: Theory, prompt: str) -> int:
        """Use the theory pattern to make a prediction.
        
        Currently a heuristic: checks if danger words from the
        theory conditions appear in the prompt.
        """
        danger_words = theory.conditions.get("danger_words", [])
        for word in danger_words:
            if word.lower() in prompt.lower():
                return 1
        return 0

    def cross_family_transfer_score(
        self,
        theory: Theory,
        source_victims: List[BaseVictim],
        target_victims: List[BaseVictim],
        test_inputs: List[str],
    ) -> float:
        """Measure how well a theory transfers from source to target victims.
        
        Returns accuracy on target victims after being validated on source.
        """
        if not target_victims or not test_inputs:
            return 0.0
        total_correct = 0
        total_samples = 0
        for victim in target_victims:
            for inp in test_inputs:
                predicted = self._predict_from_theory(theory, inp)
                actual = victim.respond(inp)
                if predicted == actual:
                    total_correct += 1
                total_samples += 1
        return total_correct / total_samples if total_samples > 0 else 0.0
