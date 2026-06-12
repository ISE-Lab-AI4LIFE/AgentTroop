"""Formal Hypothesis Optimizer — solves:

    Π* = argmin_{Π ∈ 𝒫} [ L(Π) + λ₁·C(Π) + λ₂·MDL(Π) ]

where:
    L(Π)     = prediction error (0-1 loss) on training examples
    C(Π)     = structural complexity (node count)
    MDL(Π)   = description length (repr length + parameter count)
    λ₁, λ₂   = regularization coefficients

This implements the formal optimization problem from Section 2.2 of
harmony_v5v.md, used during the Researcher Agent's synthesis phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program

from harmony.synthesis import get_synthesizer, SynthesisStats
from .grammar_exporter import GrammarExporter

logger = logging.getLogger(__name__)


@dataclass
class OptimizedProgram:
    """Result of hypothesis optimization."""

    program: Program
    accuracy: float
    complexity: int
    mdl_score: float
    total_loss: float
    loss_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program.id,
            "accuracy": round(self.accuracy, 4),
            "complexity": self.complexity,
            "mdl_score": round(self.mdl_score, 4),
            "total_loss": round(self.total_loss, 4),
            "loss_components": {k: round(v, 4) for k, v in self.loss_components.items()},
        }


class HypothesisOptimizer:
    """Solves the formal hypothesis optimization problem.

    Uses a multi-objective approach:
        1. Generate candidate programs via fitness-guided enumeration
        2. Score each by total_loss = L(Π) + λ₁·C(Π) + λ₂·MDL(Π)
        3. Select Π* with minimum total_loss
        4. Return OptimizedProgram with loss decomposition
    """

    def __init__(
        self,
        lambda_complexity: float = 0.1,
        lambda_mdl: float = 0.05,
        synthesizer: Optional[Any] = None,
        executor: Optional[ProgramExecutor] = None,
    ) -> None:
        self.lambda_complexity = lambda_complexity
        self.lambda_mdl = lambda_mdl
        self.synthesizer = synthesizer or get_synthesizer(
            mode="fitness_guided",
            config={"max_depth": 4, "beam_width": 200},
        )
        self.executor = executor or ProgramExecutor(default_registry)

        logger.info(
            "HypothesisOptimizer: λ_c=%.3f λ_m=%.3f",
            lambda_complexity, lambda_mdl,
        )

    def optimize(
        self,
        examples: List[Tuple[str, int]],
        max_candidates: int = 50,
    ) -> Optional[OptimizedProgram]:
        """Find Π* = argmin [L(Π) + λ₁·C(Π) + λ₂·MDL(Π)].

        Parameters
        ----------
        examples : list of (prompt, outcome)
        max_candidates : int
            Maximum number of candidates to evaluate.

        Returns
        -------
        OptimizedProgram or None
        """
        if not examples:
            logger.warning("No examples for optimization")
            return None

        # Step 1: Generate candidates
        candidates = self._generate_candidates(examples, max_candidates)
        if not candidates:
            logger.warning("No candidate programs generated")
            return None

        # Step 2: Score each candidate
        scored: List[Tuple[float, Program]] = []
        for prog in candidates:
            loss = self._compute_loss(prog, examples)
            scored.append((loss, prog))

        # Step 3: Select Π*
        scored.sort(key=lambda x: x[0])
        best_loss, best_prog = scored[0]

        # Decompose loss
        accuracy = self._compute_accuracy(best_prog, examples)
        complexity = best_prog.complexity()
        mdl = best_prog.mdl_score(alpha=self.lambda_mdl)

        result = OptimizedProgram(
            program=best_prog,
            accuracy=accuracy,
            complexity=complexity,
            mdl_score=mdl,
            total_loss=best_loss,
            loss_components={
                "error_loss": 1.0 - accuracy,
                "complexity_penalty": self.lambda_complexity * complexity,
                "mdl_penalty": self.lambda_mdl * mdl,
            },
        )

        logger.info(
            "Optimization: Π*=%s loss=%.4f acc=%.3f comp=%d mdl=%.3f "
            "(error=%.4f, λ_c=%.3f, λ_m=%.3f)",
            best_prog.id, best_loss, accuracy, complexity, mdl,
            1.0 - accuracy, self.lambda_complexity, self.lambda_mdl,
        )

        return result

    def _generate_candidates(
        self,
        examples: List[Tuple[str, int]],
        max_candidates: int,
    ) -> List[Program]:
        """Generate candidate programs using fitness-guided enumeration."""
        candidates: List[Program] = []

        results = self.synthesizer.synthesize(examples, k=max_candidates)
        for p in results:
            if p not in candidates:
                candidates.append(p)

        exporter = GrammarExporter(
            primitive_registry=default_registry,
            max_depth=3,
        )
        enum_progs = exporter.enumerate_programs(
            max_depth=3, examples=examples,
        )
        for p in enum_progs[:max_candidates]:
            if p not in candidates:
                candidates.append(p)

        logger.debug("Generated %d candidate programs", len(candidates))
        return candidates[:max_candidates]

    def _compute_loss(self, program: Program, examples: List[Tuple[str, int]]) -> float:
        """Compute total loss = L(Π) + λ₁·C(Π) + λ₂·MDL(Π)."""
        accuracy = self._compute_accuracy(program, examples)
        error_loss = 1.0 - accuracy
        complexity_penalty = self.lambda_complexity * program.complexity()
        mdl_penalty = self.lambda_mdl * program.mdl_score(alpha=self.lambda_mdl)
        total = error_loss + complexity_penalty + mdl_penalty
        return total

    def _compute_accuracy(self, program: Program, examples: List[Tuple[str, int]]) -> float:
        if not examples:
            return 0.0
        correct = 0
        for prompt, expected in examples:
            try:
                if self.executor.execute(program, prompt) == expected:
                    correct += 1
            except Exception:
                pass
        return correct / len(examples)

    def score(self, program: Program, examples: List[Tuple[str, int]]) -> float:
        """Convenience: score a single program."""
        return self._compute_loss(program, examples)
