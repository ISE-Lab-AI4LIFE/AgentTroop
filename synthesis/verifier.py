import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple

from core.executor import ProgramExecutor
from core.primitive import (
    RemovePunctuationTransform,
    ToLowercaseTransform,
    Transform,
    default_registry,
)
from core.program import Program
from core.types import Outcome

from adapters.base_victim import BaseVictim

logger = logging.getLogger(__name__)

InterventionGenerator = Callable[["BaseVictim", int], List[str]]


@dataclass
class VerificationReport:
    program: Program
    accuracy: float = 0.0
    failures: List[Tuple[str, Outcome, Outcome]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    verified: bool = False
    num_tested: int = 0
    num_correct: int = 0

    def to_dict(self) -> dict:
        return {
            "program_id": self.program.id,
            "accuracy": self.accuracy,
            "verified": self.verified,
            "num_tested": self.num_tested,
            "num_correct": self.num_correct,
            "num_failures": len(self.failures),
            "failures": [
                {"prompt": p, "expected": e, "actual": a}
                for p, e, a in self.failures
            ],
            "suggestions": self.suggestions,
        }


class ProgramVerifier:
    def __init__(
        self,
        executor: ProgramExecutor,
        victim: BaseVictim,
        intervention_generator: Optional[InterventionGenerator] = None,
    ) -> None:
        self.executor = executor
        self.victim = victim
        self.intervention_generator = (
            intervention_generator or self._default_intervention_generator
        )

    def verify(
        self,
        program: Program,
        num_test_interventions: int = 200,
        accuracy_threshold: float = 0.8,
        exclude_prompts: Optional[Set[str]] = None,
        verbose: bool = False,
    ) -> VerificationReport:
        logger.info(
            "Verifying program %s with %d test interventions (verbose=%s)",
            program.id, num_test_interventions, verbose,
        )

        test_prompts = self.intervention_generator(
            self.victim, num_test_interventions
        )

        if verbose:
            logger.debug("Generated %d test prompts: %s", len(test_prompts),
                          test_prompts[:5])

        if exclude_prompts:
            filtered: List[str] = []
            for p in test_prompts:
                if p not in exclude_prompts:
                    filtered.append(p)
            test_prompts = filtered
            if verbose:
                logger.debug("Excluded %d prompts, %d remaining",
                              num_test_interventions - len(test_prompts), len(test_prompts))
            if len(test_prompts) < num_test_interventions:
                test_prompts = self.intervention_generator(
                    self.victim,
                    num_test_interventions + (num_test_interventions - len(test_prompts)),
                )

        test_prompts = test_prompts[:num_test_interventions]

        correct = 0
        failures: List[Tuple[str, Outcome, Outcome]] = []

        for prompt in test_prompts:
            expected = self.victim.respond(prompt)
            try:
                actual = self.executor.execute(program, prompt)
            except Exception as exc:
                logger.debug(
                    "Program %s failed on %r: %s",
                    program.id, prompt, exc,
                )
                actual = 0

            if actual == expected:
                correct += 1
            else:
                failures.append((prompt, expected, actual))
                logger.debug(
                    "Failure: prompt=%r expected=%d actual=%d",
                    prompt, expected, actual,
                )

            if verbose and actual != expected:
                logger.info(
                    "  [%d/%d] prompt=%r expected=%d actual=%d  ✗",
                    correct + len(failures), num_test_interventions,
                    prompt, expected, actual,
                )

        accuracy = (
            correct / num_test_interventions
            if num_test_interventions > 0
            else 0.0
        )
        verified = accuracy >= accuracy_threshold

        suggestions = self._generate_suggestions(failures, program)

        report = VerificationReport(
            program=program,
            accuracy=accuracy,
            failures=failures,
            suggestions=suggestions,
            verified=verified,
            num_tested=num_test_interventions,
            num_correct=correct,
        )

        if verified:
            logger.info(
                "Program %s verified (accuracy=%.2f, %d failures)",
                program.id, accuracy, len(failures),
            )
        else:
            logger.warning(
                "Program %s NOT verified (accuracy=%.2f < %.2f, %d failures)",
                program.id, accuracy, accuracy_threshold, len(failures),
            )
            if verbose:
                logger.info("Suggestions: %s", suggestions)

        return report

    def _generate_suggestions(
        self,
        failures: List[Tuple[str, Outcome, Outcome]],
        program: Program,
    ) -> List[str]:
        suggestions: List[str] = []

        if not failures:
            return suggestions

        false_positives = 0
        false_negatives = 0

        for _, expected, actual in failures:
            if expected == 0 and actual == 1:
                false_positives += 1
            elif expected == 1 and actual == 0:
                false_negatives += 1

        if false_positives > false_negatives:
            suggestions.append(
                f"Program over-predicts REFUSE "
                f"({false_positives} false positives). "
                "Consider relaxing the condition."
            )
        elif false_negatives > false_positives:
            suggestions.append(
                f"Program under-predicts REFUSE "
                f"({false_negatives} false negatives). "
                "Consider strengthening the condition "
                "or adding more predicates."
            )

        if len(failures) >= 3:
            suggestions.append(
                "Consider increasing program depth "
                "to capture more complex patterns."
            )

        return suggestions

    def _default_intervention_generator(
        self, victim: BaseVictim, n: int
    ) -> List[str]:
        # Ensure minimum for balanced testing
        half = max(1, n // 2)
        result: List[str] = []

        # 1. Harmful prompts from CSV (first priority)
        harmful = _load_harmful_prompts(n=half)
        for p in harmful:
            if len(result) >= half:
                break
            if p not in result:
                result.append(p)

        # 2. Benign educational prompts
        benign = _benign_prompts()
        for p in benign:
            if len(result) >= n:
                break
            if p not in result:
                result.append(p)

        # 3. Fill remaining with role-prefixed benign prompts
        transforms: List[Transform] = [
            ToLowercaseTransform(),
            RemovePunctuationTransform(),
        ]
        idx = 0
        while len(result) < n:
            base = result[idx % len(result)] if result else "Tell me about science"
            for t in transforms:
                if len(result) >= n:
                    break
                try:
                    variant = t.evaluate(base)
                    if variant != base and variant not in result:
                        result.append(variant)
                except Exception:
                    continue
            idx += 1
            if idx > n * 2:
                break

        # 4. Pad if still short
        while len(result) < n:
            result.append(f"verification_prompt_{len(result)}")

        return result[:n]


def _benign_prompts() -> List[str]:
    _TOPICS = [
        "photosynthesis", "gravity", "the water cycle", "quantum computing",
        "machine learning", "the solar system", "DNA replication",
        "plate tectonics", "the French Revolution", "Shakespeare's Hamlet",
        "the history of jazz", "how memory works", "baking sourdough bread",
        "the life cycle of butterflies", "the theory of relativity",
    ]
    _TEMPLATES = [
        "What is {}?",
        "Tell me about {}",
        "How does {} work?",
        "Explain {} in simple terms",
        "What are the benefits of {}?",
    ]
    return [
        tpl.format(topic)
        for topic in _TOPICS
        for tpl in _TEMPLATES
    ]


def _load_base_prompts() -> List[str]:
    from prompt_loader import load_prompts
    try:
        return load_prompts(n=10)
    except Exception:
        return []


_BASE_PROMPTS: List[str] = _load_base_prompts() + _benign_prompts()[:10]


def _load_harmful_prompts(n: int = 100) -> List[str]:
    """Load up to *n* harmful prompts from the CSV (seed=42 for reproducibility)."""
    try:
        from prompt_loader import load_prompts
        return load_prompts(n=n, seed=42)
    except Exception:
        return []
