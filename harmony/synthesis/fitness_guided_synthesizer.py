"""Fitness-Guided Enumeration Synthesizer — enumerates all programs within bounds and scores by fitness."""

import logging
import signal
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate, ContainsAnyWordPredicate,
    LengthGtPredicate, LengthLtPredicate,
    StartsWithRoleplayPredicate, ContainsSystemOverridePredicate,
    MatchesJailbreakPatternPredicate, ContainsEncodingWrapperPredicate,
    ContainsCodeBlockPredicate, ContainsDelimiterPredicate,
    ContainsLeetPredicate,
    HasNumberPredicate, HasSpecialCharPredicate,
    IsAllCapsPredicate, HasEmojiPredicate, ContainsURLPredicate,
    IsRepetitivePredicate, IsGrammaticalQuestionPredicate,
    StartsWithImperativePredicate, SentimentPredicate, IntentPredicate,
    PrimitiveRegistry, default_registry,
)
from core.program import (
    Program, IfThenElseNode, PredicateNode,
)
from core.condition import registry as _condition_registry
from synthesis.grammar_exporter import GrammarExporter

logger = logging.getLogger(__name__)


class FitnessGuidedSynthesizer:
    """Enumerates all programs within depth/beam bounds and returns top-K by fitness.
    
    No CVC5, no exact match requirement. Uses real-valued fitness (accuracy).
    """

    class _Timeout:
        """Context manager for signal-based timeout (Unix only)."""
        def __init__(self, seconds: int):
            self.seconds = seconds

        def __enter__(self):
            if self.seconds > 0:
                signal.signal(signal.SIGALRM, self._handler)
                signal.alarm(self.seconds)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.seconds > 0:
                signal.alarm(0)
            return False

        @staticmethod
        def _handler(signum, frame):
            raise TimeoutError("Synthesis timed out")

    def __init__(
        self,
        max_depth: int = 6,
        beam_width: int = 500,
        timeout: int = 120,
        primitive_registry: Optional[PrimitiveRegistry] = None,
    ):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.timeout = timeout
        self.primitive_registry = primitive_registry or default_registry
        self.executor = ProgramExecutor(self.primitive_registry)

    def synthesize(
        self,
        examples: List[Tuple[str, int]],
        base_programs: Optional[List[Program]] = None,
        k: int = 10,
    ) -> List[Program]:
        start = time.time()
        if not examples:
            return []
        
        exporter = GrammarExporter(
            primitive_registry=self.primitive_registry,
            condition_registry=_condition_registry,
            max_depth=self.max_depth,
        )
        
        all_programs: List[Program] = []
        if base_programs:
            all_programs.extend(base_programs)
        
        try:
            with self._Timeout(self.timeout):
                for depth in range(1, min(self.max_depth + 1, 7)):
                    programs = exporter.enumerate_programs(max_depth=depth, examples=examples)
                    if not programs:
                        continue
                    if self.beam_width > 0 and len(programs) > self.beam_width:
                        programs = programs[:self.beam_width]
                    
                    for prog in programs:
                        prog.id = getattr(prog, 'id', None) or f"fg_{uuid.uuid4().hex[:8]}"
                        all_programs.append(prog)
        except TimeoutError:
            logger.warning(
                "Fitness-guided enumeration timed out after %ds (depth=%d, beam=%d, got %d candidates)",
                self.timeout, self.max_depth, self.beam_width, len(all_programs),
            )
        
        if not all_programs:
            logger.warning("Fitness-guided enumeration found 0 programs")
            return self._fallback_candidates(examples)
        
        scored = [(self._fitness(p, examples), p) for p in all_programs]
        scored.sort(key=lambda x: -x[0])
        
        results = [p for _, p in scored[:k]]
        
        logger.info(
            "Fitness-guided enumeration: %d candidates from %d total, best=%.4f, %.1fs",
            len(results), len(all_programs),
            scored[0][0] if scored else 0.0,
            time.time() - start,
        )
        return results
    
    def _fitness(self, program: Program, examples: List[Tuple[str, int]]) -> float:
        if not examples:
            return 0.0
        correct_refuse = 0
        total_refuse = 0
        correct_accept = 0
        total_accept = 0
        for prompt, expected in examples:
            try:
                pred = int(self.executor.execute(program, prompt))
                if expected == 1:
                    total_refuse += 1
                    if pred == expected:
                        correct_refuse += 1
                else:
                    total_accept += 1
                    if pred == expected:
                        correct_accept += 1
            except Exception:
                pass
        recall_refuse = correct_refuse / max(total_refuse, 1)
        recall_accept = correct_accept / max(total_accept, 1)
        return (recall_refuse + recall_accept) / 2.0
    
    def _fallback_candidates(self, examples: List[Tuple[str, int]]) -> List[Program]:
        keywords = self._extract_keywords(examples)
        candidates = []
        for kw in keywords[:5]:
            for to, eo in [(1, 0), (0, 1)]:
                prog = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(primitive=ContainsWordPredicate(word=kw)),
                        then_outcome=to, else_outcome=eo,
                    )
                )
                prog.id = f"fg_fallback_{uuid.uuid4().hex[:8]}"
                candidates.append(prog)
        return candidates
    
    def _extract_keywords(self, examples: List[Tuple[str, int]]) -> List[str]:
        import re
        from collections import Counter
        refuse_words = Counter()
        for prompt, outcome in examples:
            if outcome == 1:
                words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
                refuse_words.update(w for w in words)
        return [w for w, _ in refuse_words.most_common(20)]
