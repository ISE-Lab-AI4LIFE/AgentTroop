"""Evolutionary Synthesizer — uses genetic programming to evolve candidate programs using fitness (accuracy)."""

import json
import logging
import math
import random
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate, ContainsAnyWordPredicate,
    LengthGtPredicate, LengthLtPredicate,
    StartsWithRoleplayPredicate, ContainsSystemOverridePredicate,
    MatchesJailbreakPatternPredicate, ContainsEncodingWrapperPredicate,
    ContainsCodeBlockPredicate, ContainsDelimiterPredicate,
    HasNumberPredicate, HasEmojiPredicate, ContainsURLPredicate,
    IsRepetitivePredicate, IsGrammaticalQuestionPredicate,
    StartsWithImperativePredicate, SentimentPredicate, IntentPredicate,
    PrimitiveRegistry, default_registry,
)
from core.program import (
    Program, IfThenElseNode, PredicateNode, AndNode, OrNode, 
    NotNode, ApplyTransformNode, ThresholdNode, Node,
)
from core.types import Outcome

logger = logging.getLogger(__name__)


class EvolutionarySynthesizer:
    """Genetic-programming synthesizer using fitness (accuracy) as objective.
    
    Maintains a population of candidate programs, evolves them through
    mutation and crossover, selecting top-k by fitness each generation.
    No exact match required — uses real-valued fitness.
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 30,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        primitive_registry: Optional[PrimitiveRegistry] = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.primitive_registry = primitive_registry or default_registry
        self.executor = ProgramExecutor(self.primitive_registry)

    def synthesize(
        self,
        examples: List[Tuple[str, int]],
        base_programs: Optional[List[Program]] = None,
        k: int = 10,
    ) -> List[Program]:
        """Run evolutionary synthesis to find top-k programs."""
        start = time.time()
        if not examples:
            return []

        population = self._init_population(examples, base_programs)
        
        for gen in range(self.generations):
            fitness_scores = [self._fitness(p, examples) for p in population]
            best_fitness = max(fitness_scores) if fitness_scores else 0.0
            logger.info(
                "Evolutionary gen=%d/%d population=%d best_fitness=%.4f",
                gen + 1, self.generations, len(population), best_fitness,
            )
            
            if best_fitness >= 0.99:
                logger.info("Evolutionary synthesis converged at gen=%d", gen + 1)
                break
            
            population = self._evolve(population, fitness_scores, examples)
        
        fitness_scores = [self._fitness(p, examples) for p in population]
        scored = list(zip(fitness_scores, population))
        scored.sort(key=lambda x: -x[0])
        
        results = [p for _, p in scored[:k]]
        logger.info(
            "Evolutionary synthesis: %d candidates, best=%.4f, %.1fs",
            len(results), scored[0][0] if scored else 0.0,
            time.time() - start,
        )
        return results

    def _init_population(
        self, examples: List[Tuple[str, int]],
        base_programs: Optional[List[Program]] = None,
    ) -> List[Program]:
        population = []
        if base_programs:
            population.extend(base_programs)
        
        # Balanced initialization: only 3 keyword programs (not 10)
        keywords = self._extract_keywords(examples)
        for kw in keywords[:3]:
            for to, eo in [(1, 0), (0, 1)]:
                p = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(primitive=ContainsWordPredicate(word=kw)),
                        then_outcome=to, else_outcome=eo,
                    )
                )
                p.id = f"evo_init_{uuid.uuid4().hex[:8]}"
                population.append(p)
        
        # Multi-keyword programs (contains_any_word with top 3 keywords)
        if len(keywords) >= 3:
            for to, eo in [(1, 0), (0, 1)]:
                p = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(primitive=ContainsAnyWordPredicate(
                            words=",".join(keywords[:3])
                        )),
                        then_outcome=to, else_outcome=eo,
                    )
                )
                p.id = f"evo_multikw_{uuid.uuid4().hex[:8]}"
                population.append(p)
        
        # Length-based programs
        for threshold in [30, 80, 150]:
            for to, eo in [(1, 0), (0, 1)]:
                p = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(primitive=LengthGtPredicate(threshold=threshold)),
                        then_outcome=to, else_outcome=eo,
                    )
                )
                p.id = f"evo_len_{uuid.uuid4().hex[:8]}"
                population.append(p)
        
        # Structural predicates (surface-level removed: HasSpecialChar, IsAllCaps, ContainsLeet)
        structural_preds = [
            HasNumberPredicate(), HasEmojiPredicate(), ContainsURLPredicate(),
            IsRepetitivePredicate(), IsGrammaticalQuestionPredicate(),
            StartsWithRoleplayPredicate(), ContainsSystemOverridePredicate(),
            MatchesJailbreakPatternPredicate(), ContainsEncodingWrapperPredicate(),
            ContainsCodeBlockPredicate(), ContainsDelimiterPredicate(),
            StartsWithImperativePredicate(),
            SentimentPredicate(threshold=0.5), IntentPredicate(intent_type="malicious"),
        ]
        for sp in structural_preds:
            for to, eo in [(1, 0), (0, 1)]:
                p = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(primitive=sp),
                        then_outcome=to, else_outcome=eo,
                    )
                )
                p.id = f"evo_struct_{uuid.uuid4().hex[:8]}"
                population.append(p)
        
        random.shuffle(population)
        return population[:self.population_size]

    def _extract_keywords(self, examples: List[Tuple[str, int]]) -> List[str]:
        import re
        from collections import Counter
        refuse_words = Counter()
        for prompt, outcome in examples:
            if outcome == 1:
                words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
                refuse_words.update(w for w in words if w not in {
                    "the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "shall", "can",
                    "to", "of", "in", "for", "on", "with", "at", "by", "from",
                    "as", "into", "through", "during", "before", "after",
                    "then", "once", "here", "there", "when", "where",
                    "why", "how", "what", "which", "who", "whom", "this",
                    "these", "those", "am", "it", "its", "no", "nor", "not",
                    "or", "and", "but", "if", "so", "than", "too", "very",
                    "just", "about", "also",
                })
        return [w for w, _ in refuse_words.most_common(20)]

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
        balanced_acc = (recall_refuse + recall_accept) / 2.0

        # Penalty for surface-level predicates that cause spurious correlation
        penalty = 0.0
        root = program.root
        if hasattr(root, 'condition') and hasattr(root.condition, 'primitive'):
            pname = getattr(root.condition.primitive, 'name', '')
            surface_preds = {"length_gt", "length_lt", "has_number",
                             "has_emoji", "contains_url", "is_repetitive",
                             "is_grammatical_question"}
            if pname in surface_preds:
                penalty = 0.15

        return max(0.0, balanced_acc - penalty)

    def _evolve(
        self,
        population: List[Program],
        fitness_scores: List[float],
        examples: List[Tuple[str, int]],
    ) -> List[Program]:
        n = len(population)
        if n == 0:
            return population
        
        weights = [max(f, 0.01) for f in fitness_scores]
        new_population = []
        
        elite_count = max(1, n // 10)
        scored = list(zip(fitness_scores, population))
        scored.sort(key=lambda x: -x[0])
        for _, p in scored[:elite_count]:
            new_population.append(deepcopy(p))
        
        while len(new_population) < n:
            if random.random() < self.crossover_rate and len(population) >= 2:
                p1 = random.choices(population, weights=weights, k=1)[0]
                p2 = random.choices(population, weights=weights, k=1)[0]
                child = self._crossover(p1, p2)
            else:
                parent = random.choices(population, weights=weights, k=1)[0]
                child = deepcopy(parent)
            
            if random.random() < self.mutation_rate:
                child = self._mutate(child, examples)
            
            child.id = f"evo_{uuid.uuid4().hex[:8]}"
            new_population.append(child)
        
        return new_population[:n]

    def _crossover(self, p1: Program, p2: Program) -> Program:
        root1 = p1.root.condition if hasattr(p1.root, 'condition') else p1.root
        root2 = p2.root.condition if hasattr(p2.root, 'condition') else p2.root
        # Low clone rate — at most 20% clone
        if random.random() < 0.2:
            return deepcopy(p1)
        # Real crossover: swap conditions
        if random.random() < 0.5:
            child_cond = deepcopy(root2)
            to = p1.root.then_outcome if hasattr(p1.root, 'then_outcome') else 1
            eo = p1.root.else_outcome if hasattr(p1.root, 'else_outcome') else 0
        else:
            child_cond = deepcopy(root1)
            to = p2.root.then_outcome if hasattr(p2.root, 'then_outcome') else 1
            eo = p2.root.else_outcome if hasattr(p2.root, 'else_outcome') else 0
        child = Program(
            root=IfThenElseNode(
                condition=child_cond,
                then_outcome=to, else_outcome=eo,
            )
        )
        return child

    def _change_predicate_type(self, condition: PredicateNode) -> None:
        """Mutate the predicate type to a different kind, enabling cross-type exploration."""
        predicate_types = [
            (ContainsWordPredicate, {"word": random.choice(
                ["code", "exploit", "malware", "write", "generate", "create", "attack", "script"]
            )}),
            (ContainsAnyWordPredicate, {"words": "code,exploit,malware,attack"}),
            (LengthGtPredicate, {"threshold": random.choice([20, 50, 100, 200])}),
            (LengthLtPredicate, {"threshold": random.choice([20, 50, 100])}),
            (HasNumberPredicate, {}),
            (HasEmojiPredicate, {}),
            (ContainsURLPredicate, {}),
            (IsRepetitivePredicate, {}),
            (IsGrammaticalQuestionPredicate, {}),
            (StartsWithRoleplayPredicate, {}),
            (ContainsSystemOverridePredicate, {}),
            (MatchesJailbreakPatternPredicate, {}),
            (ContainsEncodingWrapperPredicate, {}),
            (ContainsCodeBlockPredicate, {}),
            (ContainsDelimiterPredicate, {}),
            (StartsWithImperativePredicate, {}),
            (SentimentPredicate, {"threshold": 0.5}),
            (IntentPredicate, {"intent_type": random.choice(["malicious", "harmful", "code"])}),
        ]
        weights = [
            3, 3, 1, 1, 1, 1, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ]
        cls, params = random.choices(predicate_types, weights=weights, k=1)[0]
        try:
            condition.primitive = cls(**params)
        except Exception:
            pass

    def _mutate(self, program: Program, examples: List[Tuple[str, int]]) -> Program:
        root = program.root
        if hasattr(root, 'condition'):
            condition = root.condition
            if isinstance(condition, PredicateNode):
                # 30% chance to change predicate type entirely
                if random.random() < 0.3:
                    self._change_predicate_type(condition)
                else:
                    # If it's a keyword predicate, change the keyword
                    if hasattr(condition.primitive, 'parameters'):
                        kw = condition.primitive.parameters.get('word', None)
                        if kw is not None:
                            words = self._extract_keywords(examples)
                            if words:
                                condition.primitive = ContainsWordPredicate(word=random.choice(words))
            elif isinstance(condition, ThresholdNode):
                old_t = condition.threshold
                condition.threshold = max(0.0, min(1.0, old_t + random.uniform(-0.1, 0.1)))
            if random.random() < 0.3 and hasattr(root, 'then_outcome'):
                root.then_outcome, root.else_outcome = root.else_outcome, root.then_outcome
        return program


class SynthesisStats:
    def __init__(self):
        self.duration_ms = 0.0
        self.programs_tried = 0
        self.method = "evolutionary"
        self.synthesized_candidates = 0
        self.candidates_considered = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": round(self.duration_ms, 2),
            "programs_tried": self.programs_tried,
            "method": self.method,
            "synthesized_candidates": self.synthesized_candidates,
            "candidates_considered": self.candidates_considered,
        }
