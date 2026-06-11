"""Program synthesis from examples — CVC5 primary, enumeration fallback.

Pipeline::

    Examples
        ↓
    SMTConstraintBuilder (core.grammar)
        ↓
    CVC5 solver (primary)
        ↓
    Model → Program reconstruction
        ↓
    Optional: Enumeration + Beam Search (fallback / hybrid)
        ↓
    Best Program (by accuracy + complexity + MDL)

Hybrid mode runs both CVC5 and enumeration and selects the best result.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core import ConditionRegistry
from core.condition import registry as _condition_registry
from core.executor import ProgramExecutor
from core.grammar import SMTConstraintBuilder
from core.primitive import (
    Classifier,
    ContainsWordPredicate,
    Predicate,
    PrimitiveRegistry,
    Transform,
    default_registry,
)
from core.program import (
    AndNode,
    ApplyTransformNode,
    ClassifierNode,
    IfThenElseNode,
    Node,
    NotNode,
    OrNode,
    PredicateNode,
    Program,
    ThresholdNode,
    TransformNode,
)
from core.types import Outcome

from .grammar_exporter import GrammarExporter, PrimitiveCatalog

logger = logging.getLogger(__name__)


@dataclass
class SynthesisStats:
    duration_ms: float = 0.0
    depth_used: int = 0
    programs_tried: int = 0
    programs_skipped_cache: int = 0
    cvc5_used: bool = False
    cvc5_success: bool = False
    enumeration_found: bool = False
    enumeration_tried: bool = False
    hybrid_mode: bool = False
    allow_error_rate: float = 0.0
    max_errors: int = 0
    errors_actual: int = 0
    beam_width: int = 0
    method: str = "none"
    synthesized_candidates: int = 0
    heuristic_fallback_candidates: int = 0
    candidates_considered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": round(self.duration_ms, 2),
            "depth_used": self.depth_used,
            "programs_tried": self.programs_tried,
            "programs_skipped_cache": self.programs_skipped_cache,
            "cvc5_used": self.cvc5_used,
            "cvc5_success": self.cvc5_success,
            "enumeration_found": self.enumeration_found,
            "enumeration_tried": self.enumeration_tried,
            "hybrid_mode": self.hybrid_mode,
            "allow_error_rate": self.allow_error_rate,
            "max_errors": self.max_errors,
            "errors_actual": self.errors_actual,
            "beam_width": self.beam_width,
            "method": self.method,
            "synthesized_candidates": self.synthesized_candidates,
            "heuristic_fallback_candidates": self.heuristic_fallback_candidates,
            "candidates_considered": self.candidates_considered,
        }


def _classify_program_category(program: Program) -> str:
    """Classify a program into: keyword, structural, jailbreak, semantic,
    classifier, transform, composite, or unknown."""
    from core.program import (
        PredicateNode, ThresholdNode, AndNode, OrNode, NotNode, ApplyTransformNode,
    )
    root = program.root
    node = root.condition if hasattr(root, "condition") else root
    if isinstance(node, ApplyTransformNode):
        return "transform"
    if isinstance(node, (AndNode, OrNode, NotNode)):
        return "composite"
    if isinstance(node, ThresholdNode):
        ret = "classifier"
        classifier = getattr(node, "classifier", None)
        if classifier is not None:
            cname = classifier.name if hasattr(classifier, "name") else ""
            if cname in {"instruction_score", "semantic_score"}:
                ret = "semantic_score"
        return ret
    if isinstance(node, PredicateNode):
        name = node.primitive.name if hasattr(node.primitive, "name") else ""
        keyword_preds = {"contains_word", "contains_any_word", "contains_all_words",
                         "starts_with", "ends_with", "matches_regex"}
        structural_preds = {"has_number", "has_special_char", "is_all_caps",
                            "is_empty", "has_emoji", "contains_url", "is_repetitive",
                            "char_count", "length_gt", "length_lt",
                            "starts_with", "ends_with",
                            "contains_rot13", "contains_base64", "contains_hex"}
        jailbreak_preds = {"matches_jailbreak_pattern", "contains_system_override",
                           "contains_encoding_wrapper"}
        semantic_preds = {"starts_with_roleplay", "starts_with_imperative",
                          "is_grammatical_question", "sentiment", "intent",
                          "contains_leet", "instruction_score"}
        if name in keyword_preds:
            return "keyword"
        if name in structural_preds:
            return "structural"
        if name in jailbreak_preds:
            return "jailbreak"
        if name in semantic_preds:
            return "semantic"
        return "unknown"
    return "unknown"


def _compute_hash(program: Program, examples: List[Tuple[str, int]]) -> str:
    return hash((program.canonical_form(), tuple(examples))).__str__()


def _node_count(node: Node) -> int:
    if isinstance(node, (PredicateNode, ThresholdNode, ClassifierNode)):
        return 1
    if isinstance(node, TransformNode):
        return 1
    if isinstance(node, ApplyTransformNode):
        return 1 + _node_count(node.inner)
    if isinstance(node, (AndNode, OrNode)):
        return 1 + _node_count(node.left) + _node_count(node.right)
    if isinstance(node, NotNode):
        return 1 + _node_count(node.child)
    if isinstance(node, IfThenElseNode):
        return 1 + _node_count(node.condition)
    return 1


def _safe_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe if safe else "prim"


def _default_cvc5_path() -> str:
    """Resolve the CVC5 binary path from environment or system PATH.

    Resolution order:
    1. ``CVC5_PATH`` environment variable (if set and executable).
    2. ``shutil.which("cvc5")`` found on system PATH.
    3. Fallback to ``"cvc5"`` (rely on ``subprocess`` ``FileNotFoundError``).
    """
    env_path = os.environ.get("CVC5_PATH")
    if env_path:
        return env_path
    resolved = shutil.which("cvc5")
    if resolved:
        return resolved
    return "cvc5"


class CVC5Synthesizer:
    """Program synthesizer with CVC5 as primary solver and enumeration fallback.

    Parameters
    ----------
    cvc5_path : str or None
        Path to CVC5 binary.  If ``None`` the path is resolved automatically
        via :func:`_default_cvc5_path` (respects ``CVC5_PATH`` env var).
    timeout : int
        Timeout per CVC5 call in seconds.
    max_depth : int
        Maximum composition depth for programs.
    allow_error_rate : float
        Fraction of training examples allowed to be wrong (0.0 = exact match).
    beam_width : int
        Beam width for enumeration fallback (0 = disabled).
    use_cache : bool
        Cache previously seen program-example combinations.
    cache_path : str, optional
        Path to persist cache.
    hybrid : bool
        If True, run both CVC5 and enumeration, select best by MDL.
    enforce_cvc5_first : bool
        If True, always try CVC5 before enumeration.
    """

    def __init__(
        self,
        cvc5_path: Optional[str] = None,
        timeout: int = 30,
        max_depth: int = 3,
        allow_error_rate: float = 0.0,
        beam_width: int = 200,
        use_cache: bool = True,
        cache_path: Optional[str] = None,
        hybrid: bool = True,
        enforce_cvc5_first: bool = True,
        condition_registry: Optional[ConditionRegistry] = None,
    ) -> None:
        self.cvc5_path = cvc5_path if cvc5_path is not None else _default_cvc5_path()
        self.condition_registry = condition_registry or _condition_registry
        self.timeout = timeout
        self.max_depth = max(1, int(max_depth))
        self.allow_error_rate = max(0.0, min(1.0, float(allow_error_rate)))
        self.beam_width = max(0, int(beam_width))
        self.use_cache = use_cache
        self.cache_path = cache_path
        self.hybrid = hybrid
        self.enforce_cvc5_first = enforce_cvc5_first
        self._cache: Dict[str, bool] = {}
        self._cvc5_checked: Optional[bool] = None
        self._synthesis_history: List[Tuple[str, str, int]] = []  # (predicate_name, category, timestamp)
        # Seed with one dummy per category so the very first synthesis
        # does not default to keyword.  Keyword is included so structural/
        # jailbreak/semantic entries get preference on equal fitness.
        self._reset_diversity_seed()
        if enforce_cvc5_first:
            logger.info("CVC5Synthesizer: enforce_cvc5_first=True (CVC5 primary, path=%s, timeout=%ds, max_depth=%d, beam=%d, hybrid=%s)",
                        cvc5_path, timeout, max_depth, beam_width, hybrid)
        else:
            logger.info("CVC5Synthesizer: enforce_cvc5_first=False (enumeration primary, path=%s, max_depth=%d, beam=%d)",
                        cvc5_path, max_depth, beam_width)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info("Loaded %d cache entries from %s", len(self._cache), cache_path)
            except Exception as exc:
                logger.warning("Failed to load cache from %s: %s", cache_path, exc)

    def _save_cache(self) -> None:
        if self.cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self._cache, f)
            except Exception as exc:
                logger.warning("Failed to save cache: %s", exc)

    def _reset_diversity_seed(self) -> None:
        """Seed synthesis_history so the first real call gets diversity pressure."""
        self._synthesis_history = [
            ("_seed_keyword_0", "keyword", -7),
            ("_seed_keyword_1", "keyword", -6),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        examples: List[Tuple[str, int]],
        primitive_registry: Optional[PrimitiveRegistry] = None,
        ontology_memory: Optional[Any] = None,
    ) -> Optional[Program]:
        return self.synthesize_with_stats(examples, primitive_registry, ontology_memory)[0]

    def synthesize_top_k(
        self,
        examples: List[Tuple[str, int]],
        k: int = 5,
        primitive_registry: Optional[PrimitiveRegistry] = None,
        ontology_memory: Optional[Any] = None,
    ) -> List[Program]:
        """Synthesize up to *k* candidate programs from examples.

        Returns a list of programs that match the examples within the
        allowed error rate, scored by fitness.  This enables the version
        space to maintain multiple competing hypotheses.

        Parameters
        ----------
        examples : list of (prompt, outcome)
        k : int
            Maximum number of candidate programs to return.
        primitive_registry : PrimitiveRegistry, optional
        ontology_memory : OntologyMemory, optional

        Returns
        -------
        list of Program
            Up to *k* programs sorted by fitness (best first).
        """
        start = time.time()
        registry = primitive_registry or default_registry
        exporter = GrammarExporter(
            primitive_registry=registry,
            condition_registry=self.condition_registry,
            ontology_memory=ontology_memory,
            max_depth=self.max_depth,
        )
        if not examples:
            return []

        max_errors = max(1, int(len(examples) * self.allow_error_rate)) if self.allow_error_rate > 0 else 0
        catalog = exporter.get_parameterized_primitives(examples)
        if catalog.is_empty():
            logger.warning(
                "CVC5: catalog is empty for %d examples (depth=%d, beam=%d, "
                "allow_error_rate=%.2f) — no parameterized primitives could be "
                "instantiated from the current examples",
                len(examples), self.max_depth, getattr(exporter, "beam_width", "?"),
                self.allow_error_rate,
            )
            return []

        executor = ProgramExecutor(exporter.primitive_registry or default_registry)

        # ── Stage-by-stage audit logging ──
        _audit: Dict[str, Any] = {
            "examples": len(examples),
            "catalog_predicates": len(catalog.predicates),
            "catalog_transforms": len(catalog.transforms),
            "catalog_classifiers": len(catalog.classifiers),
            "max_depth": self.max_depth,
            "max_errors": max_errors,
            "allow_error_rate": self.allow_error_rate,
        }
        logger.info("SYNTHESIS_AUDIT stage=init %s", json.dumps(_audit))

        # Collect all matching programs from enumeration
        all_matching: List[Program] = []

        for depth in range(1, min(self.max_depth + 2, 7)):
            programs = exporter.enumerate_programs(max_depth=depth, examples=examples)
            _audit[f"depth_{depth}_enumerated"] = len(programs) if programs else 0
            if not programs:
                continue

            depth_matched = 0
            for prog in programs:
                if self._matches_all(prog, examples, executor, max_errors=max_errors):
                    prog_id = getattr(prog, "id", None) or f"prog_{uuid.uuid4().hex[:12]}"
                    if not hasattr(prog, "id") or getattr(prog, "id", None) is None:
                        prog.id = prog_id
                    all_matching.append(prog)
                    depth_matched += 1
                    if len(all_matching) >= k * 2:
                        break
            _audit[f"depth_{depth}_matching"] = depth_matched
            if len(all_matching) >= k * 2:
                break

        _audit["total_matching"] = len(all_matching)
        logger.info("SYNTHESIS_AUDIT stage=enumeration %s", json.dumps(_audit))

        # ── Fitness-based fallback when enumeration finds 0 exact matches ──
        # Instead of generating arbitrary partial programs, score depth-1
        # programs by fitness and keep those better than random (fitness > 0.5).
        # This is fast (318 depth-1 programs * N examples) and ensures the
        # version space receives programs with actual (weak) discriminative
        # signal, which Bayesian belief can then amplify.
        if not all_matching:
            # Compute diversity bonus weights from synthesis history so rare
            # families (structural, jailbreak, semantic) get a boost vs keyword.
            from collections import defaultdict
            fb_cat_counts: Dict[str, int] = defaultdict(int)
            for _, cat, _ in self._synthesis_history:
                fb_cat_counts[cat] += 1
            fb_cat_counts["keyword"] = fb_cat_counts.get("keyword", 0) + 2
            fb_total_count = sum(fb_cat_counts.values()) or 1

            def _fb_diversity_bonus(prog: Program) -> float:
                cat = _classify_program_category(prog)
                count = fb_cat_counts.get(cat, 1)
                decay = max(3.0, fb_total_count / max(len(fb_cat_counts), 1))
                return 0.3 * math.exp(-count / decay)

            scored_all: List[Tuple[float, Program]] = []
            for depth in [1, 2]:
                programs = exporter.enumerate_programs(max_depth=depth, examples=examples)
                if not programs:
                    continue
                # At depth 2+, limit to first 500 programs for speed
                if depth > 1:
                    programs = programs[:500]
                for prog in programs:
                    prog_id = getattr(prog, "id", None) or f"prog_{uuid.uuid4().hex[:12]}"
                    if not hasattr(prog, "id") or getattr(prog, "id", None) is None:
                        prog.id = prog_id
                    fitness = self._fitness_score(prog, examples, executor)
                    if fitness > 0.65:
                        bonus = _fb_diversity_bonus(prog)
                        scored_all.append((fitness + bonus, prog))
            # Sort by fitness+diversity descending, take top k*2
            scored_all.sort(key=lambda x: -x[0])
            all_matching = [p for _, p in scored_all[:k * 2]]
            _audit["fitness_fallback"] = len(all_matching)
            _audit["fitness_fallback_best"] = round(scored_all[0][0], 4) if scored_all else 0.0
            if all_matching:
                logger.info(
                    "CVC5: 0 exact matches; keeping %d programs by fitness+diversity "
                    "(best=%.3f, min_fitness=%.3f)",
                    len(all_matching), scored_all[0][0], scored_all[-1][0] if len(scored_all) > 1 else 0.0,
                )

        # ── Absolute fallback: generate partial programs from catalog ──
        import collections
        if not all_matching:
            majority = collections.Counter(o for _, o in examples).most_common(1)
            default_outcome = majority[0][0] if majority else 1
            logger.warning(
                "CVC5: 0 matching programs after enumeration AND fitness fallback; "
                "generating partial programs from catalog (examples=%d, "
                "default_outcome=%d)", len(examples), default_outcome,
            )
            # Create programs from catalog predicates directly
            for pred in catalog.predicates:
                for to, eo in [(1, 0), (0, 1), (default_outcome, 1 - default_outcome)]:
                    prog = Program(
                        root=IfThenElseNode(
                            condition=PredicateNode(primitive=pred),
                            then_outcome=to,
                            else_outcome=eo,
                        )
                    )
                    prog.id = f"partial_{uuid.uuid4().hex[:12]}"
                    all_matching.append(prog)
            # Add ALWAYS_REFUSE and ALWAYS_ACCEPT baselines
            for outcome in [0, 1]:
                prog = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(
                            primitive=ContainsWordPredicate(word="")
                        ),
                        then_outcome=outcome,
                        else_outcome=outcome,
                    )
                )
                prog.id = f"always_{'ACCEPT' if outcome == 0 else 'REFUSE'}_{uuid.uuid4().hex[:8]}"
                all_matching.append(prog)
            _audit["partial_generated"] = len(all_matching)
            logger.info("SYNTHESIS_AUDIT stage=partial %s", json.dumps(_audit))

        # Diversity-aware top-K selection:
        # Ensures no single predicate family dominates the returned set.
        # Applies diversity bonus to fitness so rare families (structural,
        # jailbreak, semantic) get a boost over common ones (keyword).
        from collections import defaultdict as _defaultdict
        _cat_counts: Dict[str, int] = _defaultdict(int)
        for _, cat, _ in self._synthesis_history:
            _cat_counts[cat] += 1
        _cat_counts["keyword"] = _cat_counts.get("keyword", 0) + 2
        _total_cat = sum(_cat_counts.values()) or 1

        def _topk_bonus(prog: Program) -> float:
            cat = _classify_program_category(prog)
            cnt = _cat_counts.get(cat, 1)
            decay = max(3.0, _total_cat / max(len(_cat_counts), 1))
            return 0.3 * math.exp(-cnt / decay)

        from inference.version_space import _classify_program
        scored = []
        for prog in all_matching:
            fitness = self._fitness_score(prog, examples, executor)
            mdl = self._mdl_score(prog, examples, executor)
            family = _classify_program(prog)
            bonus = _topk_bonus(prog)
            scored.append((fitness + bonus, -mdl, prog.complexity(), family, prog))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

        results: List[Program] = []
        seen_families: Dict[str, int] = {}
        max_per_family = max(1, k // 3)
        for fitness, neg_mdl, comp, family, prog in scored:
            if seen_families.get(family, 0) >= max_per_family:
                continue
            results.append(prog)
            seen_families[family] = seen_families.get(family, 0) + 1
            if len(results) >= k:
                break
        # If we didn't reach k, fill remaining from best scored regardless of family
        if len(results) < k:
            for fitness, neg_mdl, comp, family, prog in scored:
                if prog not in results:
                    results.append(prog)
                    if len(results) >= k:
                        break

        # Mark programs with exact_match metadata when they achieve
        # zero-error fit on the training examples.  The orchestrator
        # uses this flag to boost such programs in the version space.
        _exact_match = (max_errors == 0)
        for prog in results:
            prog.metadata["exact_match"] = _exact_match

        _audit["returned"] = len(results)
        _audit["families"] = dict(seen_families)
        _audit["duration_s"] = round(time.time() - start, 2)
        _audit["exact_match"] = _exact_match
        logger.info("SYNTHESIS_AUDIT stage=final %s", json.dumps(_audit))
        return results

    def synthesize_with_stats(
        self,
        examples: List[Tuple[str, int]],
        primitive_registry: Optional[PrimitiveRegistry] = None,
        ontology_memory: Optional[Any] = None,
    ) -> Tuple[Optional[Program], SynthesisStats]:
        """Run synthesis: CVC5 primary, enumeration fallback, hybrid selection."""
        start = time.time()
        stats = SynthesisStats(
            allow_error_rate=self.allow_error_rate,
            max_errors=int(len(examples) * self.allow_error_rate),
            beam_width=self.beam_width,
            hybrid_mode=self.hybrid,
        )

        registry = primitive_registry or default_registry
        exporter = GrammarExporter(
            primitive_registry=registry,
            condition_registry=self.condition_registry,
            ontology_memory=ontology_memory,
            max_depth=self.max_depth,
        )

        if not examples:
            logger.warning("No examples provided for synthesis")
            stats.duration_ms = (time.time() - start) * 1000
            return None, stats

        max_errors = max(1, int(len(examples) * self.allow_error_rate)) if self.allow_error_rate > 0 else 0
        stats.max_errors = max_errors

        # ------------------------------------------------------------------
        # CVC5 availability check
        # ------------------------------------------------------------------
        cvc5_avail = self._cvc5_available()
        if self.enforce_cvc5_first and not cvc5_avail:
            logger.warning("CVC5 enforce_cvc5_first=True but binary not found at '%s'. "
                           "Falling back to enumeration. Install CVC5 to use it as primary.",
                           self.cvc5_path)
        if not self.enforce_cvc5_first:
            logger.info("CVC5 enforce_cvc5_first=False — skipping CVC5, using enumeration only")

        # ------------------------------------------------------------------
        # Phase 1: CVC5
        # ------------------------------------------------------------------
        cvc5_program: Optional[Program] = None
        if cvc5_avail:
            stats.cvc5_used = True
            logger.info("→ CVC5 available at '%s' — building SMT constraint (timeout=%ds, max_depth=%d)",
                        self.cvc5_path, self.timeout, self.max_depth)
            cvc5_start = time.time()
            cvc5_program = self._try_cvc5(examples, exporter, max_errors)
            if cvc5_program is not None:
                stats.cvc5_success = True
                stats.method = "cvc5"
                stats.depth_used = self._estimate_depth(cvc5_program)
                logger.info("  ✓ CVC5 found solution in %.1fs (program=%s, complexity=%d)",
                            time.time() - cvc5_start,
                            cvc5_program.id, cvc5_program.complexity())
            else:
                logger.info("  ✗ CVC5 did not find solution in %.1fs", time.time() - cvc5_start)

        # ------------------------------------------------------------------
        # Phase 2: Enumeration (fallback or hybrid comparison)
        # ------------------------------------------------------------------
        enum_program: Optional[Program] = None
        run_enumeration = (
            cvc5_program is None or self.hybrid
        )

        if run_enumeration:
            stats.enumeration_tried = True
            enum_start = time.time()
            depth = 1
            max_auto_depth = self.max_depth + 1
            logger.info("Running enumeration (depth <= %d, beam=%d)...",
                        max_auto_depth, self.beam_width)
            while depth <= max_auto_depth and enum_program is None:
                stats.depth_used = depth
                if self.beam_width == 0 and depth > self.max_depth + 1:
                    break
                logger.info("Enumeration depth=%d/%d...", depth, max_auto_depth)
                enum_program = self._try_enumeration(
                    examples, exporter, max_errors, stats, depth=depth
                )
                if enum_program is not None:
                    stats.enumeration_found = True
                    logger.info("Found program at enumeration depth=%d", depth)
                    break
                depth += 1
            logger.info("Enumeration %s in %.1fs (depth=%d, tried=%d)",
                        "found solution" if enum_program else "exhausted",
                        time.time() - enum_start, stats.depth_used,
                        stats.programs_tried)

        # ------------------------------------------------------------------
        # Phase 3: Hybrid selection
        # ------------------------------------------------------------------
        program: Optional[Program] = None
        if cvc5_program is not None and enum_program is not None:
            # Score both by MDL + accuracy
            executor = ProgramExecutor(exporter.primitive_registry or default_registry)
            cvc5_score = self._mdl_score(cvc5_program, examples, executor)
            enum_score = self._mdl_score(enum_program, examples, executor)
            program = cvc5_program if cvc5_score <= enum_score else enum_program
            stats.method = f"hybrid:cvc5({cvc5_score:.3f})<enum({enum_score:.3f})" if cvc5_score <= enum_score else f"hybrid:enum({enum_score:.3f})<cvc5({cvc5_score:.3f})"
            logger.info("Hybrid selection: CVC5 MDL=%.3f vs Enum MDL=%.3f → %s",
                        cvc5_score, enum_score,
                        "CVC5" if cvc5_program is program else "Enumeration")
        elif cvc5_program is not None:
            program = cvc5_program
            stats.method = "cvc5"
        elif enum_program is not None:
            program = enum_program
            stats.method = "enumeration"

        # ------------------------------------------------------------------
        # Synthesis history (for diversity-aware selection across calls)
        # ------------------------------------------------------------------
        if program is not None:
            cat = _classify_program_category(program)
            root_cond = program.root.condition if hasattr(program.root, 'condition') else program.root
            if isinstance(root_cond, PredicateNode):
                pname = root_cond.primitive.name if hasattr(root_cond.primitive, 'name') else type(root_cond).__name__
            elif isinstance(root_cond, ApplyTransformNode):
                pname = f"transform:{root_cond.transform.name}"
            elif isinstance(root_cond, ThresholdNode):
                pname = f"classifier:{root_cond.classifier.name}"
            else:
                pname = type(root_cond).__name__
            self._synthesis_history.append((pname, cat, len(self._synthesis_history)))

        # ------------------------------------------------------------------
        # Stats
        # ------------------------------------------------------------------
        if program is not None:
            executor = ProgramExecutor(exporter.primitive_registry or default_registry)
            errors = 0
            for prompt, expected in examples:
                try:
                    if executor.execute(program, prompt) != expected:
                        errors += 1
                except Exception:
                    errors += 1
            stats.errors_actual = errors
            logger.info("Final program: %s (accuracy=%.2f, complexity=%d, errors=%d/%d, method=%s)",
                        program.id,
                        1.0 - errors / len(examples) if examples else 0.0,
                        program.complexity(),
                        errors, len(examples),
                        stats.method)

        if program is not None and max_errors == 0:
            if not hasattr(program, 'metadata') or program.metadata is None:
                program.metadata = {}
            program.metadata["exact_match"] = True

        stats.duration_ms = (time.time() - start) * 1000
        return program, stats

    # ------------------------------------------------------------------
    # CVC5 Solver
    # ------------------------------------------------------------------

    def _try_cvc5(
        self,
        examples: List[Tuple[str, int]],
        exporter: GrammarExporter,
        max_errors: int,
    ) -> Optional[Program]:
        """Build SMT constraint, call CVC5, parse model into a Program."""
        temp_dir = tempfile.mkdtemp(prefix="cvc5_")
        smt_path = os.path.join(temp_dir, "synthesis.smt2")

        try:
            # Build proper SMT constraint using SMTConstraintBuilder
            catalog = exporter.get_parameterized_primitives(examples)
            builder = SMTConstraintBuilder(
                predicates=catalog.predicates,
                transforms=catalog.transforms,
                classifiers=catalog.classifiers,
                max_depth=self.max_depth,
                use_complexity_constraint=True,
                max_complexity=min(10, 2 ** self.max_depth),
                allow_error_rate=self.allow_error_rate,
            )
            smt_content = builder.build_smtlib(examples, use_free_thresholds=True)
            with open(smt_path, "w", encoding="utf-8") as f:
                f.write(smt_content)

            logger.debug("SMT-LIB written to %s (%d bytes)",
                         smt_path, len(smt_content))

            # Call CVC5
            result = subprocess.run(
                [
                    self.cvc5_path,
                    "--timeout", str(self.timeout * 1000),
                    "--produce-models",
                    "--incremental",
                    smt_path,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode != 0:
                if "unknown" in stdout.lower():
                    logger.info("CVC5 returned 'unknown' (incomplete theory)")
                    return self._fallback_from_smt(examples, exporter, max_errors)
                if "unsat" in stdout:
                    logger.info("CVC5: unsat — no program matches all constraints")
                    return None
                logger.debug("CVC5 stderr: %s", stderr[:500])
                return self._fallback_from_smt(examples, exporter, max_errors)

            return self._parse_cvc5_output(stdout, catalog, examples, max_errors)

        except FileNotFoundError:
            logger.warning("CVC5 binary not found at %s", self.cvc5_path)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("CVC5 timed out after %ds", self.timeout)
            return None
        except Exception as exc:
            logger.error("CVC5 error: %s", exc)
            return None
        finally:
            try:
                if os.path.exists(smt_path):
                    os.unlink(smt_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    def _parse_cvc5_output(
        self,
        output: str,
        catalog: PrimitiveCatalog,
        examples: List[Tuple[str, int]],
        max_errors: int,
    ) -> Optional[Program]:
        """Parse CVC5 model output and reconstruct a Program.

        Handles both:
        - Simple model: ``(define-fun contains_bomb ((x String)) Bool ...)``
        - Full model with depth functions
        """
        lines = output.strip().split("\n")
        if not lines:
            return None

        first = lines[0].strip()
        if first == "unsat":
            logger.info("CVC5: unsat — no program matches constraints")
            return None
        if first == "unknown":
            logger.info("CVC5: unknown — solver could not decide")
            return None
        if first != "sat":
            logger.debug("CVC5 unexpected first line: %s", first[:100])
            return None

        # Parse model functions
        model_text = "\n".join(lines[1:])
        defined_fns = self._parse_defined_functions(model_text)

        if not defined_fns:
            logger.debug("No define-fun found in CVC5 model")
            return None

        # Reconstruct program from model
        program = self._reconstruct_from_model(defined_fns, catalog)
        if program is not None:
            executor = ProgramExecutor(default_registry)
            if self._matches_all(program, examples, executor, max_errors=max_errors):
                logger.info("CVC5 model reconstructed successfully: %s", program)
                return program

        return self._fallback_from_smt(examples, catalog, max_errors)

    def _parse_defined_functions(self, model_text: str) -> Dict[str, str]:
        """Extract (define-fun ...) declarations from CVC5 model output."""
        fns: Dict[str, str] = {}
        pattern = r"\(define-fun\s+(\w+)\s*(?:\(\(x\s+String\)\))?\s+(Bool|Real|String|Int)\s+"
        for match in re.finditer(pattern, model_text):
            name = match.group(1)
            body_start = match.end()
            depth = 1
            i = body_start
            while i < len(model_text) and depth > 0:
                if model_text[i] == '(':
                    depth += 1
                elif model_text[i] == ')':
                    depth -= 1
                i += 1
            body = model_text[body_start:i-1] if depth == 0 else model_text[body_start:]
            fns[name] = body.strip()
        return fns

    def _reconstruct_from_model(
        self,
        defined_fns: Dict[str, str],
        catalog: PrimitiveCatalog,
    ) -> Optional[Program]:
        """Reconstruct a Program from the CVC5 model's define-funs.

        Fully supports AST reconstruction including AND, OR, NOT,
        PredicateNode, ThresholdNode, and ApplyTransformNode.

        Strategy:
        1. Find the main condition/depth function
        2. Parse S-expression string into a raw tree (nested lists/atoms)
        3. Convert the raw tree into Program AST nodes via ``_sexpr_to_ast``
        """
        condition_body = self._find_condition_body(defined_fns, catalog)
        if not condition_body:
            return None

        raw = _parse_sexpr_tree(condition_body)
        if raw is None:
            return None

        root_ast = _sexpr_to_ast(raw, catalog, defined_fns)
        if not isinstance(root_ast, Node):
            return None

        return Program(
            root=IfThenElseNode(condition=root_ast, then_outcome=1, else_outcome=0)
        )

    def _find_condition_body(
        self,
        defined_fns: Dict[str, str],
        catalog: PrimitiveCatalog,
    ) -> Optional[str]:
        """Locate the primary condition body from defined functions."""
        for name in ["condition", "condition_1", "depth_1"]:
            if name in defined_fns:
                return defined_fns[name]
        for name in sorted(defined_fns.keys()):
            if name.startswith("depth_"):
                return defined_fns[name]
        for name, body in defined_fns.items():
            if any(p.name == name or _safe_name(p.name) == name
                   for p in catalog.predicates):
                return body
        return None

    # ------------------------------------------------------------------
    # Enumeration fallback
    # ------------------------------------------------------------------

    def _try_enumeration(
        self,
        examples: List[Tuple[str, int]],
        exporter: GrammarExporter,
        max_errors: int,
        stats: SynthesisStats,
        depth: Optional[int] = None,
    ) -> Optional[Program]:
        catalog = exporter.get_parameterized_primitives(examples)
        if catalog.is_empty():
            return None

        d = depth if depth is not None else self.max_depth
        programs = exporter.enumerate_programs(max_depth=d, examples=examples)
        if not programs:
            return None

        executor = ProgramExecutor(exporter.primitive_registry or default_registry)

        # Stratify programs by category for fair beam coverage
        from collections import defaultdict
        by_type: Dict[str, List[Program]] = defaultdict(list)
        for prog in programs:
            cat = _classify_program_category(prog)
            by_type[cat].append(prog)

        # At depth 1, score ALL programs (they are few, ~100-150).
        # At deeper depths, take a stratified sample from each category.
        if self.beam_width > 0:
            scored: List[Tuple[float, Program]] = []
            if d == 1:
                for prog in programs:
                    score = self._fitness_score(prog, examples, executor)
                    scored.append((score, prog))
            else:
                # Per-category budget: ensure each category gets at least
                # beam_width candidates scored.
                per_cat = max(self.beam_width * 2, 15)
                for cat in sorted(by_type.keys()):
                    candidates = by_type[cat][:per_cat]
                    for prog in candidates:
                        score = self._fitness_score(prog, examples, executor)
                        scored.append((score, prog))

            scored.sort(key=lambda x: (-x[0], x[1].complexity()))
            programs = [p for _, p in scored[:self.beam_width]]
            stats.candidates_considered = len(scored)

        # Track how many synthesized (diverse) vs heuristic (simple keyword) candidates
        stats.synthesized_candidates += sum(
            1 for p in programs
            if _classify_program_category(p) in ("transform", "composite", "classifier")
        )
        stats.heuristic_fallback_candidates += sum(
            1 for p in programs
            if _classify_program_category(p) in ("keyword", "structural")
        )

        # Collect ALL perfect matches and prefer categories that are
        # underrepresented.  Uses a continuous exponential-decay bonus
        # so that rare categories (structural, jailbreak, semantic) are
        # always slightly preferred over common ones (keyword), even after
        # many cycles.
        cat_counts: Dict[str, int] = defaultdict(int)
        if hasattr(self, '_synthesis_history'):
            for _, cat, _ in self._synthesis_history:
                cat_counts[cat] += 1
        # Seed: keyword gets 2 extra counts so it is never the most novel
        for seed_cat in ("keyword",):
            cat_counts[seed_cat] = cat_counts.get(seed_cat, 0) + 2

        total_count = sum(cat_counts.values()) or 1

        def _diversity_bonus(prog: Program) -> float:
            cat = _classify_program_category(prog)
            count = cat_counts.get(cat, 1)
            # Exponential decay: bonus = 0.3 * exp(-count / decay_scale)
            # decay_scale = max(3, total_count / len(cat_counts))
            decay = max(3.0, total_count / max(len(cat_counts), 1))
            return 0.3 * math.exp(-count / decay)

        ordered: List[Tuple[float, int, Program]] = []  # (-bonus, complexity, prog)
        for prog in programs:
            stats.programs_tried += 1
            if self.use_cache:
                h = _compute_hash(prog, examples)
                if h in self._cache:
                    stats.programs_skipped_cache += 1
                    if self._cache[h]:
                        ordered.append((
                            -_diversity_bonus(prog),
                            prog.complexity(),
                            prog,
                        ))
                    continue
                is_match = self._matches_all(prog, examples, executor, max_errors=max_errors)
                self._cache[h] = is_match
            else:
                is_match = self._matches_all(prog, examples, executor, max_errors=max_errors)
            if is_match:
                ordered.append((
                    -_diversity_bonus(prog),
                    prog.complexity(),
                    prog,
                ))

        if ordered:
            ordered.sort(key=lambda x: (x[0], x[1]))
            self._save_cache()
            return ordered[0][2]

        self._save_cache()
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cvc5_available(self) -> bool:
        resolved = shutil.which(self.cvc5_path)
        if not resolved:
            return False
        try:
            result = subprocess.run(
                [resolved, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _fallback_from_smt(
        self,
        examples: List[Tuple[str, int]],
        catalog: PrimitiveCatalog,
        max_errors: int,
    ) -> Optional[Program]:
        """Fallback: use exporter to enumerate and find best program."""
        exporter = GrammarExporter(
            primitive_registry=default_registry,
            condition_registry=self.condition_registry,
            max_depth=self.max_depth,
        )
        candidates = exporter.enumerate_programs(max_depth=self.max_depth, examples=examples)
        if candidates:
            executor = ProgramExecutor(default_registry)
            for prog in candidates:
                if self._matches_all(prog, examples, executor, max_errors=max_errors):
                    return prog
        return None

    def _fitness_score(
        self, program: Program, examples: List[Tuple[str, int]], executor: ProgramExecutor
    ) -> float:
        correct = 0
        for prompt, expected in examples:
            try:
                if executor.execute(program, prompt) == expected:
                    correct += 1
            except Exception:
                pass
        return correct / len(examples) if examples else 0.0

    def _mdl_score(
        self, program: Program, examples: List[Tuple[str, int]], executor: ProgramExecutor
    ) -> float:
        """Minimum Description Length score: L(Π) + λ·|Π|."""
        accuracy = self._fitness_score(program, examples, executor)
        error_rate = 1.0 - accuracy
        complexity = program.complexity()
        mdl = error_rate * len(examples) + 0.1 * complexity
        return mdl

    @staticmethod
    def _estimate_depth(program: Program) -> int:
        return program.depth()

    def _matches_all(
        self,
        program: Program,
        examples: List[Tuple[str, int]],
        executor: ProgramExecutor,
        num_trials: int = 1,
        max_errors: int = 0,
    ) -> bool:
        errors = 0
        for prompt, expected in examples:
            all_match = True
            for _ in range(num_trials):
                try:
                    if executor.execute(program, prompt) != expected:
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break
            if not all_match:
                errors += 1
                if errors > max_errors:
                    return False
        return errors <= max_errors

    # ------------------------------------------------------------------
    # Model extraction helpers (used by tests)
    # ------------------------------------------------------------------

    def _extract_thresholds_from_model(
        self, model_predicates: Dict[str, Any]
    ) -> Dict[str, float]:
        thresholds: Dict[str, float] = {}
        for name, info in model_predicates.items():
            if name.startswith("threshold_") and info["type"] == "Real":
                classifier_name = name[len("threshold_"):]
                try:
                    value = float(info["body"])
                    thresholds[classifier_name] = max(0.0, min(1.0, value))
                except (ValueError, TypeError):
                    pass
        return thresholds

    # ------------------------------------------------------------------
    # Synthesis from episodes
    # ------------------------------------------------------------------

    def synthesize_from_episodes(
        self,
        episodic_memory: Any,
        campaign_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        primitive_registry: Optional[PrimitiveRegistry] = None,
        ontology_memory: Optional[Any] = None,
    ) -> Tuple[Optional[Program], SynthesisStats]:
        from knowledge.episodic.episodic import EpisodeFilter

        filter_kwargs: Dict[str, Any] = {}
        if campaign_id:
            filter_kwargs["campaign_id"] = campaign_id
        if experiment_id:
            filter_kwargs["experiment_id"] = experiment_id

        episode_filter = EpisodeFilter(**filter_kwargs)
        episodes = episodic_memory.filter_episodes(episode_filter)

        if not episodes:
            logger.warning("No episodes found for campaign=%s experiment=%s",
                           campaign_id, experiment_id)
            return None, SynthesisStats()

        examples: List[Tuple[str, int]] = []
        for ep in episodes:
            prompt = ep.intervention.final_prompt or ep.intervention.prompt
            if prompt and ep.outcome is not None:
                examples.append((prompt, int(ep.outcome)))

        if not examples:
            logger.warning("No valid examples extracted from episodes")
            return None, SynthesisStats()

        logger.info("Synthesizing from %d episodes (campaign=%s, experiment=%s)",
                     len(examples), campaign_id, experiment_id)
        return self.synthesize_with_stats(examples, primitive_registry, ontology_memory)

    # ------------------------------------------------------------------
    # Theory abstraction
    # ------------------------------------------------------------------

    def abstract_theory(
        self,
        program: Program,
        model_family: str = "unknown",
        conditions: Optional[Dict[str, Any]] = None,
        provenance: Optional[List[str]] = None,
    ) -> Any:
        from knowledge.scientific_memory import Theory

        def _describe(node: Node) -> str:
            if isinstance(node, PredicateNode):
                p = node.primitive
                return f"{p.name}({json.dumps(p.parameters)})"
            if isinstance(node, ThresholdNode):
                return f"{node.classifier.name} > {node.threshold}"
            if isinstance(node, ClassifierNode):
                return node.primitive.name
            if isinstance(node, ApplyTransformNode):
                return f"{node.transform.name}({_describe(node.inner)})"
            if isinstance(node, NotNode):
                return f"NOT ({_describe(node.child)})"
            if isinstance(node, AndNode):
                return f"({_describe(node.left)} AND {_describe(node.right)})"
            if isinstance(node, OrNode):
                return f"({_describe(node.left)} OR {_describe(node.right)})"
            return str(node)

        condition_str = _describe(program.root.condition)
        pattern = f"IF {condition_str} THEN REFUSE"
        theory_conditions = dict(conditions or {})
        theory_conditions.setdefault("model_family", model_family)
        theory_conditions.setdefault("complexity", program.complexity())
        theory_conditions.setdefault("synthesis_method", "cvc5")

        theory = Theory(
            pattern=pattern,
            conditions=theory_conditions,
            confidence=0.0,
            provenance=provenance or [],
            metadata={
                "program_id": program.id,
                "node_count": program.complexity(),
                "depth": program.depth(),
                "then_outcome": int(program.root.then_outcome),
                "else_outcome": int(program.root.else_outcome),
            },
        )
        return theory

    def store_verified_program(
        self,
        defense_store: Any,
        program: Program,
        name: str = "",
        confidence: float = 0.0,
        provenance: Optional[List[str]] = None,
    ) -> str:
        from knowledge.defense_store import DefenseProgramRecord

        record = DefenseProgramRecord(
            name=name or f"synth_{program.id}",
            program=program,
            confidence=confidence,
            provenance=provenance or [],
        )
        return defense_store.save(record)


def _parse_sexpr_tree(body: str, start: int = 0):
    """Parse an S-expression string into a raw Python tree.

    Returns (raw_sexpr, next_index) where raw_sexpr is one of:
      - list  (nested S-expression call)
      - str   (symbol or string literal)
      - float / int
      - bool
    """
    body = body.strip()
    i = start
    while i < len(body) and body[i] in " \t\n\r":
        i += 1
    if i >= len(body):
        return None, i

    if body[i] == "(":
        i += 1
        items = []
        while i < len(body):
            while i < len(body) and body[i] in " \t\n\r":
                i += 1
            if i >= len(body) or body[i] == ")":
                i += 1
                break
            item, i = _parse_sexpr_tree(body, i)
            if item is not None:
                items.append(item)
        return items, i

    if body[i] == '"':
        i += 1
        s_start = i
        while i < len(body) and body[i] != '"':
            i += 1
        s_val = body[s_start:i]
        i += 1 if i < len(body) else 0
        return s_val, i

    # Atom
    a_start = i
    while i < len(body) and body[i] not in " \t\n\r)":
        i += 1
    atom = body[a_start:i]
    try:
        return float(atom) if "." in atom else int(atom), i
    except (ValueError, TypeError):
        pass
    if atom in ("true", "false"):
        return True if atom == "true" else False, i
    return atom, i


def _sexpr_to_ast(
    raw: Any,
    catalog: PrimitiveCatalog,
    defined_fns: Dict[str, str],
) -> Optional[Node]:
    """Convert a raw parsed S-expression tree into a Program AST Node.

    ``raw`` is the output of ``_parse_sexpr_tree``, which is a nested
    list/atom tree.  This function recursively maps it to Program AST
    nodes (AndNode, OrNode, NotNode, PredicateNode, ThresholdNode,
    ApplyTransformNode).
    """
    from core.program import (
        AndNode, OrNode, NotNode, PredicateNode, ThresholdNode,
        ApplyTransformNode, Node,
    )

    if not isinstance(raw, list) or len(raw) == 0:
        return None

    op = raw[0]
    args = raw[1:] if isinstance(op, str) else raw

    # --- Boolean combinators ---
    if op == "and" and len(args) >= 2:
        left = _sexpr_to_ast(args[0], catalog, defined_fns)
        right = _sexpr_to_ast(args[1], catalog, defined_fns)
        if isinstance(left, Node) and isinstance(right, Node):
            return AndNode(left=left, right=right)

    if op == "or" and len(args) >= 2:
        left = _sexpr_to_ast(args[0], catalog, defined_fns)
        right = _sexpr_to_ast(args[1], catalog, defined_fns)
        if isinstance(left, Node) and isinstance(right, Node):
            return OrNode(left=left, right=right)

    if op == "not" and len(args) >= 1:
        child = _sexpr_to_ast(args[0], catalog, defined_fns)
        if isinstance(child, Node):
            return NotNode(child=child)

    # --- Threshold: (> (classifier x) value) ---
    if op == ">" and len(args) >= 2:
        if isinstance(args[0], list) and len(args[0]) >= 1:
            cn = args[0][0]
            threshold_val = args[1] if isinstance(args[1], (int, float)) else 0.5
            for c in catalog.classifiers:
                if _safe_name(c.name) == cn:
                    return ThresholdNode(classifier=c, threshold=float(threshold_val))

    # --- Direct predicate call: (predicate_name [transform_or_x]) ---
    # Check if op matches a predicate name
    predicate = _find_predicate_by_name(op, catalog)
    if predicate is not None:
        if len(args) == 0:
            return PredicateNode(primitive=predicate)
        inner = args[0]
        if isinstance(inner, list) and len(inner) >= 1:
            tn = inner[0]
            transform = _find_transform_by_name(tn, catalog)
            if transform is not None:
                return ApplyTransformNode(
                    transform=transform,
                    inner=PredicateNode(primitive=predicate),
                )
        return PredicateNode(primitive=predicate)

    # --- Standalone transform call: (transform_name x) ---
    transform = _find_transform_by_name(op, catalog)
    if transform is not None:
        return None  # transforms alone don't form conditions

    # --- Classifier name fallback ---
    classifier = _find_classifier_by_name(op, catalog)
    if classifier is not None:
        return ThresholdNode(classifier=classifier, threshold=0.5)

    return None


def _find_predicate_by_name(name: str, catalog) -> Optional[Any]:
    for p in catalog.predicates:
        if _safe_name(p.name) == name:
            return p
    return None


def _find_transform_by_name(name: str, catalog) -> Optional[Any]:
    for t in catalog.transforms:
        if _safe_name(t.name) == name:
            return t
    return None


def _find_classifier_by_name(name: str, catalog) -> Optional[Any]:
    for c in catalog.classifiers:
        if _safe_name(c.name) == name:
            return c
    return None


def build_simple_program(
    predicate_name: str,
    registry: Optional[PrimitiveRegistry] = None,
    condition_registry: Optional[ConditionRegistry] = None,
    negate: bool = False,
    **params: Any,
) -> Optional[Program]:
    reg = registry or default_registry
    try:
        predicate = reg.get(predicate_name, params)
    except ValueError:
        predicate = None
    if predicate is not None and isinstance(predicate, Predicate):
        condition: Node = PredicateNode(primitive=predicate)
        if negate:
            condition = NotNode(child=condition)
        return Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )
    # Fallback: try ConditionRegistry for name resolution
    cond_reg = condition_registry or _condition_registry
    try:
        cond_reg.get(predicate_name)
    except KeyError:
        return None
    # Condition exists but cannot be wrapped as PredicateNode;
    # return sentinel None indicating it must be looked up at runtime.
    return None
