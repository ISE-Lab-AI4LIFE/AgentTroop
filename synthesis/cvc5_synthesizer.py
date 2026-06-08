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
import os
import pickle
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.executor import ProgramExecutor
from core.grammar import SMTConstraintBuilder
from core.primitive import (
    Classifier,
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
        }


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


class CVC5Synthesizer:
    """Program synthesizer with CVC5 as primary solver and enumeration fallback.

    Parameters
    ----------
    cvc5_path : str
        Path to CVC5 binary.
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
        cvc5_path: str = "cvc5",
        timeout: int = 30,
        max_depth: int = 3,
        allow_error_rate: float = 0.0,
        beam_width: int = 200,
        use_cache: bool = True,
        cache_path: Optional[str] = None,
        hybrid: bool = True,
        enforce_cvc5_first: bool = True,
    ) -> None:
        self.cvc5_path = cvc5_path
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
            ontology_memory=ontology_memory,
            max_depth=self.max_depth,
        )

        if not examples:
            logger.warning("No examples provided for synthesis")
            stats.duration_ms = (time.time() - start) * 1000
            return None, stats

        max_errors = int(len(examples) * self.allow_error_rate)
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

        Strategy:
        1. Find the condition/condition_1 function — that's the root
        2. Walk the S-expression tree and map to Program AST nodes
        3. Build IfThenElseNode with the reconstructed condition
        """
        # Find the main condition function
        condition_body = None
        for name in ["condition", "condition_1", "depth_1"]:
            if name in defined_fns:
                condition_body = defined_fns[name]
                break

        if not condition_body:
            # Try any depth function
            for name in sorted(defined_fns.keys()):
                if name.startswith("depth_"):
                    condition_body = defined_fns[name]
                    break

        if not condition_body:
            # Try any predicate that's directly defined as Bool
            for name, body in defined_fns.items():
                if any(p.name == name or _safe_name(p.name) == name
                       for p in catalog.predicates):
                    condition_body = body
                    break

        if not condition_body:
            return None

        # Build program from the condition body
        # Try to find which predicate is used
        for p in catalog.predicates:
            pn = p.name
            pn_safe = _safe_name(pn)
            if pn_safe in condition_body or pn in condition_body:
                condition: Node = PredicateNode(primitive=p)
                return Program(
                    root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
                )

        # Try classifiers with thresholds
        for c in catalog.classifiers:
            cn = c.name
            cn_safe = _safe_name(cn)
            if cn_safe in condition_body:
                # Extract threshold from model
                tv = f"threshold_{cn_safe}"
                threshold = 0.5
                if tv in defined_fns:
                    try:
                        threshold = float(defined_fns[tv])
                    except (ValueError, TypeError):
                        pass
                condition = ThresholdNode(classifier=c, threshold=threshold)
                return Program(
                    root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
                )

        # Try to find predicate+transform combinations
        for t in catalog.transforms:
            tn = _safe_name(t.name)
            if tn in condition_body:
                for p in catalog.predicates:
                    pn = _safe_name(p.name)
                    if pn in condition_body:
                        inner_node: Node = ApplyTransformNode(
                            transform=t,
                            inner=PredicateNode(primitive=p)
                        )
                        return Program(
                            root=IfThenElseNode(condition=inner_node, then_outcome=1, else_outcome=0)
                        )

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
        programs.sort(key=lambda p: p.complexity())

        if self.beam_width > 0:
            scored: List[Tuple[float, Program]] = []
            for prog in programs[:self.beam_width * 2]:
                score = self._fitness_score(prog, examples, executor)
                scored.append((score, prog))
            scored.sort(key=lambda x: (-x[0], x[1].complexity()))
            programs = [p for _, p in scored[:self.beam_width]]

        for prog in programs:
            stats.programs_tried += 1
            if self.use_cache:
                h = _compute_hash(prog, examples)
                if h in self._cache:
                    stats.programs_skipped_cache += 1
                    if self._cache[h]:
                        return prog
                    continue
                is_match = self._matches_all(prog, examples, executor, max_errors=max_errors)
                self._cache[h] = is_match
            else:
                is_match = self._matches_all(prog, examples, executor, max_errors=max_errors)
            if is_match:
                self._save_cache()
                return prog

        self._save_cache()
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cvc5_available(self) -> bool:
        try:
            result = subprocess.run(
                [self.cvc5_path, "--version"],
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


def build_simple_program(
    predicate_name: str,
    registry: Optional[PrimitiveRegistry] = None,
    negate: bool = False,
    **params: Any,
) -> Optional[Program]:
    reg = registry or default_registry
    try:
        predicate = reg.get(predicate_name, params)
    except ValueError:
        return None
    if not isinstance(predicate, Predicate):
        return None
    condition: Node = PredicateNode(primitive=predicate)
    if negate:
        condition = NotNode(child=condition)
    return Program(
        root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
    )
