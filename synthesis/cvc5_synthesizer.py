"""Program synthesis from examples — enumeration primary, CVC5 optional."""
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
    enumeration_found: bool = False
    allow_error_rate: float = 0.0
    max_errors: int = 0
    errors_actual: int = 0
    beam_width: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": round(self.duration_ms, 2),
            "depth_used": self.depth_used,
            "programs_tried": self.programs_tried,
            "programs_skipped_cache": self.programs_skipped_cache,
            "cvc5_used": self.cvc5_used,
            "enumeration_found": self.enumeration_found,
            "allow_error_rate": self.allow_error_rate,
            "max_errors": self.max_errors,
            "errors_actual": self.errors_actual,
            "beam_width": self.beam_width,
        }


def _compute_hash(program: Program, examples: List[Tuple[str, int]]) -> str:
    return hash((program.canonical_form(), tuple(examples))).__str__()


def _node_count(condition: Node) -> int:
    from core.program import (
        AndNode,
        ApplyTransformNode,
        BinaryNode,
        ClassifierNode,
        IfThenElseNode,
        NotNode,
        OrNode,
        PredicateNode,
        ThresholdNode,
    )
    if isinstance(condition, (PredicateNode, ThresholdNode, ClassifierNode)):
        return 1
    if isinstance(condition, TransformNode):
        return 1
    if isinstance(condition, ApplyTransformNode):
        return 1 + _node_count(condition.inner)
    if isinstance(condition, (AndNode, OrNode)):
        return 1 + _node_count(condition.left) + _node_count(condition.right)
    if isinstance(condition, NotNode):
        return 1 + _node_count(condition.child)
    if isinstance(condition, IfThenElseNode):
        return 1 + _node_count(condition.condition)
    return 1


class CVC5Synthesizer:
    def __init__(
        self,
        cvc5_path: str = "cvc5",
        timeout: int = 30,
    max_depth: int = 3,
    allow_error_rate: float = 0.0,
    beam_width: int = 200,
        use_cache: bool = True,
        cache_path: Optional[str] = None,
    ) -> None:
        self.cvc5_path = cvc5_path
        self.timeout = timeout
        self.max_depth = max(1, int(max_depth))
        self.allow_error_rate = max(0.0, min(1.0, float(allow_error_rate)))
        self.beam_width = max(0, int(beam_width))
        self.use_cache = use_cache
        self.cache_path = cache_path
        self._cache: Dict[str, bool] = {}
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
                logger.debug("Saved %d cache entries to %s", len(self._cache), self.cache_path)
            except Exception as exc:
                logger.warning("Failed to save cache: %s", exc)

    def synthesize(
        self,
        examples: List[Tuple[str, int]],
        primitive_registry: Optional[PrimitiveRegistry] = None,
        ontology_memory: Optional[Any] = None,
    ) -> Optional[Program]:
        return self.synthesize_with_stats(
            examples, primitive_registry, ontology_memory
        )[0]

    def synthesize_with_stats(
        self,
        examples: List[Tuple[str, int]],
        primitive_registry: Optional[PrimitiveRegistry] = None,
        ontology_memory: Optional[Any] = None,
    ) -> Tuple[Optional[Program], SynthesisStats]:
        start = time.time()
        stats = SynthesisStats(
            allow_error_rate=self.allow_error_rate,
            max_errors=int(len(examples) * self.allow_error_rate),
            beam_width=self.beam_width,
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

        depth = 1
        max_auto_depth = self.max_depth + 2
        program: Optional[Program] = None

        while depth <= max_auto_depth and program is None:
            stats.depth_used = depth
            logger.info(
                "Trying enumeration at depth %d/%d (max_errors=%d, beam=%d)...",
                depth, max_auto_depth, max_errors, self.beam_width,
            )
            # Skip depths where combinatorial explosion is guaranteed
            if self.beam_width == 0 and depth > self.max_depth + 1:
                logger.warning("Beam=0 and depth=%d too expensive, skipping to CVC5", depth)
                break
            program = self._try_enumeration(
                examples, exporter, max_errors, stats, depth=depth
            )
            if program is not None:
                stats.enumeration_found = True
                logger.info("Enumeration found program at depth %d", depth)
                break
            depth += 1

        if program is None and self._cvc5_available():
            logger.info("Enumeration exhausted, trying CVC5...")
            stats.cvc5_used = True
            program = self._try_cvc5(examples, exporter, max_errors)
            if program is not None:
                logger.info("CVC5 found a solution")

        stats.errors_actual = 0
        if program is not None:
            executor = ProgramExecutor(
                exporter.primitive_registry or default_registry
            )
            errors = 0
            for prompt, expected in examples:
                try:
                    actual = executor.execute(program, prompt)
                    if actual != expected:
                        errors += 1
                except Exception:
                    errors += 1
            stats.errors_actual = errors

        stats.duration_ms = (time.time() - start) * 1000

        if program is None:
            logger.warning(
                "Synthesis failed after %d depths, "
                "%d programs tried in %.0fms",
                stats.depth_used,
                stats.programs_tried,
                stats.duration_ms,
            )

        return program, stats

    def _fitness_score(
        self,
        program: Program,
        examples: List[Tuple[str, int]],
        executor: ProgramExecutor,
    ) -> float:
        correct = 0
        for prompt, expected in examples:
            try:
                result = executor.execute(program, prompt)
                if result == expected:
                    correct += 1
            except Exception:
                pass
        return correct / len(examples) if examples else 0.0

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
        programs = exporter.enumerate_programs(
            max_depth=d,
            examples=examples,
        )
        if not programs:
            return None

        executor = ProgramExecutor(
            exporter.primitive_registry or default_registry
        )

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
                is_match = self._matches_all(
                    prog, examples, executor, max_errors=max_errors
                )
                self._cache[h] = is_match
            else:
                is_match = self._matches_all(
                    prog, examples, executor, max_errors=max_errors
                )

            if is_match:
                self._save_cache()
                return prog

        self._save_cache()
        return None

    def _matches_all(
        self,
        program: Program,
        examples: List[Tuple[str, int]],
        executor: ProgramExecutor,
        num_trials: int = 10,
        max_errors: int = 0,
    ) -> bool:
        errors = 0
        for prompt, expected in examples:
            all_match = True
            for _ in range(num_trials):
                try:
                    result = executor.execute(program, prompt)
                    if result != expected:
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

    def _cvc5_available(self) -> bool:
        try:
            result = subprocess.run(
                [self.cvc5_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _try_cvc5(
        self,
        examples: List[Tuple[str, int]],
        exporter: GrammarExporter,
        max_errors: int,
    ) -> Optional[Program]:
        temp_dir = tempfile.mkdtemp(prefix="cvc5_")
        smt_path = os.path.join(temp_dir, "synthesis.smt2")

        try:
            exporter.export_to_smtlib(
                examples, output_file=smt_path, use_free_thresholds=True
            )

            result = subprocess.run(
                [
                    self.cvc5_path,
                    "--timeout",
                    str(self.timeout * 1000),
                    smt_path,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )

            if result.returncode != 0:
                stderr_preview = (result.stderr or "")[:300]
                logger.debug("CVC5 returned non-zero: %s", stderr_preview)
                if "syntax error" in (result.stderr or "").lower():
                    logger.warning("CVC5 syntax error — trying fallback enumeration")
                    return self._fallback_from_smt(examples, exporter, max_errors)
                return None

            return self._parse_cvc5_output(
                result.stdout, exporter, examples, max_errors
            )

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

    def _fallback_from_smt(
        self,
        examples: List[Tuple[str, int]],
        exporter: GrammarExporter,
        max_errors: int,
    ) -> Optional[Program]:
        candidates = exporter.enumerate_programs(
            max_depth=exporter.max_depth, examples=examples
        )
        if candidates:
            executor = ProgramExecutor(
                exporter.primitive_registry or default_registry
            )
            for prog in candidates:
                if self._matches_all(
                    prog, examples, executor, max_errors=max_errors
                ):
                    return prog
        return None

    def _parse_cvc5_output(
        self,
        output: str,
        exporter: GrammarExporter,
        examples: List[Tuple[str, int]],
        max_errors: int,
    ) -> Optional[Program]:
        lines = output.strip().split("\n")
        if not lines:
            return None

        first = lines[0].strip()
        if first == "unsat":
            logger.info("CVC5: unsat — no program matches all examples")
            return None
        if first != "sat":
            logger.debug("CVC5 unexpected output: %s", first[:100])
            return None

        model_text = "\n".join(lines[1:])
        model_predicates = self._parse_smt_model(model_text)

        if model_predicates:
            thresholds = self._extract_thresholds_from_model(model_predicates)
            matched = self._build_from_model(
                model_predicates, exporter, thresholds
            )
            if matched is not None:
                executor = ProgramExecutor(
                    exporter.primitive_registry or default_registry
                )
                if self._matches_all(
                    matched, examples, executor, max_errors=max_errors
                ):
                    return matched

        return self._fallback_from_smt(examples, exporter, max_errors)

    def _parse_smt_model(
        self, model_text: str
    ) -> Dict[str, Any]:
        predicates: Dict[str, Any] = {}
        pattern = r"\(define-fun\s+(\w+)\s*\(\(x\s+String\)\)\s+(Bool|Real|String)\s+"
        for match in re.finditer(pattern, model_text):
            name = match.group(1)
            rtype = match.group(2)
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
            predicates[name] = {"type": rtype, "body": body.strip()}
        return predicates

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

    def _build_from_model(
        self,
        model_predicates: Dict[str, Any],
        exporter: GrammarExporter,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Optional[Program]:
        catalog = exporter.get_primitives()
        if not catalog.predicates:
            return None

        for p in catalog.predicates:
            pname = _safe_name(p.name)
            if pname in model_predicates:
                condition: Node = PredicateNode(primitive=p)
                return Program(
                    root=IfThenElseNode(
                        condition=condition, then_outcome=1, else_outcome=0
                    )
                )

        for c in catalog.classifiers:
            cname = _safe_name(c.name)
            if cname in model_predicates and thresholds and cname in thresholds:
                condition = ThresholdNode(
                    classifier=c, threshold=thresholds[cname]
                )
                return Program(
                    root=IfThenElseNode(
                        condition=condition, then_outcome=1, else_outcome=0
                    )
                )

        return None

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

        logger.info(
            "Synthesizing from %d episodes (campaign=%s, experiment=%s)",
            len(examples), campaign_id, experiment_id,
        )

        return self.synthesize_with_stats(
            examples, primitive_registry, ontology_memory
        )

    def abstract_theory(
        self,
        program: Program,
        model_family: str = "unknown",
        conditions: Optional[Dict[str, Any]] = None,
        provenance: Optional[List[str]] = None,
    ) -> Any:
        from knowledge.scientific_memory import Theory

        def _describe(node: Node) -> str:
            from core.program import (
                AndNode,
                ApplyTransformNode,
                ClassifierNode,
                NotNode,
                OrNode,
                PredicateNode,
                ThresholdNode,
            )
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


def _safe_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe if safe else "prim"


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
