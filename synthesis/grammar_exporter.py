import itertools
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.primitive import (
    Classifier,
    ContainsWordPredicate,
    LengthGtPredicate,
    MatchesRegexPredicate,
    Predicate,
    Primitive,
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

logger = logging.getLogger(__name__)

THRESHOLD_CANDIDATES = [0.3, 0.5, 0.7, 0.9]
LENGTH_THRESHOLDS = [50, 100, 200]


@dataclass
class PrimitiveCatalog:
    predicates: List[Predicate] = field(default_factory=list)
    transforms: List[Transform] = field(default_factory=list)
    classifiers: List[Classifier] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.predicates or self.transforms or self.classifiers)

    def total_primitives(self) -> int:
        return len(self.predicates) + len(self.transforms) + len(self.classifiers)


class GrammarExporter:
    def __init__(
        self,
        primitive_registry: Optional[PrimitiveRegistry] = None,
        ontology_memory: Optional[Any] = None,
        max_depth: int = 3,
    ) -> None:
        self.primitive_registry = primitive_registry or default_registry
        self.ontology_memory = ontology_memory
        self.max_depth = max(1, int(max_depth))

    def get_primitives(self) -> PrimitiveCatalog:
        if self.ontology_memory is not None:
            return self._get_from_ontology()
        return self._get_from_registry()

    def _get_from_ontology(self) -> PrimitiveCatalog:
        catalog = PrimitiveCatalog()
        try:
            all_primitives = self.ontology_memory.list_primitives()
        except Exception as exc:
            logger.warning("Failed to read ontology memory: %s", exc)
            return self._get_from_registry()

        for p in all_primitives:
            ptype = getattr(p, "primitive_type", "")
            pname = getattr(p, "name", "")
            try:
                instance = self.primitive_registry.get(pname)
            except ValueError:
                continue
            if ptype == "predicate" and isinstance(instance, Predicate):
                catalog.predicates.append(instance)
            elif ptype == "transform" and isinstance(instance, Transform):
                catalog.transforms.append(instance)
            elif ptype == "classifier" and isinstance(instance, Classifier):
                catalog.classifiers.append(instance)
        return catalog

    def _get_from_registry(self) -> PrimitiveCatalog:
        catalog = PrimitiveCatalog()
        names = self.primitive_registry.list_primitives()
        for name in names:
            try:
                instance = self.primitive_registry.get(name)
            except ValueError:
                continue
            if isinstance(instance, Predicate):
                catalog.predicates.append(instance)
            elif isinstance(instance, Transform):
                catalog.transforms.append(instance)
            elif isinstance(instance, Classifier):
                catalog.classifiers.append(instance)
        return catalog

    def get_parameterized_primitives(
        self, examples: List[Tuple[str, int]]
    ) -> PrimitiveCatalog:
        base = self.get_primitives()
        result = PrimitiveCatalog()

        keywords = _extract_keywords(examples)

        for p in base.predicates:
            if isinstance(p, ContainsWordPredicate):
                for kw in keywords:
                    result.predicates.append(
                        ContainsWordPredicate(word=kw)
                    )
            elif isinstance(p, LengthGtPredicate):
                for t in LENGTH_THRESHOLDS:
                    result.predicates.append(
                        LengthGtPredicate(threshold=t)
                    )
            else:
                result.predicates.append(p)

        if not result.predicates:
            result.predicates = list(base.predicates)

        result.transforms = list(base.transforms)
        result.classifiers = list(base.classifiers)
        return result

    def enumerate_conditions(
        self,
        max_depth: Optional[int] = None,
        examples: Optional[List[Tuple[str, int]]] = None,
        max_programs: int = 0,
    ) -> List[Node]:
        depth = max_depth if max_depth is not None else self.max_depth
        if examples:
            catalog = self.get_parameterized_primitives(examples)
        else:
            catalog = self.get_primitives()
        if catalog.is_empty():
            return []
        return _enumerate_conditions(depth, catalog, max_programs=max_programs)

    def enumerate_programs(
        self,
        max_depth: Optional[int] = None,
        examples: Optional[List[Tuple[str, int]]] = None,
        max_programs: int = 0,
    ) -> List[Program]:
        conditions = self.enumerate_conditions(
            max_depth=max_depth, examples=examples, max_programs=max_programs
        )
        programs = [
            Program(
                root=IfThenElseNode(
                    condition=cond, then_outcome=1, else_outcome=0
                )
            )
            for cond in conditions
        ]
        programs.sort(key=lambda p: p.complexity())
        return programs

    def export_to_smtlib(
        self,
        examples: List[Tuple[str, int]],
        output_file: Optional[str] = None,
        max_depth: Optional[int] = None,
        use_free_thresholds: bool = False,
    ) -> str:
        catalog = self.get_parameterized_primitives(examples)
        depth = max_depth if max_depth is not None else self.max_depth
        lines: List[str] = []
        lines.append(";; HARMONY-X Program Synthesis — SMT-LIB v2.6")
        lines.append(
            f";; Primitives: {catalog.total_primitives()}, "
            f"Examples: {len(examples)}, Depth: {depth}"
        )
        lines.append("(set-logic NRA)")
        lines.append("")

        for p in catalog.predicates:
            pn = _safe_name(p.name)
            lines.append(f"(declare-fun {pn} (String) Bool)")

        threshold_vars: List[str] = []
        for c in catalog.classifiers:
            cn = _safe_name(c.name)
            lines.append(f"(declare-fun {cn} (String) Real)")
            if use_free_thresholds:
                tv = f"threshold_{cn}"
                lines.append(f"(declare-fun {tv} () Real)")
                lines.append(f"(assert (and (<= 0 {tv}) (<= {tv} 1)))")
                threshold_vars.append(tv)

        for t in catalog.transforms:
            tn = _safe_name(t.name)
            lines.append(f"(declare-fun {tn} (String) String)")

        lines.append("")
        Np = len(catalog.predicates)
        Nc = len(catalog.classifiers)

        if Np + Nc + len(catalog.transforms) == 0:
            lines.append(
                "(define-fun condition ((x String)) Bool false)"
            )
        else:
            all_atomics: List[str] = []
            for p in catalog.predicates:
                all_atomics.append(f"({_safe_name(p.name)} x)")
            for i, c in enumerate(catalog.classifiers):
                cn = _safe_name(c.name)
                if use_free_thresholds:
                    tv = f"threshold_{cn}"
                    all_atomics.append(f"(> ({cn} x) {tv})")
                else:
                    all_atomics.append(f"(> ({cn} x) 0.5)")

            has_transform_pred = catalog.predicates and len(catalog.predicates) > 0
            for t in catalog.transforms:
                tn = _safe_name(t.name)
                if has_transform_pred:
                    pn = _safe_name(catalog.predicates[0].name)
                    all_atomics.append(f"({pn} ({tn} x))")
                else:
                    all_atomics.append("false")

            combinator = "or"
            lines.append(
                "(define-fun condition ((x String)) Bool"
            )
            if len(all_atomics) == 1:
                lines.append(f"  {all_atomics[0]}")
            else:
                lines.append(f"  ({combinator}")
                for a in all_atomics:
                    lines.append(f"    {a}")
                lines.append("  )")
            lines.append(")")

        lines.append("")
        lines.append(
            "(define-fun program ((x String)) Int"
        )
        lines.append("  (ite (condition x) 1 0)")
        lines.append(")")
        lines.append("")

        for i, (prompt, outcome) in enumerate(examples):
            escaped = _escape_str(prompt)
            lines.append(
                f"(assert (= (program \"{escaped}\") {outcome}))"
            )

        lines.append("")
        lines.append("(check-sat)")
        lines.append("(get-model)")

        result = "\n".join(lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            logger.info("Exported SMT-LIB to %s", output_file)

        return result


def _extract_keywords(examples: List[Tuple[str, int]]) -> List[str]:
    keywords: Set[str] = set()
    for prompt, outcome in examples:
        if outcome == 1:
            words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
            keywords.update(words)
    return sorted(keywords) if keywords else ["test"]


def _enumerate_conditions(
    max_depth: int,
    catalog: PrimitiveCatalog,
    max_programs: int = 0,
) -> List[Node]:
    memo: Dict[int, List[Node]] = {}
    total_generated = 0
    limit = max_programs if max_programs > 0 else 2000

    for d in range(1, max_depth + 1):
        if total_generated >= limit:
            break
        results: List[Node] = []

        if d == 1:
            for p in catalog.predicates:
                results.append(PredicateNode(primitive=p))
            for c in catalog.classifiers:
                for t in THRESHOLD_CANDIDATES:
                    results.append(ThresholdNode(classifier=c, threshold=t))
        else:
            prev = _get_at_depth(d - 1, catalog, memo)

            for node in prev:
                results.append(NotNode(child=node))

            for t in catalog.transforms:
                for node in _nodes_at_transform_depth(d, catalog, memo):
                    results.append(
                        ApplyTransformNode(transform=t, inner=node)
                    )

            for d1 in range(1, d):
                d2 = d - d1
                lefts = _get_at_depth(d1, catalog, memo)
                rights = _get_at_depth(d2, catalog, memo)
                for l, r in itertools.product(lefts, rights):
                    results.append(AndNode(left=l, right=r))
                    results.append(OrNode(left=l, right=r))
                    if len(results) + total_generated >= limit:
                        break
                if len(results) + total_generated >= limit:
                    break

        seen: Set[str] = set()
        unique: List[Node] = []
        for node in results:
            key = str(node)
            if key not in seen:
                seen.add(key)
                unique.append(node)

        available = limit - total_generated
        if len(unique) > available > 0:
            unique = unique[:available]

        memo[d] = unique
        total_generated += len(unique)

    out: List[Node] = []
    for d in range(1, max_depth + 1):
        out.extend(memo.get(d, []))
    return out


def _nodes_at_transform_depth(
    depth: int, catalog: PrimitiveCatalog, memo: Dict[int, List[Node]]
) -> List[Node]:
    if depth <= 1:
        return _get_at_depth(depth - 1, catalog, memo) if depth > 1 else []
    candidates: List[Node] = []
    prev = _get_at_depth(depth - 1, catalog, memo)
    for node in prev:
        if isinstance(node, (PredicateNode, ThresholdNode)):
            candidates.append(node)
        if isinstance(node, ApplyTransformNode):
            candidates.append(node)
        if isinstance(node, NotNode):
            candidates.append(node)
        if isinstance(node, (AndNode, OrNode)):
            candidates.append(node)
    return candidates


def _get_at_depth(
    depth: int, catalog: PrimitiveCatalog, memo: Dict[int, List[Node]]
) -> List[Node]:
    if depth in memo:
        return memo[depth]
    if depth <= 0:
        return []
    _enumerate_conditions(depth, catalog)
    return memo.get(depth, [])


def _safe_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe if safe else "prim"


def _escape_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
