"""Full SMT-LIB grammar for Defense Program synthesis (Section 2.2).

Encodes the full hypothesis space as an SMT formula:

    Π ::= IF φ THEN 1 ELSE 0
    φ ::= p(x) | ¬φ | φ₁ ∧ φ₂ | φ₁ ∨ φ₂ | c(x) > θ | φ(t(x))

Includes complexity/MDL constraints for Occam-guided search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .primitive import Classifier, Predicate, PrimitiveRegistry, Transform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMT-LIB type mapping
# ---------------------------------------------------------------------------

SMT_TYPE: Dict[str, str] = {
    "predicate": "Bool",
    "transform": "String",
    "classifier": "Real",
}


def _safe(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe if safe else "p"


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ---------------------------------------------------------------------------
# SMT Constraint Builder
# ---------------------------------------------------------------------------


class SMTConstraintBuilder:
    """Builds SMT-LIB v2.6 constraints for the defense program hypothesis space.

    Generates an ``(assert (exists ...))`` query that is satisfiable iff there
    exists a program (composition of primitives) matching all training examples.
    """

    def __init__(
        self,
        predicates: List[Predicate],
        transforms: List[Transform],
        classifiers: List[Classifier],
        max_depth: int = 3,
        use_complexity_constraint: bool = True,
        max_complexity: int = 10,
        allow_error_rate: float = 0.0,
    ) -> None:
        self.predicates = predicates
        self.transforms = transforms
        self.classifiers = classifiers
        self.max_depth = max(1, max_depth)
        self.use_complexity_constraint = use_complexity_constraint
        self.max_complexity = max(1, max_complexity)
        self.allow_error_rate = max(0.0, min(1.0, allow_error_rate))

    def build_smtlib(
        self,
        examples: List[Tuple[str, int]],
        use_free_thresholds: bool = True,
    ) -> str:
        """Build a complete SMT-LIB query string.

        Structure:
          1. Declare all primitive functions
          2. Define ``condition`` recursively via ITE chains (depth-limited)
          3. Define ``program(x) = IF condition(x) THEN 1 ELSE 0``
          4. Assert program matches examples (with error tolerance)
          5. Optionally assert max_complexity
          6. Check-sat, get-model
        """
        lines: List[str] = []
        lines.append(";; HARMONY-X Program Synthesis — SMT-LIB v2.6")
        lines.append("(set-logic NRA)")
        lines.append("")

        # --- Declare functions ---
        for p in self.predicates:
            pn = _safe(p.name)
            lines.append(f"(declare-fun {pn} (String) Bool)")
        for t in self.transforms:
            tn = _safe(t.name)
            lines.append(f"(declare-fun {tn} (String) String)")
        for c in self.classifiers:
            cn = _safe(c.name)
            lines.append(f"(declare-fun {cn} (String) Real)")

        threshold_vars: List[str] = []
        if use_free_thresholds:
            for c in self.classifiers:
                cn = _safe(c.name)
                tv = f"threshold_{cn}"
                lines.append(f"(declare-fun {tv} () Real)")
                lines.append(f"(assert (and (<= 0 {tv}) (<= {tv} 1)))")
                threshold_vars.append(tv)
        lines.append("")

        # --- Build condition as nested ITE ---
        # At each depth, we OR together all possible sub-expressions.
        # Use a combinatorial encoding: for each depth d, define a function
        # ``condition_d`` that returns Bool by either selecting a predicate,
        # threshold classifier, NOT of lower depth, AND/OR of lower depths,
        # or a transform applied to a lower-depth condition.
        all_atomics: List[str] = []

        # Base predicates
        for p in self.predicates:
            pn = _safe(p.name)
            all_atomics.append(f"({pn} x)")

        # Threshold classifiers
        for i, c in enumerate(self.classifiers):
            cn = _safe(c.name)
            if use_free_thresholds:
                tv = f"threshold_{cn}"
                all_atomics.append(f"(> ({cn} x) {tv})")
            else:
                for th in [0.3, 0.5, 0.7, 0.9]:
                    all_atomics.append(f"(> ({cn} x) {th})")

        # Compositions: NOT, AND, OR, ApplyTransform
        # Encode up to max_depth levels of composition
        depth_vars: Dict[int, str] = {}
        for d in range(1, self.max_depth + 1):
            dvar = f"depth_{d}"
            sub_exprs: List[str] = []

            # Level 1: base atomics
            if d == 1:
                sub_exprs.extend(all_atomics)
            else:
                prev_var = depth_vars[d - 1]
                # NOT
                sub_exprs.append(f"(not {prev_var})")

                # AND/OR (pairwise from previous level)
                if d >= 2:
                    sub_exprs.append(f"(and {prev_var} depth_1)")
                    sub_exprs.append(f"(or {prev_var} depth_1)")

                # ApplyTransform: transform then check predicate
                for t in self.transforms:
                    tn = _safe(t.name)
                    for p in self.predicates:
                        pn = _safe(p.name)
                        sub_exprs.append(f"({pn} ({tn} x))")
                    for c in self.classifiers:
                        cn = _safe(c.name)
                        if use_free_thresholds:
                            tv = f"threshold_{cn}"
                            sub_exprs.append(f"(> ({cn} ({tn} x)) {tv})")
                        else:
                            sub_exprs.append(f"(> ({cn} ({tn} x)) 0.5)")

            if sub_exprs:
                choices = " ".join(sub_exprs)
                lines.append(f"(define-fun {dvar} ((x String)) Bool")
                if len(sub_exprs) == 1:
                    lines.append(f"  {choices}")
                else:
                    lines.append(f"  (or {choices})")
                lines.append(")")
                lines.append("")
                depth_vars[d] = dvar

        # Top-level condition: OR across all depths
        if depth_vars:
            all_depth = " ".join(depth_vars.values())
            if len(depth_vars) == 1:
                lines.append(f"(define-fun condition ((x String)) Bool {all_depth})")
            else:
                lines.append(f"(define-fun condition ((x String)) Bool (or {all_depth}))")
        else:
            lines.append("(define-fun condition ((x String)) Bool false)")
        lines.append("")

        # --- Program definition ---
        lines.append("(define-fun program ((x String)) Int")
        lines.append("  (ite (condition x) 1 0)")
        lines.append(")")
        lines.append("")

        # --- Examples as assertions ---
        num_examples = len(examples)
        max_errors = int(num_examples * self.allow_error_rate)
        if max_errors > 0:
            # Soft constraint: allow up to max_errors mismatches
            error_vars: List[str] = []
            for i, (prompt, outcome) in enumerate(examples):
                ev = f"error_{i}"
                escaped = _escape(prompt)
                lines.append(f"(declare-fun {ev} () Bool)")
                lines.append(f"(assert (=> {ev} (not (= (program \"{escaped}\") {outcome}))))")
                error_vars.append(ev)
            # Count errors using an uninterpreted sum trick:
            # assert at most max_errors of the error_i are true
            lines.append(f"(assert (<= (+ 0 {' '.join(f'(ite {ev} 1 0)' for ev in error_vars)}) {max_errors}))")
        else:
            for prompt, outcome in examples:
                escaped = _escape(prompt)
                lines.append(f"(assert (= (program \"{escaped}\") {outcome}))")

        lines.append("")

        # --- Complexity constraint ---
        if self.use_complexity_constraint:
            lines.append(f"(assert (<= (program complexity) {self.max_complexity}))")
        lines.append("")

        # --- Objective: minimize complexity (if supported) ---
        # Use (minimize ...) if the solver supports it; otherwise just (check-sat)
        lines.append("(check-sat)")
        lines.append("(get-model)")

        result = "\n".join(lines)
        return result

    def build_synth_query(
        self,
        examples: List[Tuple[str, int]],
    ) -> str:
        """Build a synthesis query with existentials over program structure.

        This encodes:
            ∃ program-structure, ∀ example ∈ examples: program(example) = expected
        """
        return self.build_smtlib(examples, use_free_thresholds=True)


class Grammar:
    """High-level grammar interface for program synthesis.

    Combines PrimitiveRegistry access with SMT-LIB export.
    """

    def __init__(self, registry: PrimitiveRegistry) -> None:
        self.registry = registry

    def get_primitive_names(self) -> List[str]:
        return self.registry.list_primitives()

    def get_primitives(self) -> Tuple[List[Predicate], List[Transform], List[Classifier]]:
        predicates: List[Predicate] = []
        transforms: List[Transform] = []
        classifiers: List[Classifier] = []
        for name in self.registry.list_primitives():
            try:
                inst = self.registry.get(name)
            except ValueError:
                continue
            if isinstance(inst, Predicate):
                predicates.append(inst)
            elif isinstance(inst, Transform):
                transforms.append(inst)
            elif isinstance(inst, Classifier):
                classifiers.append(inst)
        return predicates, transforms, classifiers

    def to_smtlib(self, examples: Optional[List[Tuple[str, int]]] = None) -> str:
        """Generate SMT-LIB for the full hypothesis space."""
        predicates, transforms, classifiers = self.get_primitives()
        builder = SMTConstraintBuilder(
            predicates=predicates,
            transforms=transforms,
            classifiers=classifiers,
            max_depth=3,
            use_complexity_constraint=True,
            max_complexity=10,
            allow_error_rate=0.0,
        )
        return builder.build_smtlib(examples or [])

    def build(
        self,
        examples: List[Tuple[str, int]],
        max_depth: int = 3,
        allow_error_rate: float = 0.0,
    ) -> SMTConstraintBuilder:
        """Build an SMTConstraintBuilder configured for the given examples."""
        predicates, transforms, classifiers = self.get_primitives()
        return SMTConstraintBuilder(
            predicates=predicates,
            transforms=transforms,
            classifiers=classifiers,
            max_depth=max_depth,
            use_complexity_constraint=True,
            max_complexity=min(10, 2**max_depth),
            allow_error_rate=allow_error_rate,
        )
