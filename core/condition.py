"""Unified ontology registry for conditions, predicates, and transforms.

Provides a single source of truth for all ``Primitive`` lookups used by
the grammar exporter, cvc5 synthesizer, compile path, hypothesis generation,
and any other component that needs to resolve a condition name to a callable
or type constraint.

All 29 predicate types are auto-registered from ``PrimitiveRegistry`` so
that adding a new predicate in ``core/primitive.py`` automatically makes it
available everywhere — no manual dispatch-table updates needed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


@dataclass(frozen=True)
class ConditionDef:
    """Describes a registered condition / predicate / transform.

    Attributes
    ----------
    name:
        Canonical name (e.g. ``"contains_word"``).
    fn:
        The Python callable that implements this condition.
    schema:
        The type schema dict (used by cvc5 to construct the right
        constant/parameter).
    param_names:
        Ordered list of parameter names extracted from the signature
        (excluding the first positional arg which is the input text).
    description:
        Human-readable description.
    tags:
        Arbitrary tags for filtering (e.g. ``{"predicate", "toxicity"}``).
    primitive_class:
        Optional reference to the Primitive subclass that implements
        this condition in the DSL.  When set, ``compile_to_node()``
        can build a ``PredicateNode`` for ProgramExecutor execution.
    dsl_keyword:
        The keyword used to match this condition in condition strings
        (e.g. ``"contains_word"``).  Defaults to ``name``.
    needs_threshold:
        If True, this condition requires a numeric threshold extraction.
    needs_regex:
        If True, this condition requires a regex pattern extraction.
    needs_string_param:
        If True, this condition requires a string parameter extraction
        from single-quoted values in the condition string.
    param_extractor:
        Optional callable that extracts parameters from a condition string.
        Takes ``(cond_str: str) -> Optional[Dict[str, Any]]``.
    """

    name: str
    fn: Callable
    schema: Dict[str, Any]
    param_names: List[str] = field(default_factory=list)
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    primitive_class: Optional[type] = None
    dsl_keyword: str = ""
    needs_threshold: bool = False
    needs_regex: bool = False
    needs_string_param: bool = False
    param_extractor: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None

    def __post_init__(self) -> None:
        if not self.dsl_keyword:
            object.__setattr__(self, "dsl_keyword", self.name)

    def compile_to_node(self, **params: Any) -> Any:
        """Compile this condition into a Program AST Node.

        Requires ``primitive_class`` to be set.  Returns a
        ``PredicateNode(primitive=primitive_instance)`` that can
        be wrapped in an ``IfThenElseNode`` and executed by
        ``ProgramExecutor``.

        Raises ``TypeError`` if ``primitive_class`` is None.
        """
        if self.primitive_class is None:
            raise TypeError(
                f"Cannot compile ConditionDef '{self.name}': "
                f"no primitive_class registered"
            )
        from core.program import PredicateNode
        prim = self.primitive_class(**params)
        return PredicateNode(primitive=prim)

    def extract_params(self, cond_str: str) -> Optional[Dict[str, Any]]:
        """Extract parameters from a condition string using the registered
        ``param_extractor`` (if any), or by heuristics based on
        ``needs_string_param`` / ``needs_threshold`` / ``needs_regex``.
        """
        if self.param_extractor is not None:
            return self.param_extractor(cond_str)
        params: Dict[str, Any] = {}
        if self.needs_string_param:
            # Match both single and double quotes
            m = re.findall(r"""['"]([^'"]*)['"]""", cond_str)
            if m:
                pname = self.param_names[0] if self.param_names else "word"
                params[pname] = m[0]
        if self.needs_threshold:
            m = re.search(r">\s*([\d.]+)", cond_str)
            if m:
                pname = self.param_names[0] if self.param_names else "threshold"
                params[pname] = float(m.group(1))
        if self.needs_regex:
            m = re.search(r"matches_regex\s*\(\s*(?:r)?['\"]([^'\"]+)['\"]\s*\)", cond_str, re.IGNORECASE)
            if m:
                params["pattern"] = m.group(1)
        # Length-based: char_count(prompt) > N / < N
        if "char_count" in cond_str:
            for op in (">", "<"):
                lm = re.search(rf"char_count\s*\(\s*prompt\s*\)\s*{re.escape(op)}\s*(\d+)", cond_str)
                if lm:
                    params["threshold"] = int(lm.group(1))
        return params if params else None


class ConditionRegistry:
    """Global registry for all conditions used across the system.

    Usage::

        registry = ConditionRegistry()
        registry.populate_from_primitive_registry(default_registry)
        cond = registry.get("contains_word")
        node = cond.compile_to_node(word="bomb")
    """

    def __init__(self) -> None:
        self._conditions: Dict[str, ConditionDef] = {}
        self._frozen: bool = False
        self._keyword_index: Dict[str, str] = {}  # dsl_keyword → name

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        fn: Callable,
        *,
        name: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[Set[str]] = None,
        dsl_keyword: Optional[str] = None,
        needs_threshold: bool = False,
        needs_regex: bool = False,
        needs_string_param: bool = False,
        param_extractor: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ) -> ConditionDef:
        """Register a callable as a condition.

        If *schema* is not provided, one is inferred from the function's
        type annotations (if available) or from its parameter list.

        Returns the created ``ConditionDef``.
        """
        if self._frozen:
            raise RuntimeError("Cannot register after freeze")
        canonical = name or fn.__name__
        if canonical in self._conditions:
            raise KeyError(f"Condition {canonical!r} already registered")
        if not callable(fn):
            raise TypeError(f"Expected callable, got {type(fn)}")

        schema = schema or _infer_schema(fn)
        param_names = _infer_param_names(fn)
        tags_set = tags or set()

        cond = ConditionDef(
            name=canonical,
            fn=fn,
            schema=schema,
            param_names=param_names,
            description=description,
            tags=tags_set,
            dsl_keyword=dsl_keyword or canonical,
            needs_threshold=needs_threshold,
            needs_regex=needs_regex,
            needs_string_param=needs_string_param,
            param_extractor=param_extractor,
        )
        self._conditions[canonical] = cond
        self._keyword_index[cond.dsl_keyword] = canonical
        return cond

    def register_many(
        self,
        fns: Dict[str, Callable],
        *,
        base_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[ConditionDef]:
        """Register multiple functions at once."""
        results: List[ConditionDef] = []
        for name, fn in fns.items():
            results.append(self.register(fn, name=name, schema=base_schema, tags=tags))
        return results

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ConditionDef:
        """Look up a condition by canonical name."""
        _ensure_populated()
        if name not in self._conditions:
            available = ", ".join(sorted(self._conditions))
            raise KeyError(f"Unknown condition {name!r}. Available: {available}")
        return self._conditions[name]

    def get_fn(self, name: str) -> Callable:
        """Look up the callable for a condition by name."""
        return self.get(name).fn

    def find(self, *, tags: Optional[Set[str]] = None, name_pattern: Optional[str] = None) -> List[ConditionDef]:
        """Search conditions by tags and/or name regex pattern."""
        results = list(self._conditions.values())
        if tags:
            results = [c for c in results if tags.issubset(c.tags)]
        if name_pattern:
            pat = re.compile(name_pattern)
            results = [c for c in results if pat.search(c.name)]
        return results

    def find_by_keyword(self, keyword: str) -> Optional[ConditionDef]:
        """Resolve a DSL keyword to a ConditionDef."""
        _ensure_populated()
        name = self._keyword_index.get(keyword)
        if name:
            return self._conditions.get(name)
        for c in self._conditions.values():
            if c.dsl_keyword == keyword:
                return c
        return None

    def __getitem__(self, name: str) -> ConditionDef:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._conditions

    def __len__(self) -> int:
        _ensure_populated()
        return len(self._conditions)

    def __bool__(self) -> bool:
        return len(self._conditions) > 0

    def __iter__(self):
        _ensure_populated()
        return iter(self._conditions.values())

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        _ensure_populated()
        return sorted(self._conditions)

    @property
    def all(self) -> Dict[str, ConditionDef]:
        _ensure_populated()
        return dict(self._conditions)

    def as_grammar_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return a grammar-export-friendly dict of name * schema."""
        _ensure_populated()
        return {c.name: c.schema for c in self._conditions.values()}

    def as_synthesis_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return a synthesis-friendly dict of name * param info."""
        _ensure_populated()
        return {
            c.name: {
                "param_names": c.param_names,
                "tags": sorted(c.tags),
            }
            for c in self._conditions.values()
        }

    def as_hypothesis_spec(self) -> str:
        """Return a formatted string listing all available predicates with
        their descriptions, param names, and example condition syntax, for
        inclusion in the LLM hypothesis generation prompt.

        Includes transform-wrapped condition examples so the LLM can
        generate hypotheses like ``rot13(contains_word("bomb"))``.
        """
        _ensure_populated()
        lines: List[str] = []
        for name in sorted(self._conditions):
            c = self._conditions[name]
            if "predicate" not in c.tags:
                continue
            desc = c.description or ""
            params = c.param_names
            example = _example_condition(c)
            lines.append(
                f"  - {name}({', '.join(params)}): {desc}"
                f"{'  e.g. ' + example if example else ''}"
            )
        # Transform section — show how to wrap predicates
        transform_names: List[str] = []
        for name in sorted(self._conditions):
            c = self._conditions[name]
            if "transform" not in c.tags:
                continue
            transform_names.append(name)
        if transform_names:
            lines.append("")
            lines.append("  Transforms (wrap around any predicate):")
            for tn in transform_names[:6]:
                lines.append(f"    - {tn}(predicate_name(params)): applies {tn} to prompt before evaluating predicate")
            lines.append("    - ... (other transforms available)")
            lines.append("")
            lines.append("  Transform-wrapped examples (use these in conditions):")
            transform_examples = [
                f'    - {tn}({example})'
                for tn in transform_names[:4]
                for example in [_example_condition(c) for c in [self._conditions.get(n) for n in sorted(self._conditions) if "predicate" in self._conditions[n].tags][:2]]
            ]
            transform_examples = [
                f'    - rot13(contains_word("bomb"))',
                f'    - base64_decode(contains_word("kill"))',
                f'    - leet_speak(contains_any_word(["weapon", "hack"]))',
                f'    - rot13(starts_with_roleplay(prompt))',
                f'    - base64_decode(startsWith(prompt) = "evil")',
            ]
            lines.extend(transform_examples[:5])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Compile bridge — condition string -> Program
    # ------------------------------------------------------------------

    def compile_condition_str(
        self,
        cond_str: str,
        then_out: int = 0,
        else_out: int = 1,
    ) -> Optional[Any]:
        """Compile a single condition string to a ``Program`` using the
        registry's auto-discovered predicate dispatch.

        Supports transform-wrapped conditions: ``rot13(contains_word('bomb'))``
        will produce ``ApplyTransformNode(rot13, PredicateNode(contains_word))``.

        Returns ``None`` if the condition cannot be parsed.
        """
        _ensure_populated()
        from core.program import Program, IfThenElseNode, PredicateNode, ApplyTransformNode
        import re as _re

        cond_lower = cond_str.lower().strip()
        if cond_lower.startswith("if "):
            cond_lower = cond_lower[3:]

        # Strip trailing THEN outcome suffix
        then_idx = cond_lower.rfind(" then ")
        if then_idx != -1:
            cond_lower = cond_lower[:then_idx].strip()

        # Handle NOT prefix
        if cond_lower.startswith("not "):
            inner = cond_lower[4:]
            inner_prog = self.compile_condition_str(inner, then_out, else_out)
            if inner_prog is not None:
                from core.program import NotNode
                not_cond = NotNode(child=inner_prog.root.condition)
                prog = Program(root=IfThenElseNode(
                    condition=not_cond, then_outcome=then_out, else_outcome=else_out,
                ))
                prog.source = "compiled_from_condition"
                return prog

        # 0. Transform wrapping: transform_name(predicate_name(...))
        #    Build a mapping of transform names to their ConditionDef entries.
        transform_names: Dict[str, str] = {}
        for cname, cd in self._conditions.items():
            if "transform" in cd.tags and cd.primitive_class is not None:
                key = getattr(cd.primitive_class, "name", cname).lower()
                transform_names[key] = cname

        # Check for transform-wrapped conditions: e.g. "rot13(contains_word('bomb'))"
        # Match: word_chars ( word_chars ( ... ) )
        # Strategy: for each known transform, see if cond starts with "transform_name("
        # and has a matching ")" at the end.
        if transform_names:
            for tn_key, tn_cname in sorted(transform_names.items(), key=lambda x: -len(x[0])):
                if cond_lower.startswith(tn_key + "(") and cond_lower.endswith(")"):
                    inner = cond_lower[len(tn_key) + 1:-1].strip()
                    if inner:
                        cd = self._conditions.get(tn_cname)
                        inner_prog = self.compile_condition_str(inner, then_out, else_out)
                        if inner_prog is not None:
                            transform_inst = cd.primitive_class()
                            wrapped = ApplyTransformNode(
                                transform=transform_inst,
                                inner=inner_prog.root.condition,
                            )
                            prog = Program(root=IfThenElseNode(
                                condition=wrapped,
                                then_outcome=then_out, else_outcome=else_out,
                            ))
                            prog.source = "compiled_from_condition"
                            return prog

        # 1. Simple keyword match from registry
        canonical_keys = sorted(self._keyword_index.keys(), key=len, reverse=True)
        for kw in canonical_keys:
            if kw in cond_lower:
                cd = self._conditions.get(self._keyword_index[kw])
                if cd is None or cd.primitive_class is None:
                    continue
                params = cd.extract_params(cond_lower)
                try:
                    inst = cd.primitive_class(**(params or {}))
                except Exception:
                    continue
                prog = Program(root=IfThenElseNode(
                    condition=PredicateNode(primitive=inst),
                    then_outcome=then_out, else_outcome=else_out,
                ))
                prog.source = "compiled_from_condition"
                return prog

        # 2. Special patterns not matched by keyword
        #    matches_regex(r'...')
        if "matches_regex" in cond_lower:
            rx_m = _re.search(r"matches_regex\s*\(\s*(?:r)?['\"]([^'\"]+)['\"]\s*\)", cond_lower, _re.IGNORECASE)
            if rx_m:
                prog = Program(root=IfThenElseNode(
                    condition=PredicateNode(primitive=_get_primitive("matches_regex")(pattern=rx_m.group(1))),
                    then_outcome=then_out, else_outcome=else_out,
                ))
                prog.source = "compiled_from_condition"
                return prog

        # 3. sentiment(prompt) > T
        if "sentiment" in cond_lower:
            th_m = _re.search(r">\s*([\d.]+)", cond_lower)
            if th_m:
                prog = Program(root=IfThenElseNode(
                    condition=PredicateNode(primitive=_get_primitive("sentiment")(threshold=float(th_m.group(1)))),
                    then_outcome=then_out, else_outcome=else_out,
                ))
                prog.source = "compiled_from_condition"
                return prog

        # 4. intent(prompt) = 'X'
        if "intent" in cond_lower:
            it_m = _re.search(r"=\s*'([^']+)'", cond_lower)
            if it_m:
                prog = Program(root=IfThenElseNode(
                    condition=PredicateNode(primitive=_get_primitive("intent")(intent_type=it_m.group(1))),
                    then_outcome=then_out, else_outcome=else_out,
                ))
                prog.source = "compiled_from_condition"
                return prog

        # 5. char_count(prompt) > N / < N
        for op_name, op_sym in [("length_gt", ">"), ("length_lt", "<")]:
            m = _re.search(rf"char_count\s*\(\s*prompt\s*\)\s*{_re.escape(op_sym)}\s*(\d+)", cond_lower)
            if m:
                prog = Program(root=IfThenElseNode(
                    condition=PredicateNode(primitive=_get_primitive(op_name)(threshold=int(m.group(1)))),
                    then_outcome=then_out, else_outcome=else_out,
                ))
                prog.source = "compiled_from_condition"
                return prog

        # 6. Legacy keyword fallback (single-quoted words)
        kw_match = _re.findall(r"'([^']*)'", cond_lower)
        if kw_match:
            for kw in kw_match:
                if len(kw) > 1 and not kw.isdigit():
                    from core.primitive import ContainsWordPredicate
                    prog = Program(root=IfThenElseNode(
                        condition=PredicateNode(primitive=ContainsWordPredicate(word=kw)),
                        then_outcome=then_out, else_outcome=else_out,
                    ))
                    prog.source = "compiled_from_condition"
                    return prog

        return None

    # ------------------------------------------------------------------
    # Synthesis / grammar helpers
    # ------------------------------------------------------------------

    def get_parameterized_primitives(
        self,
        examples: Optional[List[Tuple[str, int]]] = None,
        max_params_per_type: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return a diverse list of parameterized primitive instances for
        grammar export and synthesis.

        Ensures no single predicate family dominates (>50% of returned items).
        """
        from core.primitive import ContainsWordPredicate, LengthGtPredicate
        result: List[Dict[str, Any]] = []

        def _add(pred_cls, params=None):
            result.append({"class": pred_cls, "params": params or {}})

        # 1. ContainsWord with keywords from examples (if available)
        keywords: List[str] = []
        if examples:
            for prompt, outcome in examples:
                if outcome == 1:
                    for w in prompt.lower().split():
                        if len(w) > 3 and w.isalpha() and w not in keywords:
                            keywords.append(w)
                            if len(keywords) >= max_params_per_type:
                                break
                if len(keywords) >= max_params_per_type:
                    break
        for kw in keywords[:max_params_per_type]:
            _add(ContainsWordPredicate, {"word": kw})

        # 2. Length predicates with diverse thresholds
        for th in [30, 50, 100, 200, 500]:
            if len([x for x in result if x["class"].__name__.startswith("Length")]) >= max_params_per_type:
                break
            _add(LengthGtPredicate, {"threshold": th})

        # 3. Add one instance of every other predicate type from registry
        seen_families: Set[str] = set()
        for name in sorted(self._conditions):
            cd = self._conditions[name]
            if "predicate" not in cd.tags:
                continue
            if cd.primitive_class is None:
                continue
            family = cd.primitive_class.__name__.replace("Predicate", "")
            if family in seen_families:
                continue
            seen_families.add(family)
            # Skip families already covered
            if any(r["class"].__name__ == cd.primitive_class.__name__ for r in result):
                continue
            _add(cd.primitive_class)

        # Ensure no single family >50%
        from collections import Counter
        family_counts: Counter = Counter()
        for r in result:
            fam = r["class"].__name__.replace("Predicate", "")
            family_counts[fam] += 1
        total = len(result)
        for fam, cnt in family_counts.most_common():
            if total > 0 and cnt / total > 0.5:
                excess = int(cnt - total * 0.5)
                for r in list(result):
                    if r["class"].__name__.replace("Predicate", "") == fam and excess > 0:
                        result.remove(r)
                        excess -= 1
                        total -= 1

        return result

    # ------------------------------------------------------------------
    # Fix 5: Hypothesis compilation validation
    # ------------------------------------------------------------------

    def validate_condition_str(
        self,
        cond_str: str,
        test_prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate that a condition string compiles without semantic drift.

        Checks:
        1. Compilation success.
        2. Keyword resolution: which predicate keyword matched.
        3. Parameter extraction: what params were extracted.
        4. (Optional) Prediction consistency on test prompts.

        Returns a dict with keys:
          - valid (bool): whether the condition compiled.
          - program (Program or None): compiled program if successful.
          - matched_keyword (str or None): which keyword resolved.
          - params (dict): extracted parameters.
          - issues (list[str]): warnings about potential semantic drift.
          - predictions (dict): prompt -> prediction on test_prompts if provided.
        """
        _ensure_populated()
        from core.program import Program, IfThenElseNode, PredicateNode, ApplyTransformNode
        from core.executor import ProgramExecutor
        from core.primitive import default_registry
        import re as _re

        result: Dict[str, Any] = {
            "valid": False,
            "program": None,
            "matched_keyword": None,
            "params": {},
            "issues": [],
            "predictions": {},
        }

        cond_lower = cond_str.lower().strip()
        if cond_lower.startswith("if "):
            cond_lower = cond_lower[3:]
        then_idx = cond_lower.rfind(" then ")
        if then_idx != -1:
            cond_lower = cond_lower[:then_idx].strip()

        # Track what matched
        transform_names: Dict[str, str] = {}
        for cname, cd in self._conditions.items():
            if "transform" in cd.tags and cd.primitive_class is not None:
                key = getattr(cd.primitive_class, "name", cname).lower()
                transform_names[key] = cname

        # Check transform wrapping
        if transform_names:
            for tn_key, tn_cname in sorted(transform_names.items(), key=lambda x: -len(x[0])):
                if cond_lower.startswith(tn_key + "(") and cond_lower.endswith(")"):
                    result["matched_keyword"] = tn_cname
                    break

        if result["matched_keyword"] is None:
            canonical_keys = sorted(self._keyword_index.keys(), key=len, reverse=True)
            for kw in canonical_keys:
                if kw in cond_lower:
                    result["matched_keyword"] = self._keyword_index[kw]
                    cd = self._conditions.get(self._keyword_index[kw])
                    if cd is not None:
                        result["params"] = cd.extract_params(cond_lower) or {}
                    break

        if result["matched_keyword"] is None and _re.findall(r"'([^']*)'", cond_lower):
            result["matched_keyword"] = "contains_word (legacy fallback)"
            result["issues"].append(
                "No registered predicate keyword matched; using legacy "
                "single-quote fallback.  This may not match the intended semantics."
            )

        # Compile
        try:
            prog = self.compile_condition_str(cond_str)
            result["valid"] = prog is not None
            result["program"] = prog

            if prog is not None and result["matched_keyword"] and "legacy" in result["matched_keyword"]:
                result["issues"].append(
                    f"Compiled via legacy fallback; matched keyword='{result['matched_keyword']}' "
                    f"with params={result['params']}.  Consider using explicit "
                    f"contains_word() syntax."
                )

            if prog is not None and test_prompts:
                executor = ProgramExecutor(default_registry)
                for p in test_prompts:
                    try:
                        pred = int(executor.execute(prog, p))
                        result["predictions"][p] = pred
                    except Exception as e:
                        result["predictions"][p] = str(e)
        except Exception as e:
            result["issues"].append(f"Compilation raised exception: {e}")

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Prevent further registration (idempotent)."""
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def clear(self) -> None:
        if self._frozen:
            raise RuntimeError("Cannot clear after freeze")
        self._conditions.clear()
        self._keyword_index.clear()

    def populate_from_primitive_registry(
        self,
        prim_reg: Any,
        *,
        overwrite: bool = False,
    ) -> int:
        """Auto-register all ``Predicate`` subclasses from a ``PrimitiveRegistry``.

        For each predicate in *prim_reg*, creates a ``ConditionDef`` with
        ``primitive_class`` set so that ``compile_to_node()`` works.

        Also registers Transform subclasses with ``"transform"`` tags.

        Returns the number of conditions registered.
        """
        from core.primitive import Predicate, Transform, Classifier, PrimitiveRegistry
        count = 0
        for name in prim_reg.list_primitives():
            if name in self._conditions and not overwrite:
                continue
            try:
                prim = prim_reg.get(name)
            except (ValueError, KeyError):
                continue
            pclass = type(prim)
            tags: Set[str] = set()

            if isinstance(prim, Predicate):
                tags.add("predicate")
                need_str = bool(getattr(pclass, "__dataclass_fields__", {}).get("word") or
                              getattr(pclass, "__dataclass_fields__", {}).get("prefix") or
                              getattr(pclass, "__dataclass_fields__", {}).get("suffix"))
                need_th = bool(getattr(pclass, "__dataclass_fields__", {}).get("threshold"))
                need_re = "matches_regex" in name or "pattern" in getattr(pclass, "__dataclass_fields__", {})
                self._conditions[name] = ConditionDef(
                    name=name,
                    fn=prim.evaluate,
                    schema=_infer_schema(pclass),
                    param_names=_infer_param_names(pclass),
                    description=getattr(prim, "description", ""),
                    tags=tags | {"predicate", pclass.__module__.split(".")[-1]},
                    primitive_class=pclass,
                    dsl_keyword=name,
                    needs_threshold=need_th,
                    needs_regex=need_re,
                    needs_string_param=need_str,
                )
                self._keyword_index[name] = name
                count += 1
            elif isinstance(prim, Transform):
                tags.add("transform")
                self._conditions[name] = ConditionDef(
                    name=name,
                    fn=prim.evaluate,
                    schema=_infer_schema(pclass),
                    param_names=_infer_param_names(pclass),
                    description=getattr(prim, "description", ""),
                    tags=tags | {"transform", pclass.__module__.split(".")[-1]},
                    primitive_class=pclass,
                    dsl_keyword=name,
                )
                self._keyword_index[name] = name
                count += 1
            elif isinstance(prim, Classifier):
                tags.add("classifier")
                self._conditions[name] = ConditionDef(
                    name=name,
                    fn=prim.evaluate,
                    schema=_infer_schema(pclass),
                    param_names=_infer_param_names(pclass),
                    description=getattr(prim, "description", ""),
                    tags=tags | {"classifier", pclass.__module__.split(".")[-1]},
                    primitive_class=pclass,
                    dsl_keyword=name,
                )
                self._keyword_index[name] = name
                count += 1
        return count

    def populate_from_registry(
        self,
        prim_reg: Any,
        *,
        overwrite: bool = False,
    ) -> int:
        """Alias for ``populate_from_primitive_registry``."""
        return self.populate_from_primitive_registry(prim_reg, overwrite=overwrite)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_primitive(name: str) -> Any:
    """Lazy-import a primitive class by name."""
    from core.primitive import (
        ContainsWordPredicate, ContainsAnyWordPredicate, ContainsAllWordsPredicate,
        LengthGtPredicate, LengthLtPredicate, HasNumberPredicate, HasSpecialCharPredicate,
        IsAllCapsPredicate, IsEmptyPredicate, ContainsLeetPredicate, MatchesRegexPredicate,
        StartsWithPredicate, EndsWithPredicate, StartsWithRoleplayPredicate,
        ContainsSystemOverridePredicate, ContainsDelimiterPredicate, ContainsCodeBlockPredicate,
        HasEmojiPredicate, ContainsURLPredicate, SentimentPredicate, IntentPredicate,
        MatchesJailbreakPatternPredicate, ContainsEncodingWrapperPredicate,
        IsRepetitivePredicate, IsGrammaticalQuestionPredicate, StartsWithImperativePredicate,
        IsAllCapsPredicate,
    )
    _MAP = {c.__name__.replace("Predicate", "").lower(): c for c in
            [ContainsWordPredicate, ContainsAnyWordPredicate, ContainsAllWordsPredicate,
             LengthGtPredicate, LengthLtPredicate, HasNumberPredicate, HasSpecialCharPredicate,
             IsAllCapsPredicate, IsEmptyPredicate, ContainsLeetPredicate, MatchesRegexPredicate,
             StartsWithPredicate, EndsWithPredicate, StartsWithRoleplayPredicate,
             ContainsSystemOverridePredicate, ContainsDelimiterPredicate, ContainsCodeBlockPredicate,
             HasEmojiPredicate, ContainsURLPredicate, SentimentPredicate, IntentPredicate,
             MatchesJailbreakPatternPredicate, ContainsEncodingWrapperPredicate,
             IsRepetitivePredicate, IsGrammaticalQuestionPredicate, StartsWithImperativePredicate]}
    _MAP["length_gt"] = LengthGtPredicate
    _MAP["length_lt"] = LengthLtPredicate
    _MAP["matches_regex"] = MatchesRegexPredicate
    _MAP["sentiment"] = SentimentPredicate
    _MAP["intent"] = IntentPredicate
    return _MAP.get(name)


def _example_condition(cd: ConditionDef) -> str:
    """Generate an example condition string for a ConditionDef."""
    if cd.name == "contains_word":
        return 'contains_word("bomb")'
    if cd.name == "contains_any_word":
        return 'contains_any_word(["kill", "weapon"])'
    if cd.name == "contains_all_words":
        return 'contains_all_words(["step1", "step2"])'
    if cd.name == "starts_with_roleplay":
        return 'starts_with_roleplay(prompt)'
    if cd.name in ("contains_system_override", "has_number", "has_special_char",
                    "is_all_caps", "is_empty", "contains_code_block",
                    "contains_delimiter", "has_emoji", "contains_url",
                    "is_repetitive", "is_grammatical_question", "starts_with_imperative",
                    "contains_leet"):
        return f'{cd.name}(prompt)'
    if cd.name == "matches_jailbreak_pattern":
        return 'matches_jailbreak_pattern(prompt)'
    if cd.name == "contains_encoding_wrapper":
        return 'contains_encoding_wrapper(prompt)'
    if cd.name == "starts_with":
        return 'starts_with("prefix")'
    if cd.name == "ends_with":
        return 'ends_with("suffix")'
    if cd.name == "matches_regex":
        return 'matches_regex(r"(?i)\\b(kill|bomb)\\b")'
    if cd.name == "sentiment":
        return 'sentiment(prompt) > 0.5'
    if cd.name == "intent":
        return 'intent(prompt) = "harmful"'
    if cd.name in ("length_gt", "length_lt"):
        return f'char_count(prompt) {"<" if "lt" in cd.name else ">"} 50'
    return cd.name


# ------------------------------------------------------------------
# Schema / param inference helpers
# ------------------------------------------------------------------

def _infer_schema(fn: Callable) -> Dict[str, Any]:
    """Try to extract a type schema from the function's annotations.

    For dataclass types, uses ``dataclasses.fields()``.
    For regular functions, inspects the signature.
    """
    import dataclasses
    if dataclasses.is_dataclass(fn):
        schema: Dict[str, Any] = {"type": "object", "properties": {}}
        for f in dataclasses.fields(fn):
            schema["properties"][f.name] = _type_hint_to_schema(f.type)
        return schema
    try:
        sig = _safe_signature(fn)
    except (ValueError, TypeError):
        return {"type": "any"}
    hints = getattr(fn, "__annotations__", {})
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (
        p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD
    )]
    # Skip 'self' if present
    if positional and positional[0].name in ("self", "cls"):
        positional = positional[1:]

    schema = {"type": "object", "properties": {}}
    for p in positional:
        ann = hints.get(p.name, Any)
        schema["properties"][p.name] = _type_hint_to_schema(ann)
    return schema


def _infer_param_names(fn: Callable) -> List[str]:
    """Return ordered parameter names.

    For dataclass types, uses ``dataclasses.fields()`` to get the actual
    field names.  For regular functions, returns all param names.
    """
    import dataclasses
    if dataclasses.is_dataclass(fn):
        return [f.name for f in dataclasses.fields(fn)]
    try:
        sig = _safe_signature(fn)
    except (ValueError, TypeError):
        return []
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (
        p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD
    )]
    # Skip 'self' if present
    if positional and positional[0].name in ("self", "cls"):
        positional = positional[1:]
    return [p.name for p in positional]


def _safe_signature(fn: Callable):
    """Wrap inspect.signature to handle builtins and partials."""
    import inspect
    try:
        return inspect.signature(fn)
    except (ValueError, TypeError):
        return inspect.Signature()


def _type_hint_to_schema(ann: Any) -> Dict[str, Any]:
    """Map a Python type annotation to a JSON-like schema dict."""
    origin = getattr(ann, "__origin__", None)
    if origin is not None:
        args = getattr(ann, "__args__", ())
        if origin is list or origin is List:
            return {"type": "array", "items": _type_hint_to_schema(args[0]) if args else {"type": "any"}}
        if origin is dict or origin is Dict:
            return {"type": "object"}
        if origin is Union or origin is Optional:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return _type_hint_to_schema(non_none[0])
            return {"type": "any"}
        return {"type": "any"}
    if ann is str:
        return {"type": "string"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is bool:
        return {"type": "boolean"}
    return {"type": "any"}


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

registry = ConditionRegistry()

# Lazy auto-population on first access.
_auto_populated: bool = False


def _ensure_populated() -> None:
    global _auto_populated
    if _auto_populated:
        return
    try:
        from core.primitive import default_registry
        count = registry.populate_from_primitive_registry(default_registry)
        _auto_populated = True
    except ImportError:
        _auto_populated = True
