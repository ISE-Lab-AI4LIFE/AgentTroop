import itertools
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.condition import ConditionDef, ConditionRegistry, registry as _condition_registry
from core.primitive import (
    Classifier,
    ContainsWordPredicate,
    ContainsAnyWordPredicate,
    ContainsAllWordsPredicate,
    EndsWithPredicate,
    IntentPredicate,
    LengthGtPredicate,
    LengthLtPredicate,
    MatchesRegexPredicate,
    Predicate,
    Primitive,
    PrimitiveRegistry,
    SemanticScorePrimitive,
    SentimentPredicate,
    StartsWithPredicate,
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
SEMANTIC_THRESHOLD_CANDIDATES = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
LENGTH_THRESHOLDS = [50, 100, 200]


@dataclass
class PrimitiveCatalog:
    predicates: List[Predicate] = field(default_factory=list)
    transforms: List[Transform] = field(default_factory=list)
    classifiers: List[Classifier] = field(default_factory=list)
    conditions: List[ConditionDef] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.predicates or self.transforms or self.classifiers or self.conditions)

    def total_primitives(self) -> int:
        return len(self.predicates) + len(self.transforms) + len(self.classifiers) + len(self.conditions)


class GrammarExporter:
    def __init__(
        self,
        primitive_registry: Optional[PrimitiveRegistry] = None,
        condition_registry: Optional[ConditionRegistry] = None,
        ontology_memory: Optional[Any] = None,
        max_depth: int = 3,
    ) -> None:
        self.primitive_registry = primitive_registry or default_registry
        self.condition_registry = condition_registry or _condition_registry
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
            elif ptype == "condition":
                try:
                    catalog.conditions.append(self.condition_registry.get(pname))
                except KeyError:
                    pass
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
        # Also pull condition-only entries from ConditionRegistry
        for cond in self.condition_registry:
            if cond.name not in names:
                catalog.conditions.append(cond)
        return catalog

    def get_parameterized_primitives(
        self, examples: List[Tuple[str, int]]
    ) -> PrimitiveCatalog:
        """Build a diverse, parameterized catalog of primitives from examples.

        Key improvements over the previous implementation:
        - Generates appropriate parameter sets for ALL 29 predicate types,
          not just ContainsWord / LengthGt / MatchesRegex.
        - Generates transform-wrapped predicate instances so enumeration
          includes ``ApplyTransformNode`` composites at depth 1.
        - Enforces diversity across predicate families and transform
          families (no single family >50%).
        """
        base = self.get_primitives()
        result = PrimitiveCatalog()

        keywords = _extract_keywords(examples)

        from collections import Counter
        family_counts: Counter = Counter()

        def _add_pred(p: Predicate) -> None:
            fam = type(p).__name__.replace("Predicate", "")
            result.predicates.append(p)
            family_counts[fam] += 1

        def _family_size(fam: str) -> int:
            return family_counts.get(fam, 0)

        for p in base.predicates:
            ptype = type(p).__name__

            # 1. ContainsWord — keyword-parameterized from examples
            if isinstance(p, ContainsWordPredicate):
                for kw in keywords[:3]:
                    _add_pred(ContainsWordPredicate(word=kw))

            # 2. ContainsAnyWord / ContainsAllWords — keyword-parameterized
            elif isinstance(p, ContainsAnyWordPredicate):
                if keywords:
                    for split_size in [1, 2]:
                        kws = keywords[:split_size]
                        _add_pred(ContainsAnyWordPredicate(words=kws))
                        _add_pred(ContainsAllWordsPredicate(words=kws))

            # 3. Length predicates — diverse thresholds from examples
            elif isinstance(p, LengthGtPredicate):
                for t in LENGTH_THRESHOLDS + _infer_length_thresholds(examples):
                    if _family_size("Length") < 4:
                        _add_pred(LengthGtPredicate(threshold=t))

            elif isinstance(p, LengthLtPredicate):
                for t in LENGTH_THRESHOLDS + _infer_length_thresholds(examples):
                    if _family_size("Length") < 4:
                        _add_pred(LengthLtPredicate(threshold=t))

            # 4. MatchesRegex — common jailbreak / keyword patterns
            elif isinstance(p, MatchesRegexPredicate):
                patterns = _infer_regex_patterns(examples, keywords)
                for pat in patterns:
                    _add_pred(MatchesRegexPredicate(pattern=pat))

            # 5. StartsWith / EndsWith — prefix/suffix from keywords
            elif isinstance(p, (StartsWithPredicate, EndsWithPredicate)):
                for kw in keywords[:2]:
                    _add_pred(type(p)(prefix=kw) if isinstance(p, StartsWithPredicate) else type(p)(suffix=kw))

            # 6. Sentiment — diverse thresholds
            elif isinstance(p, SentimentPredicate):
                for t in [0.3, 0.5, 0.7]:
                    _add_pred(SentimentPredicate(threshold=t))

            # 7. Intent — common harmful intents
            elif isinstance(p, IntentPredicate):
                for intent in ["harmful", "jailbreak", "roleplay"]:
                    _add_pred(IntentPredicate(intent_type=intent))

            # 8. All other predicates — add one instance (they are parameterless
            #    or already correct with default params)
            else:
                fam = ptype.replace("Predicate", "")
                if _family_size(fam) == 0:
                    _add_pred(p)

        # Enforce diversity: no family >50% of total
        total = len(result.predicates)
        if total > 0:
            for fam, cnt in family_counts.most_common():
                if cnt / total > 0.5:
                    excess = int(cnt - total * 0.5)
                    removed = 0
                    new_preds = []
                    for p in reversed(result.predicates):
                        pfam = type(p).__name__.replace("Predicate", "")
                        if pfam == fam and removed < excess:
                            removed += 1
                            continue
                        new_preds.insert(0, p)
                    result.predicates = new_preds

        if not result.predicates:
            result.predicates = list(base.predicates)

        # --- Transforms: select a diverse subset, avoid flooding ---
        base_transforms = self._select_diverse_transforms(base.transforms, max_transforms=8)
        result.transforms = list(base_transforms)
        result.classifiers = list(base.classifiers)
        result.conditions = list(base.conditions)
        return result

    @staticmethod
    def _select_diverse_transforms(
        transforms: List[Transform], max_transforms: int = 8,
    ) -> List[Transform]:
        """Select a diverse subset of transforms, one per family.

        Families: encoding (rot13/base64/hex), obfuscation (zero-width,
        unicode), casing (upper/lower/random), structural (prefix/suffix/
        truncate/pad), adversarial (typos/shuffle/synonyms), formatting
        (json/markdown/html), cipher (caesar/atbash/vigenere).
        """
        families: Dict[str, List[Transform]] = {}
        for t in transforms:
            name_lower = t.name.lower()
            if any(x in name_lower for x in ("rot13", "base64", "hex", "binary", "quoted")):
                fam = "encoding"
            elif any(x in name_lower for x in ("zero_width", "unicode", "morse", "pig_latin")):
                fam = "obfuscation"
            elif any(x in name_lower for x in ("lower", "upper", "random_case")):
                fam = "casing"
            elif any(x in name_lower for x in ("prefix", "suffix", "truncate", "pad")):
                fam = "structural"
            elif any(x in name_lower for x in ("typo", "shuffle", "synonym", "substitution")):
                fam = "adversarial"
            elif any(x in name_lower for x in ("json", "markdown", "html", "escape")):
                fam = "formatting"
            elif any(x in name_lower for x in ("caesar", "atbash", "vigenere", "rail", "boustrophedon")):
                fam = "cipher"
            elif any(x in name_lower for x in ("roleplay", "interrogative", "imperative", "declarative")):
                fam = "linguistic"
            elif any(x in name_lower for x in ("remove", "whitespace", "vowel")):
                fam = "filter"
            else:
                fam = "other"
            families.setdefault(fam, []).append(t)
        selected: List[Transform] = []
        # Round-robin across families
        while len(selected) < max_transforms:
            added = 0
            for fam in sorted(families.keys()):
                if families[fam]:
                    selected.append(families[fam].pop(0))
                    added += 1
                    if len(selected) >= max_transforms:
                        break
            if added == 0:
                break
        return selected

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
        programs: List[Program] = []

        # ALWAYS_ACCEPT baseline
        programs.append(Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=ContainsWordPredicate(word="")),
                then_outcome=0, else_outcome=0,
            )
        ))
        # ALWAYS_REFUSE baseline
        programs.append(Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=ContainsWordPredicate(word="")),
                then_outcome=1, else_outcome=1,
            )
        ))

        for cond in conditions:
            # Variant A: IF cond THEN REFUSE ELSE ACCEPT
            programs.append(Program(
                root=IfThenElseNode(
                    condition=cond, then_outcome=1, else_outcome=0
                )
            ))
            # Variant B: IF cond THEN ACCEPT ELSE REFUSE
            programs.append(Program(
                root=IfThenElseNode(
                    condition=cond, then_outcome=0, else_outcome=1
                )
            ))

        programs.sort(key=lambda p: p.complexity())
        return programs

    def export_to_smtlib(
        self,
        examples: List[Tuple[str, int]],
        output_file: Optional[str] = None,
        max_depth: Optional[int] = None,
        use_free_thresholds: bool = False,
    ) -> str:
        """Export SMT-LIB using the full SMTConstraintBuilder from core.grammar.

        Delegates to ``SMTConstraintBuilder.build_smtlib()`` for a complete
        encoding of the hypothesis space with depth-limited composition,
        complexity constraints, and optional error tolerance.
        """
        from core.grammar import SMTConstraintBuilder

        catalog = self.get_parameterized_primitives(examples)
        depth = max_depth if max_depth is not None else self.max_depth

        builder = SMTConstraintBuilder(
            predicates=catalog.predicates,
            transforms=catalog.transforms,
            classifiers=catalog.classifiers,
            max_depth=depth,
            use_complexity_constraint=True,
            max_complexity=min(10, 2**depth),
            allow_error_rate=0.0,
        )
        result = builder.build_smtlib(
            examples,
            use_free_thresholds=use_free_thresholds,
        )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            logger.info("Exported SMT-LIB to %s", output_file)

        return result


def _infer_length_thresholds(examples: List[Tuple[str, int]]) -> List[int]:
    """Infer length thresholds from example prompts."""
    thresholds: List[int] = []
    if not examples:
        return thresholds
    lengths = [len(p) for p, _ in examples]
    avg = sum(lengths) / len(lengths) if lengths else 100
    thresholds.append(max(10, int(avg * 0.5)))
    thresholds.append(int(avg * 1.5))
    return list(set(t for t in thresholds if 10 <= t <= 2000))


def _infer_regex_patterns(
    examples: List[Tuple[str, int]], keywords: List[str],
) -> List[str]:
    """Build regex patterns from example keywords plus common fallbacks."""
    patterns: List[str] = []
    if keywords:
        escaped = [re.escape(kw) for kw in keywords[:3]]
        if len(escaped) >= 2:
            patterns.append(r"(?i)\b(" + "|".join(escaped) + r")\b")
        elif escaped:
            patterns.append(r"(?i)\b" + escaped[0] + r"\b")
    static_patterns = [
        r"(?i)\b(kill|bomb|weapon|attack|hack)\b",
        r"(?i)\b(ignore|override|bypass)\b",
        r"(?i)\b(password|secret|admin)\b",
    ]
    for sp in static_patterns:
        if sp not in patterns:
            patterns.append(sp)
    return patterns[:3]


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
                thresholds = SEMANTIC_THRESHOLD_CANDIDATES if isinstance(c, SemanticScorePrimitive) else THRESHOLD_CANDIDATES
                for t in thresholds:
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


# =============================================================================
# Ontology consistency analysis
# =============================================================================

# Map from predicate symbolic name to condition parser pattern and
# hypothesis rendering template.  This is the canonical ontology registry
# that bridges the three representations:
#   1. DSL predicate (core/primitive.py)
#   2. Condition parser pattern (strategist._score_condition)
#   3. Hypothesis generation template (cognitive._fallback_hypotheses)
#
# Each entry: predicate_name -> (parser_has_handler, hypothesis_template)
PREDICATE_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "contains_word": {
        "parser_supported": True,
        "hypothesis_template": "IF contains_word('{keyword}') THEN REFUSE",
        "dsl_class": "ContainsWordPredicate",
        "category": "lexical",
    },
    "contains_any_word": {
        "parser_supported": True,
        "hypothesis_template": "IF contains_any_word(['{keyword}']) THEN REFUSE",
        "dsl_class": "ContainsAnyWordPredicate",
        "category": "lexical",
    },
    "contains_all_words": {
        "parser_supported": False,
        "hypothesis_template": "IF contains_all_words(['{keyword}']) THEN REFUSE",
        "dsl_class": "ContainsAllWordsPredicate",
        "category": "lexical",
    },
    "length_gt": {
        "parser_supported": True,  # via char_count(prompt) > N
        "hypothesis_template": "IF char_count(prompt) > {threshold} THEN REFUSE",
        "dsl_class": "LengthGtPredicate",
        "category": "structural",
    },
    "length_lt": {
        "parser_supported": True,  # via char_count(prompt) < N
        "hypothesis_template": "IF char_count(prompt) < {threshold} THEN REFUSE",
        "dsl_class": "LengthLtPredicate",
        "category": "structural",
    },
    "matches_regex": {
        "parser_supported": True,
        "hypothesis_template": "IF matches_regex(r'{pattern}') THEN REFUSE",
        "dsl_class": "MatchesRegexPredicate",
        "category": "lexical",
    },
    "starts_with": {
        "parser_supported": False,
        "hypothesis_template": "IF starts_with('{prefix}') THEN REFUSE",
        "dsl_class": "StartsWithPredicate",
        "category": "lexical",
    },
    "ends_with": {
        "parser_supported": False,
        "hypothesis_template": "IF ends_with('{suffix}') THEN REFUSE",
        "dsl_class": "EndsWithPredicate",
        "category": "lexical",
    },
    "has_number": {
        "parser_supported": True,
        "hypothesis_template": "IF has_number(prompt) THEN REFUSE",
        "dsl_class": "HasNumberPredicate",
        "category": "structural",
    },
    "has_special_char": {
        "parser_supported": False,
        "hypothesis_template": "IF has_special_char(prompt) THEN REFUSE",
        "dsl_class": "HasSpecialCharPredicate",
        "category": "structural",
    },
    "is_all_caps": {
        "parser_supported": False,
        "hypothesis_template": "IF is_all_caps(prompt) THEN REFUSE",
        "dsl_class": "IsAllCapsPredicate",
        "category": "structural",
    },
    "contains_leet": {
        "parser_supported": True,
        "hypothesis_template": "IF contains_leet(prompt) THEN REFUSE",
        "dsl_class": "ContainsLeetPredicate",
        "category": "structural",
    },



    "is_empty": {
        "parser_supported": False,
        "hypothesis_template": "IF is_empty(prompt) THEN REFUSE",
        "dsl_class": "IsEmptyPredicate",
        "category": "structural",
    },
    "starts_with_roleplay": {
        "parser_supported": False,
        "hypothesis_template": "IF starts_with_roleplay(prompt) THEN REFUSE",
        "dsl_class": "StartsWithRoleplayPredicate",
        "category": "jailbreak_specific",
    },
    "contains_system_override": {
        "parser_supported": False,
        "hypothesis_template": "IF contains_system_override(prompt) THEN REFUSE",
        "dsl_class": "ContainsSystemOverridePredicate",
        "category": "jailbreak_specific",
    },
    "contains_delimiter": {
        "parser_supported": False,
        "hypothesis_template": "IF contains_delimiter(prompt) THEN REFUSE",
        "dsl_class": "ContainsDelimiterPredicate",
        "category": "structural",
    },
    "contains_code_block": {
        "parser_supported": False,
        "hypothesis_template": "IF contains_code_block(prompt) THEN REFUSE",
        "dsl_class": "ContainsCodeBlockPredicate",
        "category": "structural",
    },
    "has_emoji": {
        "parser_supported": False,
        "hypothesis_template": "IF has_emoji(prompt) THEN REFUSE",
        "dsl_class": "HasEmojiPredicate",
        "category": "structural",
    },
    "contains_url": {
        "parser_supported": False,
        "hypothesis_template": "IF contains_url(prompt) THEN REFUSE",
        "dsl_class": "ContainsURLPredicate",
        "category": "structural",
    },
    "sentiment": {
        "parser_supported": False,
        "hypothesis_template": "IF sentiment(prompt) > {threshold} THEN REFUSE",
        "dsl_class": "SentimentPredicate",
        "category": "semantic",
    },
    "intent": {
        "parser_supported": False,
        "hypothesis_template": "IF intent(prompt) = '{intent_type}' THEN REFUSE",
        "dsl_class": "IntentPredicate",
        "category": "semantic",
    },
    "matches_jailbreak_pattern": {
        "parser_supported": False,
        "hypothesis_template": "IF matches_jailbreak_pattern(prompt) THEN REFUSE",
        "dsl_class": "MatchesJailbreakPatternPredicate",
        "category": "jailbreak_specific",
    },
    "contains_encoding_wrapper": {
        "parser_supported": False,
        "hypothesis_template": "IF contains_encoding_wrapper(prompt) THEN REFUSE",
        "dsl_class": "ContainsEncodingWrapperPredicate",
        "category": "jailbreak_specific",
    },
    "is_repetitive": {
        "parser_supported": False,
        "hypothesis_template": "IF is_repetitive(prompt) THEN REFUSE",
        "dsl_class": "IsRepetitivePredicate",
        "category": "structural",
    },
    # Discourse patterns (FLAW-6): now have DSL classes
    "is_grammatical_question": {
        "parser_supported": True,
        "hypothesis_template": "IF is_grammatical_question(prompt) THEN ACCEPT",
        "dsl_class": "IsGrammaticalQuestionPredicate",
        "category": "discourse",
    },
    "starts_with_imperative": {
        "parser_supported": True,
        "hypothesis_template": "IF starts_with_imperative(prompt) THEN ACCEPT",
        "dsl_class": "StartsWithImperativePredicate",
        "category": "discourse",
    },
    "is_instruction_request": {
        "parser_supported": True,
        "hypothesis_template": "IF is_instruction_request(prompt) THEN REFUSE",
        "dsl_class": "IsInstructionRequestPredicate",
        "category": "discourse",
    },
    "instruction_score": {
        "parser_supported": False,
        "hypothesis_template": "IF instruction_score(prompt) > {threshold} THEN REFUSE",
        "dsl_class": "InstructionScorePrimitive",
        "category": "semantic_score",
    },
}


def analyze_dsl_coverage() -> Dict[str, Any]:
    """Analyze coverage across the three ontology representations.

    Returns a report with:
    - **parser_rules**: all condition patterns supported by ``_score_condition``
    - **predicate_registry**: all predicates registered in DSL
    - **hypothesis_templates**: all condition types generated by fallback
    - **orphan_predicates**: predicates with no parser rule
    - **orphan_conditions**: condition patterns with no DSL predicate
    - **covered**: predicates with both parser and hypothesis template
    - **coverage_by_category**: lexical, structural, semantic, discourse, jailbreak
    """
    parser_rules = {
        name for name, info in PREDICATE_ONTOLOGY.items()
        if info["parser_supported"]
    }
    dsl_predicates = {
        name for name, info in PREDICATE_ONTOLOGY.items()
        if info["dsl_class"] is not None
    }
    all_registered = set(PREDICATE_ONTOLOGY.keys())

    orphan_predicates = dsl_predicates - parser_rules
    orphan_conditions = parser_rules - dsl_predicates

    covered = dsl_predicates & parser_rules
    total_dsl = len(dsl_predicates)
    total_parser = len(parser_rules)

    by_category: Dict[str, Dict[str, int]] = {}
    for name, info in PREDICATE_ONTOLOGY.items():
        cat = info.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "parser_supported": 0, "dsl_class": 0}
        by_category[cat]["total"] += 1
        if info["parser_supported"]:
            by_category[cat]["parser_supported"] += 1
        if info["dsl_class"] is not None:
            by_category[cat]["dsl_class"] += 1

    return {
        "parser_rules": sorted(parser_rules),
        "predicate_registry": sorted(dsl_predicates),
        "hypothesis_templates": sorted(all_registered),
        "orphan_predicates": sorted(orphan_predicates),
        "orphan_conditions": sorted(orphan_conditions),
        "covered": sorted(covered),
        "dsl_coverage_pct": round(len(covered) / max(total_dsl, 1) * 100, 1),
        "parser_coverage_pct": round(len(covered) / max(total_parser, 1) * 100, 1),
        "total_dsl_predicates": total_dsl,
        "total_parser_rules": total_parser,
        "total_ontology_entries": len(all_registered),
        "coverage_by_category": by_category,
    }


def _safe_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe if safe else "prim"


def _escape_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
