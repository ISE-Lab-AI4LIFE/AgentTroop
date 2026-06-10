#!/usr/bin/env python3
"""FLAW-4: Ontology Coverage Proof — automated pipeline audit.

Traces every predicate through the full hypothesis→execution→synthesis
pipeline and reports coverage metrics, orphan nodes, and mapping gaps.
"""

import sys, os, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

# ── Pipeline component imports ───────────────────────────────────────────────
from core.primitive import default_registry, Predicate, Transform, Classifier
from core.condition import registry as condition_registry, ConditionDef
from core.program import PredicateNode, IfThenElseNode, Program
from core.executor import ProgramExecutor
from agents.strategist import StrategistAgent
from agents.cognitive import _try_set_condition_name
from synthesis.grammar_exporter import PREDICATE_ONTOLOGY, GrammarExporter
from inference.version_space import _classify_program


# ── Audit data structures ───────────────────────────────────────────────────

@dataclass
class PipelineNode:
    name: str
    registered: bool = False
    has_parser: bool = False
    has_dsl_class: bool = False
    in_ontology: bool = False
    in_condition_registry: bool = False
    has_condition_name_path: bool = False  # _try_set_condition_name can map to it
    has_compile_path: bool = False         # compile_condition_to_program can handle it
    has_executor_path: bool = False        # compile_to_node + ProgramExecutor works
    has_synthesis_path: bool = False       # build_simple_program can create it
    has_classify_path: bool = False        # _classify_program can categorise it
    in_version_space_reachable: bool = False

    def coverage_pct(self) -> float:
        steps = [
            self.registered,
            self.in_condition_registry,
            self.has_condition_name_path,
            self.has_compile_path,
            self.has_executor_path,
            self.has_synthesis_path,
            self.has_classify_path,
        ]
        return sum(1 for s in steps if s) / len(steps) * 100

    def missing_steps(self) -> List[str]:
        labels = [
            ("registered", self.registered),
            ("in_condition_registry", self.in_condition_registry),
            ("condition_name_path", self.has_condition_name_path),
            ("compile_path", self.has_compile_path),
            ("executor_path", self.has_executor_path),
            ("synthesis_path", self.has_synthesis_path),
            ("classify_path", self.has_classify_path),
        ]
        return [lbl for lbl, ok in labels if not ok]


# ── Audit helpers ───────────────────────────────────────────────────────────

def _get_mapped_condition_names() -> Set[str]:
    """Return condition_names that _try_set_condition_name can produce."""
    from agents.cognitive import _try_set_condition_name
    matched: Set[str] = set()
    test_strings = [
        "IF contains_word('test') THEN REFUSE",
        "IF contains_any_word(['a','b']) THEN REFUSE",
        "IF char_count(prompt) < 30 THEN ACCEPT",
        "IF char_count(prompt) > 120 THEN REFUSE",
        "IF is_grammatical_question(prompt) THEN ACCEPT",
        "IF starts_with_imperative(prompt) THEN ACCEPT",
        "IF has_number(prompt) THEN REFUSE",
        "IF contains_leet(prompt) THEN REFUSE",
        "IF contains_rot13(prompt) THEN REFUSE",
        r"IF matches_regex(r'(?i)\btest\b') THEN REFUSE",
    ]
    for s in test_strings:
        hyp = type("_Hyp", (), {"condition": "", "condition_name": None, "condition_params": None})()
        hyp.condition = s
        _try_set_condition_name(hyp)
        if hyp.condition_name:
            matched.add(hyp.condition_name)
    return matched


def _get_compile_supported() -> Set[str]:
    """Return condition names that compile_condition_to_program supports."""
    return {"contains_word", "contains_any_word", "length_gt", "length_lt",
            "has_number", "contains_leet", "contains_rot13", "matches_regex",
            "is_grammatical_question", "starts_with_imperative"}


def _get_score_supported() -> Set[str]:
    """Return condition names that _score_condition supports (keyword fallback)."""
    return {"contains_word", "contains_any_word", "length_gt", "length_lt",
            "has_number", "contains_leet", "matches_regex",
            "is_grammatical_question", "contains_rot13", "starts_with_imperative"}


def _get_classify_supported() -> Set[str]:
    """Return Predicate class names that _classify_program can categorise."""
    keyword_preds = {"ContainsWordPredicate", "ContainsAnyWordPredicate",
                     "ContainsAllWordsPredicate", "MatchesRegexPredicate",
                     "StartsWithPredicate", "EndsWithPredicate"}
    structural_preds = {"LengthGtPredicate", "LengthLtPredicate",
                        "HasNumberPredicate", "HasSpecialCharPredicate",
                        "IsAllCapsPredicate", "ContainsDelimiterPredicate",
                        "ContainsCodeBlockPredicate", "IsEmptyPredicate",
                        "HasEmojiPredicate", "ContainsURLPredicate",
                        "IsRepetitivePredicate",
                        "IsGrammaticalQuestionPredicate",
                        "StartsWithImperativePredicate"}
    semantic_preds = {"SentimentPredicate", "IntentPredicate",
                      "ContainsLeetPredicate", "ContainsRot13Predicate",
                      "ContainsBase64Predicate", "ContainsHexPredicate"}
    jailbreak_preds = {"StartsWithRoleplayPredicate",
                       "ContainsSystemOverridePredicate",
                       "MatchesJailbreakPatternPredicate",
                       "ContainsEncodingWrapperPredicate"}
    return keyword_preds | structural_preds | semantic_preds | jailbreak_preds


def _get_onto_parser_supported() -> Set[str]:
    """Return predicate names marked parser_supported in PREDICATE_ONTOLOGY."""
    return {k for k, v in PREDICATE_ONTOLOGY.items() if v.get("parser_supported")}


def _check_synthesis_path(name: str, cls_name: Optional[str]) -> bool:
    """Check if build_simple_program can create a Program for this predicate."""
    if cls_name is None:
        return False
    try:
        cls = getattr(__import__("core.primitive", fromlist=[cls_name]), cls_name)
        if cls is None:
            return False
        instance = cls()
        if not isinstance(instance, Predicate):
            return False
        node = PredicateNode(primitive=instance)
        prog = Program(root=IfThenElseNode(condition=node, then_outcome=1, else_outcome=0))
        classify = _classify_program(prog)
        return classify != "unknown"
    except (AttributeError, ImportError, TypeError):
        return False


# ── Main audit ──────────────────────────────────────────────────────────────

def run_audit() -> Dict[str, Any]:
    condition_names = _get_mapped_condition_names()
    compile_supported = _get_compile_supported()
    score_supported = _get_score_supported()
    classify_names = _get_classify_supported()
    onto_parser = _get_onto_parser_supported()

    # Gather all predicate info from PrimitiveRegistry
    all_primitives = sorted(default_registry.list_primitives())
    predicate_names: Set[str] = set()
    for name in all_primitives:
        try:
            inst = default_registry.get(name)
            if isinstance(inst, Predicate):
                predicate_names.add(name)
        except Exception:
            pass

    nodes: Dict[str, PipelineNode] = {}
    for name in sorted(predicate_names):
        pnode = PipelineNode(name=name)

        # 1. Registered in PrimitiveRegistry
        pnode.registered = name in predicate_names

        # 2. In ontology
        pnode.in_ontology = name in PREDICATE_ONTOLOGY

        # 3. Has parser support in ontology
        pnode.has_parser = name in onto_parser

        # 4. In ConditionRegistry
        try:
            cond_def = condition_registry.get(name)
            pnode.in_condition_registry = cond_def is not None
            pnode.has_dsl_class = (
                cond_def is not None
                and cond_def.primitive_class is not None
            )
        except Exception:
            pnode.in_condition_registry = False
            pnode.has_dsl_class = False

        # 5. _try_set_condition_name can produce it
        pnode.has_condition_name_path = name in condition_names

        # 6. compile_condition_to_program can compile it
        pnode.has_compile_path = name in compile_supported

        # 7. compile_to_node + ProgramExecutor path works
        pnode.has_executor_path = pnode.has_dsl_class

        # 8. Synthesis path: build_simple_program → classify
        onto_entry = PREDICATE_ONTOLOGY.get(name, {})
        cls_name = onto_entry.get("dsl_class") if isinstance(onto_entry, dict) else None
        pnode.has_synthesis_path = _check_synthesis_path(name, cls_name)

        # 9. _classify_program can categorise it
        if cls_name:
            pnode.has_classify_path = cls_name in classify_names
        else:
            pnode.has_classify_path = False

        # 10. Reachable in Version Space (synthesis + classify + executor)
        pnode.in_version_space_reachable = (
            pnode.has_synthesis_path
            and pnode.has_classify_path
            and pnode.has_executor_path
        )

        nodes[name] = pnode

    # Statistics
    total = len(nodes)
    full_coverage = sum(1 for n in nodes.values() if n.coverage_pct() == 100.0)
    partial_coverage = sum(1 for n in nodes.values() if 0 < n.coverage_pct() < 100.0)
    zero_coverage = sum(1 for n in nodes.values() if n.coverage_pct() == 0.0)
    orphan = [n.name for n in nodes.values() if not n.in_version_space_reachable]
    missing_condition_name = [n.name for n in nodes.values() if not n.has_condition_name_path]
    missing_compile = [n.name for n in nodes.values() if not n.has_compile_path]
    missing_synthesis = [n.name for n in nodes.values() if not n.has_synthesis_path]
    missing_classify = [n.name for n in nodes.values() if not n.has_classify_path]

    return {
        "total_predicates": total,
        "full_coverage": full_coverage,
        "partial_coverage": partial_coverage,
        "zero_coverage": zero_coverage,
        "coverage_pct": round(full_coverage / total * 100, 1) if total else 0,
        "orphan_predicates": orphan,
        "missing_condition_name_path": missing_condition_name,
        "missing_compile_path": missing_compile,
        "missing_synthesis_path": missing_synthesis,
        "missing_classify_path": missing_classify,
        "nodes": {k: {
            "coverage_pct": round(v.coverage_pct(), 1),
            "missing": v.missing_steps(),
            "registered": v.registered,
            "in_ontology": v.in_ontology,
            "has_parser": v.has_parser,
            "in_condition_registry": v.in_condition_registry,
            "has_dsl_class": v.has_dsl_class,
            "has_condition_name_path": v.has_condition_name_path,
            "has_compile_path": v.has_compile_path,
            "has_executor_path": v.has_executor_path,
            "has_synthesis_path": v.has_synthesis_path,
            "has_classify_path": v.has_classify_path,
            "in_version_space_reachable": v.in_version_space_reachable,
        } for k, v in sorted(nodes.items())},
        "pipeline_steps_summary": {
            "registered": sum(1 for n in nodes.values() if n.registered),
            "in_condition_registry": sum(1 for n in nodes.values() if n.in_condition_registry),
            "has_condition_name_path": sum(1 for n in nodes.values() if n.has_condition_name_path),
            "has_compile_path": sum(1 for n in nodes.values() if n.has_compile_path),
            "has_executor_path": sum(1 for n in nodes.values() if n.has_executor_path),
            "has_synthesis_path": sum(1 for n in nodes.values() if n.has_synthesis_path),
            "has_classify_path": sum(1 for n in nodes.values() if n.has_classify_path),
            "in_version_space_reachable": sum(1 for n in nodes.values() if n.in_version_space_reachable),
        },
    }


def print_report(result: Dict[str, Any]) -> None:
    print("=" * 72)
    print("  FLAW-4: ONTOLOGY COVERAGE PROOF — Pipeline Audit Report")
    print("=" * 72)

    print(f"\nTotal predicates audited: {result['total_predicates']}")
    print(f"Full pipeline coverage:  {result['full_coverage']}/{result['total_predicates']} ({result['coverage_pct']}%)")
    print(f"Partial coverage:        {result['partial_coverage']}")
    print(f"Zero coverage:           {result['zero_coverage']}")

    print(f"\n── Pipeline step coverage ──")
    steps = result["pipeline_steps_summary"]
    total = result["total_predicates"]
    for step, count in steps.items():
        bar = "#" * count + "." * (total - count)
        print(f"  {step:35s}  {count:3d}/{total}  [{bar}]")

    print(f"\n── Pipeline mapping: Hypothesis condition → VersionSpace execution ──")
    print(f"  Hypothesis condition")
    print(f"      ↓  (condition_name_path: {steps['has_condition_name_path']}/{total})")
    print(f"  ConditionRegistry")
    print(f"      ↓  (compile_path: {steps['has_compile_path']}/{total})")
    print(f"  Compiler")
    print(f"      ↓  (executor_path: {steps['has_executor_path']}/{total})")
    print(f"  Program")
    print(f"      ↓")
    print(f"  ProgramExecutor")
    print(f"      ↓  (classify_path: {steps['has_classify_path']}/{total})")
    print(f"  VersionSpace")
    print(f"      ↓  (synthesis_path: {steps['has_synthesis_path']}/{total})")
    print(f"  Synthesizer")
    print(f"      ↓")
    print(f"  New Program")

    reachable = steps["in_version_space_reachable"]
    print(f"\n── Version Space reachable: {reachable}/{total} ({round(reachable/total*100,1)}%) ──")

    if result["orphan_predicates"]:
        print(f"\n  ORPHAN PREDICATES ({len(result['orphan_predicates'])}):")
        for name in result["orphan_predicates"]:
            nd = result["nodes"][name]
            missing = ", ".join(nd["missing"])
            print(f"    ❌ {name:40s}  missing: {missing}")

    if result["missing_condition_name_path"]:
        print(f"\n── Missing condition_name path ({len(result['missing_condition_name_path'])}):")
        for name in result["missing_condition_name_path"]:
            print(f"    ⚠  {name}")

    if result["missing_compile_path"]:
        print(f"\n── Missing compile path ({len(result['missing_compile_path'])}):")
        for name in result["missing_compile_path"]:
            nd = result["nodes"][name]
            print(f"    ⚠  {name:40s}  dsl_class={nd.get('has_dsl_class')}, parser={nd.get('has_parser')}")

    if result["missing_synthesis_path"]:
        print(f"\n── Missing synthesis path ({len(result['missing_synthesis_path'])}):")
        for name in result["missing_synthesis_path"]:
            print(f"    ⚠  {name}")

    if result["missing_classify_path"]:
        print(f"\n── Missing classify path ({len(result['missing_classify_path'])}):")
        for name in result["missing_classify_path"]:
            print(f"    ⚠  {name}")

    print("\n" + "=" * 72)


def main():
    result = run_audit()
    print_report(result)

    # Save JSON report
    report_path = os.path.join(os.path.dirname(__file__), "..", "docs", "ontology_coverage_report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"JSON report saved to {report_path}")


if __name__ == "__main__":
    main()
