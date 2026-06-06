"""Tests for GrammarExporter — program enumeration and SMT-LIB export."""

from core.primitive import default_registry
from core.program import (
    AndNode,
    ApplyTransformNode,
    NotNode,
    OrNode,
    PredicateNode,
    ThresholdNode,
)
from synthesis.grammar_exporter import GrammarExporter


class TestGrammarExporter:
    def setup_method(self) -> None:
        self.exporter = GrammarExporter(
            primitive_registry=default_registry,
            max_depth=2,
        )

    def test_get_primitives_returns_catalog(self) -> None:
        catalog = self.exporter.get_primitives()
        assert catalog.total_primitives() > 0
        assert len(catalog.predicates) > 0
        assert len(catalog.transforms) > 0
        assert len(catalog.classifiers) > 0

    def test_get_primitives_includes_contains_word(self) -> None:
        catalog = self.exporter.get_primitives()
        names = [p.name for p in catalog.predicates]
        assert "contains_word" in names

    def test_get_primitives_includes_rot13(self) -> None:
        catalog = self.exporter.get_primitives()
        names = [t.name for t in catalog.transforms]
        assert "rot13" in names

    def test_get_primitives_includes_toxicity(self) -> None:
        catalog = self.exporter.get_primitives()
        names = [c.name for c in catalog.classifiers]
        assert "toxicity_score" in names

    def test_enumerate_conditions_depth1(self) -> None:
        conditions = self.exporter.enumerate_conditions(max_depth=1)
        assert len(conditions) > 0
        for cond in conditions:
            assert isinstance(cond, (PredicateNode, ThresholdNode))

    def test_enumerate_conditions_depth2_includes_not(self) -> None:
        conditions = self.exporter.enumerate_conditions(max_depth=2)
        has_not = any(isinstance(c, NotNode) for c in conditions)
        assert has_not, "Depth 2 should include NotNode structures"

    def test_enumerate_conditions_depth2_includes_and_or(self) -> None:
        from core.primitive import ContainsWordPredicate, LengthGtPredicate
        from synthesis.grammar_exporter import PrimitiveCatalog
        exporter = GrammarExporter(max_depth=2)
        exporter.get_primitives = lambda: PrimitiveCatalog(
            predicates=[
                ContainsWordPredicate(word="test"),
                LengthGtPredicate(threshold=5),
            ],
        )
        conditions = exporter.enumerate_conditions(max_programs=100)
        has_binary = any(
            isinstance(c, (AndNode, OrNode)) for c in conditions
        )
        assert has_binary, "Depth 2 should include AND/OR structures with few primitives"

    def test_enumerate_conditions_depth2_includes_apply(
        self,
    ) -> None:
        conditions = self.exporter.enumerate_conditions(max_depth=2)
        has_apply = any(
            isinstance(c, ApplyTransformNode) for c in conditions
        )
        assert has_apply, "Depth 2 should include ApplyTransformNode"

    def test_enumerate_programs_returns_programs(self) -> None:
        programs = self.exporter.enumerate_programs(max_depth=1)
        assert len(programs) > 0
        for prog in programs:
            assert prog.root is not None

    def test_enumerate_programs_all_have_ite_root(self) -> None:
        from core.program import IfThenElseNode

        programs = self.exporter.enumerate_programs(max_depth=2)
        for prog in programs:
            assert isinstance(prog.root, IfThenElseNode)

    def test_export_to_smtlib_returns_string(self) -> None:
        examples = [("hello", 0), ("bomb", 1)]
        smt = self.exporter.export_to_smtlib(examples)
        assert isinstance(smt, str)
        assert len(smt) > 50

    def test_export_to_smtlib_contains_examples(self) -> None:
        examples = [("test prompt", 1)]
        smt = self.exporter.export_to_smtlib(examples)
        assert "test prompt" in smt
        assert "check-sat" in smt
        assert "get-model" in smt

    def test_export_to_smtlib_includes_primitives(self) -> None:
        examples = [("hello", 0)]
        smt = self.exporter.export_to_smtlib(examples)
        assert "declare-fun" in smt
        assert "contains_word" in smt or "toxicity_score" in smt

    def test_empty_catalog_returns_empty_list(self) -> None:
        from synthesis.grammar_exporter import PrimitiveCatalog

        exporter = GrammarExporter(max_depth=2)
        exporter.get_primitives = lambda: PrimitiveCatalog()
        conditions = exporter.enumerate_conditions()
        assert conditions == []

    def test_smt_output_file(self, tmp_path) -> None:
        output = tmp_path / "test.smt2"
        examples = [("hello", 0)]
        self.exporter.export_to_smtlib(examples, output_file=str(output))
        assert output.exists()
        content = output.read_text()
        assert "(check-sat)" in content

    def test_unique_conditions_no_duplicates(self) -> None:
        conditions = self.exporter.enumerate_conditions(max_depth=2)
        str_reps = [str(c) for c in conditions]
        assert len(str_reps) == len(set(str_reps)), (
            "Conditions should be unique by string representation"
        )

    def test_many_primitives_still_works(self) -> None:
        examples = [("hello", 0), ("world", 1)]
        smt = self.exporter.export_to_smtlib(examples)
        assert smt
