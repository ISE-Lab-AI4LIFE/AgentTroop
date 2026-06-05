from adapters.toy_victims.rule_based import KeywordFilterVictim
from evaluation.scientific_discovery import (
    ScientificDiscoveryEvaluator,
    Theory,
)


class TestTheory:
    def test_create_theory(self):
        theory = Theory(
            id="t1",
            pattern="contains_danger_word",
            conditions={"danger_words": ["bomb", "kill"]},
            confidence=0.95,
        )
        assert theory.id == "t1"
        assert theory.confidence == 0.95

    def test_theory_serialization_roundtrip(self):
        theory = Theory(
            id="t1",
            pattern="keyword_filter",
            conditions={"danger_words": ["bomb"]},
            confidence=0.9,
            provenance=[{"source": "test", "details": {}}],
        )
        data = theory.to_dict()
        restored = Theory.from_dict(data)
        assert restored.id == theory.id
        assert restored.pattern == theory.pattern
        assert restored.confidence == theory.confidence


class TestScientificDiscoveryEvaluator:
    def test_evaluate_theory_perfect(self):
        victims = [
            KeywordFilterVictim(keywords=["bomb"]),
            KeywordFilterVictim(keywords=["kill"]),
        ]
        theory = Theory(
            id="t1",
            pattern="keyword_filter",
            conditions={"danger_words": ["bomb", "kill"]},
            confidence=0.9,
        )
        evaluator = ScientificDiscoveryEvaluator()
        # Theory predicts 1 if any danger word is found
        accuracy = evaluator.evaluate_theory(
            theory, victims, ["bomb here", "hello world"]
        )
        assert 0.0 <= accuracy <= 1.0

    def test_cross_family_transfer(self):
        source = [KeywordFilterVictim(keywords=["bomb"])]
        target = [KeywordFilterVictim(keywords=["kill"])]
        theory = Theory(
            id="t1",
            pattern="keyword_filter",
            conditions={"danger_words": ["bomb"]},
            confidence=0.9,
        )
        evaluator = ScientificDiscoveryEvaluator()
        score = evaluator.cross_family_transfer_score(
            theory, source, target, ["bomb here", "hello"]
        )
        assert 0.0 <= score <= 1.0

    def test_empty_victims(self):
        evaluator = ScientificDiscoveryEvaluator()
        theory = Theory(
            id="t1", pattern="test", conditions={}, confidence=0.5
        )
        assert evaluator.evaluate_theory(theory, [], ["test"]) == 0.0
