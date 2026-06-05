from core.primitive import ContainsWordPredicate, ToxicityScoreClassifier

from adapters.toy_victims.hybrid_logic import (
    AndVictim,
    NotVictim,
    OrVictim,
    ThresholdVictim,
)
from adapters.toy_victims.rule_based import KeywordFilterVictim


class TestAndVictim:
    def test_refuses_when_all_refuse(self):
        v1 = KeywordFilterVictim(keywords=["bomb"])
        v2 = KeywordFilterVictim(keywords=["kill"])
        victim = AndVictim(victims=[v1, v2])
        # AND: refuses when ALL children refuse
        assert victim.respond("bomb and kill") == 1
        # "just bomb" → v1 refuses (true) AND v2 accepts (false) → false → ACCEPT
        assert victim.respond("just bomb") == 0
        assert victim.respond("hello") == 0

    def test_accepts_when_one_accepts(self):
        v1 = KeywordFilterVictim(keywords=["bomb"])
        v2 = KeywordFilterVictim(keywords=["kill"])
        victim = AndVictim(victims=[v1, v2])
        # "just bomb" → v1 refuses (true), v2 accepts (false) → AND=false → ACCEPT
        assert victim.respond("just bomb") == 0
        # "hello" → both accept (false) → AND=false → ACCEPT
        assert victim.respond("hello") == 0

    def test_ground_truth_program(self):
        v1 = KeywordFilterVictim(keywords=["bomb"])
        v2 = KeywordFilterVictim(keywords=["kill"])
        victim = AndVictim(victims=[v1, v2])
        gt = victim.get_ground_truth_program()
        assert gt is not None
        assert gt.complexity() > 2


class TestOrVictim:
    def test_refuses_when_any_refuses(self):
        v1 = KeywordFilterVictim(keywords=["bomb"])
        v2 = KeywordFilterVictim(keywords=["kill"])
        victim = OrVictim(victims=[v1, v2])
        assert victim.respond("bomb here") == 1
        assert victim.respond("kill here") == 1
        assert victim.respond("bomb and kill") == 1
        assert victim.respond("hello world") == 0


class TestNotVictim:
    def test_inverts_decision(self):
        inner = KeywordFilterVictim(keywords=["bomb"])
        victim = NotVictim(victim=inner)
        assert victim.respond("bomb") == 0  # inverted from 1
        assert victim.respond("hello") == 1  # inverted from 0

    def test_ground_truth_program(self):
        inner = KeywordFilterVictim(keywords=["bomb"])
        victim = NotVictim(victim=inner)
        gt = victim.get_ground_truth_program()
        assert gt is not None

    def test_double_negation(self):
        v1 = KeywordFilterVictim(keywords=["bomb"])
        v2 = NotVictim(victim=NotVictim(victim=v1))
        assert v1.respond("bomb") == v2.respond("bomb")


class TestThresholdVictim:
    def test_responds_within_bounds(self):
        classifier = ToxicityScoreClassifier()
        victim = ThresholdVictim(classifier=classifier, threshold=0.5)
        result = victim.respond("any prompt")
        assert result in (0, 1)

    def test_ground_truth_program(self):
        classifier = ToxicityScoreClassifier()
        victim = ThresholdVictim(classifier=classifier, threshold=0.5)
        gt = victim.get_ground_truth_program()
        assert gt is not None

    def test_metadata(self):
        classifier = ToxicityScoreClassifier()
        victim = ThresholdVictim(classifier=classifier, threshold=0.7)
        meta = victim.get_metadata()
        assert meta["threshold"] == 0.7
