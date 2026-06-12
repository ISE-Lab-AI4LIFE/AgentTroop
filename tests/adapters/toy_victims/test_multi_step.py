from core.primitive import (
    ContainsWordPredicate,
    MatchesRegexPredicate,
    RemovePunctuationTransform,
    ToLowercaseTransform,
)

from adapters.toy_victims.multi_step import NormalizeThenFilterVictim


class TestNormalizeThenFilterVictim:
    def test_lowercase_and_remove_punctuation_then_filter(self):
        predicate = ContainsWordPredicate(word="bomb")
        victim = NormalizeThenFilterVictim(predicate=predicate)
        assert victim.respond("BOMB!") == 1
        assert victim.respond("bomb.") == 1
        assert victim.respond("safe text") == 0

    def test_ground_truth_consistency(self):
        predicate = ContainsWordPredicate(word="attack")
        victim = NormalizeThenFilterVictim(predicate=predicate)
        assert victim.respond("ATTACK!") == 1
        assert victim.respond("Attack.") == 1
        assert victim.respond("no danger") == 0

    def test_metadata(self):
        predicate = ContainsWordPredicate(word="bomb")
        victim = NormalizeThenFilterVictim(predicate=predicate)
        meta = victim.get_metadata()
        assert meta["type"] == "multi_step"
