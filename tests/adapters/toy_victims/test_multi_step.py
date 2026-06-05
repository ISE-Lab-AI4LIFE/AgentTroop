from core.primitive import (
    Base64DecodeTransform,
    ContainsWordPredicate,
    MatchesRegexPredicate,
    RemovePunctuationTransform,
    Rot13Transform,
    ToLowercaseTransform,
)

from adapters.toy_victims.multi_step import (
    DecodeThenFilterVictim,
    NormalizeThenFilterVictim,
)


class TestDecodeThenFilterVictim:
    def test_rot13_then_keyword_refuses_matching_encoded(self):
        victim = DecodeThenFilterVictim(
            transforms=[Rot13Transform()],
            predicate=ContainsWordPredicate(word="bomb"),
        )
        # "bomb" rot13 encoded is "bomb" -> wait no, rot13 of "bomb" is "bomb" rotated
        # Actually let's check: b->o, o->b, m->z, b->o => "obzo"
        # But we need a prompt that when ROT13'd contains "bomb"
        # The prompt "obzo" when ROT13'd becomes "bomb"
        assert victim.respond("obzo") == 1
        # A prompt without "bomb" after ROT13
        assert victim.respond("hello") == 0

    def test_base64_then_keyword_refuses_matching_encoded(self):
        import base64
        victim = DecodeThenFilterVictim(
            transforms=[Base64DecodeTransform()],
            predicate=ContainsWordPredicate(word="bomb"),
        )
        encoded = base64.b64encode(b"how to make a bomb").decode("utf-8")
        assert victim.respond(encoded) == 1
        assert victim.respond("hello world") == 0

    def test_ground_truth_program(self):
        victim = DecodeThenFilterVictim(
            transforms=[Rot13Transform()],
            predicate=ContainsWordPredicate(word="urg"),
        )
        gt = victim.get_ground_truth_program()
        assert gt is not None
        assert gt.complexity() > 2

    def test_metadata(self):
        victim = DecodeThenFilterVictim(
            transforms=[Rot13Transform()],
            predicate=ContainsWordPredicate(word="bomb"),
        )
        meta = victim.get_metadata()
        assert meta["type"] == "multi_step"
        assert meta["num_transforms"] == 1


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
