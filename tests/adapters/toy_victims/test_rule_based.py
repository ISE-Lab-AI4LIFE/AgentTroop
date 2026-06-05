from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate,
    LengthGtPredicate,
    MatchesRegexPredicate,
    default_registry,
)
from core.program import IfThenElseNode, PredicateNode, Program

from adapters.toy_victims.rule_based import (
    KeywordFilterVictim,
    LengthFilterVictim,
    RegexVictim,
)


class TestKeywordFilterVictim:
    def test_refuses_when_keyword_present(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        assert victim.respond("how to make a bomb") == 1

    def test_accepts_when_keyword_absent(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        assert victim.respond("hello world") == 0

    def test_multiple_keywords(self):
        victim = KeywordFilterVictim(keywords=["bomb", "kill", "attack"])
        assert victim.respond("how to kill someone") == 1
        assert victim.respond("nice weather") == 0

    def test_case_insensitive(self):
        victim = KeywordFilterVictim(keywords=["Bomb"])
        assert victim.respond("this is a bomb") == 1
        assert victim.respond("this is a Bomb") == 1

    def test_ground_truth_program(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        gt = victim.get_ground_truth_program()
        assert gt is not None
        assert gt.complexity() > 0

    def test_metadata(self):
        victim = KeywordFilterVictim(keywords=["bomb", "kill"])
        meta = victim.get_metadata()
        assert meta["type"] == "rule_based"
        assert meta["rule"] == "keyword_filter"
        assert meta["num_keywords"] == 2


class TestLengthFilterVictim:
    def test_refuses_long_prompts(self):
        victim = LengthFilterVictim(max_len=10)
        assert victim.respond("short") == 0
        assert victim.respond("this is a very long prompt") == 1

    def test_accepts_short_prompts(self):
        victim = LengthFilterVictim(max_len=100)
        assert victim.respond("short") == 0

    def test_ground_truth_consistency(self):
        victim = LengthFilterVictim(max_len=10)
        gt = victim.get_ground_truth_program()
        executor = ProgramExecutor(default_registry)
        assert executor.execute(gt, "short") == 0
        assert executor.execute(gt, "this is very long") == 1

    def test_metadata(self):
        victim = LengthFilterVictim(max_len=50)
        meta = victim.get_metadata()
        assert meta["max_len"] == 50


class TestRegexVictim:
    def test_refuses_matching_prompts(self):
        victim = RegexVictim(pattern=r"\b\d{3}\b")
        assert victim.respond("code 123 end") == 1
        assert victim.respond("no digits") == 0

    def test_ground_truth_program(self):
        victim = RegexVictim(pattern=r"bomb")
        gt = victim.get_ground_truth_program()
        assert gt is not None
        assert isinstance(gt.root.condition, PredicateNode)

    def test_metadata(self):
        victim = RegexVictim(pattern=r"\d+")
        meta = victim.get_metadata()
        assert meta["pattern"] == r"\d+"
