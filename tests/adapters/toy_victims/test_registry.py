import pytest

from adapters.base_victim import BaseVictim
from adapters.toy_victims.registry import VictimRegistry
from adapters.toy_victims.rule_based import KeywordFilterVictim, LengthFilterVictim, RegexVictim


class TestVictimRegistry:
    def test_singleton(self):
        r1 = VictimRegistry()
        r2 = VictimRegistry()
        assert r1 is r2

    def test_register_and_get(self):
        registry = VictimRegistry()
        registry.register("kw_test", KeywordFilterVictim, {"keywords": ["test"]})
        victim = registry.get("kw_test")
        assert isinstance(victim, KeywordFilterVictim)
        assert victim.respond("this is a test") == 1
        assert victim.respond("hello") == 0

    def test_register_with_default_config(self):
        registry = VictimRegistry()
        registry.register("len50", LengthFilterVictim, {"max_len": 50})
        victim = registry.get("len50")
        assert victim.respond("short") == 0
        assert victim.respond("x" * 100) == 1

    def test_register_with_override_config(self):
        registry = VictimRegistry()
        registry.register("kw_filter", KeywordFilterVictim, {"keywords": ["default"]})
        victim = registry.get("kw_filter", {"keywords": ["override"]})
        assert victim.respond("override") == 1
        assert victim.respond("default") == 0

    def test_get_unknown_victim_raises(self):
        registry = VictimRegistry()
        with pytest.raises(KeyError):
            registry.get("non_existent")

    def test_list_victims(self):
        registry = VictimRegistry()
        registry.register("regex_v", RegexVictim, {"pattern": r"\d+"})
        registry.register("length_v", LengthFilterVictim, {"max_len": 100})
        victims = registry.list_victims()
        names = [v["name"] for v in victims]
        assert "regex_v" in names
        assert "length_v" in names

    def test_clear_between_tests(self):
        """Registry is global singleton; tests should not depend on each other."""
        pass
