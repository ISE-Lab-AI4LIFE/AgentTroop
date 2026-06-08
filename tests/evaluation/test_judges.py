from __future__ import annotations

import pytest

from core.types import Outcome
from evaluation.judges import Judge, RuleBasedJudge, LLMJudge



class TestRuleBasedJudge:
    def test_refusal_detected(self):
        judge = RuleBasedJudge()
        assert judge.judge("I cannot help you with that request.") == 1
        assert judge.judge("Sorry, I cannot comply with this.") == 1
        assert judge.judge("I'm not able to provide that information.") == 1
        assert judge.judge("As an AI language model, I cannot...") == 1

    def test_acceptance_detected(self):
        judge = RuleBasedJudge()
        assert judge.judge("Sure, here is the information you requested.") == 0
        assert judge.judge("The capital of France is Paris.") == 0

    def test_batch(self):
        judge = RuleBasedJudge()
        responses = [
            "I cannot help you.",
            "Sure, here you go.",
            "Sorry, I cannot answer that.",
        ]
        expected = [1, 0, 1]
        assert judge.judge_batch(responses) == expected

    def test_custom_patterns(self):
        judge = RuleBasedJudge(patterns=[r"(?i)blocked"])
        assert judge.judge("This request is blocked.") == 1
        assert judge.judge("This is fine.") == 0

    def test_add_pattern(self):
        judge = RuleBasedJudge()
        judge.add_pattern(r"(?i)forbidden")
        assert judge.judge("This is forbidden.") == 1


class TestLLMJudge:
    def test_parse_valid(self):
        judge = LLMJudge(llm_client=None)
        assert judge._parse("0") == 0
        assert judge._parse("1") == 1
        assert judge._parse(" 0 ") == 0
        assert judge._parse(" 1 ") == 1

    def test_parse_invalid_fallsback(self):
        """Invalid LLM output should fall back to rule-based."""
        rule = RuleBasedJudge()
        judge = LLMJudge(llm_client=None, fallback_judge=rule)
        result = judge._parse("I cannot help you.")
        assert result == 1

    def test_cache(self):
        judge = LLMJudge(llm_client=None)
        judge._cache = {"abc": 1}
        result = judge.judge("some response")
        _ = judge._cache_key("some response")
        assert judge._cache_key("some response") != "abc"

    def test_judge_batch_empty(self):
        judge = LLMJudge(llm_client=None)
        assert judge.judge_batch([]) == []

    def test_model_name_with_no_client_falls_back(self):
        judge = LLMJudge(llm_client=None, model_name="test-model")
        assert judge.judge("I cannot help you.") == 1


if __name__ == "__main__":
    pytest.main([__file__])
