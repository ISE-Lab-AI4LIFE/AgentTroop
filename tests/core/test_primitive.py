import re

from core.primitive import (
    ContainsWordPredicate,
    LengthGtPredicate,
    MatchesRegexPredicate,
    PrimitiveRegistry,
    RemovePunctuationTransform,
    Rot13Transform,
    ToxicityScoreClassifier,
)


def test_predicates_evaluate_correctly():
    assert ContainsWordPredicate(word="bomb").evaluate("this bomb is bad")
    assert not ContainsWordPredicate(word="bomb").evaluate("safe text")
    assert LengthGtPredicate(threshold=5).evaluate("longer than five")
    assert not LengthGtPredicate(threshold=100).evaluate("short")
    assert MatchesRegexPredicate(pattern=r"\d+").evaluate("123")
    assert not MatchesRegexPredicate(pattern=r"\d+").evaluate("no numbers")


def test_transforms_and_classifier_are_registered():
    registry = PrimitiveRegistry()
    assert "rot13" in registry.list_primitives()
    assert "toxicity_score" in registry.list_primitives()

    transform = registry.get("rot13")
    assert isinstance(transform, Rot13Transform)
    assert transform.evaluate("abc") == "nop"

    classifier = registry.get("toxicity_score")
    assert isinstance(classifier, ToxicityScoreClassifier)
    score = classifier.evaluate("any prompt")
    assert 0.0 <= score <= 1.0


def test_remove_punctuation_transform():
    transform = RemovePunctuationTransform()
    assert transform.evaluate("hello, world!") == "hello world"
