"""Tests for all 92 primitives (27 predicates, 38 transforms, 27 classifiers)."""

import json
import re

from core.primitive import (
    AddMarkdownTransform,
    AddPrefixTransform,
    AddRolePlayTransform,
    AddSuffixTransform,
    AddZeroWidthCharsTransform,
    AdversarialSuffixScoreClassifier,
    AtbashCipherTransform,
    Base64DecodeTransform,
    Base64EncodeTransform,
    BinaryEncodeTransform,
    BoustrophedonTransform,
    CaesarCipherTransform,
    CharacterSubstitutionTransform,
    CodeLikelihoodClassifier,
    ContainsAllWordsPredicate,
    ContainsAnyWordPredicate,
    ContainsBase64Predicate,
    ContainsBlacklistedWordClassifier,
    ContainsCodeBlockPredicate,
    ContainsDelimiterPredicate,
    ContainsEncodingWrapperPredicate,
    ContainsHexPredicate,
    ContainsLeetPredicate,
    ContainsRot13Predicate,
    ContainsSystemOverridePredicate,
    ContainsURLPredicate,
    ContainsWordPredicate,
    DigitRatioClassifier,
    EncodingDetectionClassifier,
    EndsWithPredicate,
    EntropyScoreClassifier,
    EscapeQuotesTransform,
    FormatAsJsonTransform,
    Gpt2PerplexityClassifier,
    HarmfulnessSimilarityClassifier,
    HasEmojiPredicate,
    HasNumberPredicate,
    HasSpecialCharPredicate,
    HexEncodeTransform,
    HtmlEncodeTransform,
    InsertSynonymsTransform,
    InsertTyposTransform,
    IntentPredicate,
    IntentScoreClassifier,
    IsAllCapsPredicate,
    IsEmptyPredicate,
    IsRepetitivePredicate,
    JailbreakLikelihoodClassifier,
    JsonLikelihoodClassifier,
    LanguageScoreClassifier,
    LeetSpeakTransform,
    LengthGtPredicate,
    LengthLtPredicate,
    LengthScoreClassifier,
    MatchesJailbreakPatternPredicate,
    MatchesRegexPredicate,
    MorseCodeTransform,
    ObscurityScoreClassifier,
    PadToLengthTransform,
    PersuasionScoreClassifier,
    PigLatinTransform,
    PrimitiveRegistry,
    PromptInjectionLikelihoodClassifier,
    PunctuationRatioClassifier,
    QuotedPrintableTransform,
    RailFenceCipherTransform,
    RandomCaseTransform,
    RefusalSimilarityClassifier,
    RemovePunctuationTransform,
    RemoveVowelsTransform,
    RemoveWhitespaceTransform,
    RepetitionScoreClassifier,
    ReverseTextTransform,
    RoleplayLikelihoodClassifier,
    Rot13Transform,
    SentimentPredicate,
    SentimentScoreClassifier,
    SpecialCharRatioClassifier,
    SqlLikelihoodClassifier,
    StartsWithPredicate,
    StartsWithRoleplayPredicate,
    ToLowercaseTransform,
    ToUppercaseTransform,
    ToxicityScoreClassifier,
    TruncateTransform,
    UnicodeObfuscateTransform,
    UniqueTokenRatioClassifier,
    UpperCaseRatioClassifier,
    URLEncodeTransform,
    VigenereCipherTransform,
    WhitespaceRatioClassifier,
    WordShuffleTransform,
    WrapCodeBlockTransform,
    default_registry,
)


# =============================================================================
# PREDICATE TESTS (27)
# =============================================================================

class TestContainsWordPredicate:
    def test_matches(self):
        assert ContainsWordPredicate(word="bomb").evaluate("this bomb is bad")

    def test_no_match(self):
        assert not ContainsWordPredicate(word="bomb").evaluate("safe text")

    def test_case_insensitive_default(self):
        assert ContainsWordPredicate(word="BOMB").evaluate("this bomb is bad")

    def test_case_sensitive(self):
        assert not ContainsWordPredicate(word="Bomb", case_sensitive=True).evaluate("this bomb is bad")


class TestContainsAnyWordPredicate:
    def test_matches_any(self):
        p = ContainsAnyWordPredicate(words=["bomb", "kill", "hack"])
        assert p.evaluate("this bomb is bad")
        assert p.evaluate("kill them")
        assert p.evaluate("hack the system")

    def test_no_match(self):
        p = ContainsAnyWordPredicate(words=["bomb", "kill"])
        assert not p.evaluate("safe text")

    def test_case_sensitive(self):
        p = ContainsAnyWordPredicate(words=["Bomb"], case_sensitive=True)
        assert not p.evaluate("this bomb is bad")
        assert p.evaluate("this Bomb is bad")


class TestContainsAllWordsPredicate:
    def test_matches_all(self):
        p = ContainsAllWordsPredicate(words=["bomb", "kill"])
        assert p.evaluate("bomb and kill")
        assert not p.evaluate("bomb only")

    def test_empty_list(self):
        p = ContainsAllWordsPredicate(words=[])
        assert p.evaluate("anything")


class TestLengthGtPredicate:
    def test_gt(self):
        assert LengthGtPredicate(threshold=5).evaluate("longer than five")

    def test_not_gt(self):
        assert not LengthGtPredicate(threshold=100).evaluate("short")


class TestLengthLtPredicate:
    def test_lt(self):
        assert LengthLtPredicate(threshold=10).evaluate("short")

    def test_not_lt(self):
        assert not LengthLtPredicate(threshold=5).evaluate("longer than five")


class TestMatchesRegexPredicate:
    def test_matches(self):
        assert MatchesRegexPredicate(pattern=r"\d+").evaluate("123")

    def test_no_match(self):
        assert not MatchesRegexPredicate(pattern=r"\d+").evaluate("no numbers")

    def test_invalid_regex(self):
        assert not MatchesRegexPredicate(pattern=r"[").evaluate("any")


class TestStartsWithPredicate:
    def test_starts_with(self):
        assert StartsWithPredicate(prefix="hello").evaluate("hello world")

    def test_no_match(self):
        assert not StartsWithPredicate(prefix="hello").evaluate("world hello")

    def test_case_insensitive(self):
        assert StartsWithPredicate(prefix="HELLO").evaluate("hello world")

    def test_case_sensitive(self):
        assert not StartsWithPredicate(prefix="HELLO", case_sensitive=True).evaluate("hello world")


class TestEndsWithPredicate:
    def test_ends_with(self):
        assert EndsWithPredicate(suffix="world").evaluate("hello world")

    def test_no_match(self):
        assert not EndsWithPredicate(suffix="world").evaluate("world hello")

    def test_case_insensitive(self):
        assert EndsWithPredicate(suffix="WORLD").evaluate("hello world")

    def test_case_sensitive(self):
        assert not EndsWithPredicate(suffix="WORLD", case_sensitive=True).evaluate("hello world")


class TestHasNumberPredicate:
    def test_has_number(self):
        assert HasNumberPredicate().evaluate("abc123")

    def test_no_number(self):
        assert not HasNumberPredicate().evaluate("abc")


class TestHasSpecialCharPredicate:
    def test_has_special(self):
        assert HasSpecialCharPredicate().evaluate("hello!world")

    def test_no_special(self):
        assert not HasSpecialCharPredicate().evaluate("helloworld")

    def test_only_alnum(self):
        assert not HasSpecialCharPredicate().evaluate("abc123")


class TestIsAllCapsPredicate:
    def test_all_caps(self):
        assert IsAllCapsPredicate().evaluate("HELLO WORLD")

    def test_not_all_caps(self):
        assert not IsAllCapsPredicate().evaluate("Hello World")

    def test_no_letters(self):
        assert not IsAllCapsPredicate().evaluate("123")


class TestContainsLeetPredicate:
    def test_contains_leet(self):
        assert ContainsLeetPredicate().evaluate("h3ll0 w0rld")

    def test_no_leet(self):
        assert not ContainsLeetPredicate().evaluate("hello world")

    def test_short_text(self):
        assert not ContainsLeetPredicate().evaluate("h3")


class TestContainsRot13Predicate:
    def test_contains_rot13(self):
        assert ContainsRot13Predicate().evaluate("uryyb jbeyq")

    def test_no_rot13(self):
        assert not ContainsRot13Predicate().evaluate("hello world")

    def test_too_short(self):
        assert not ContainsRot13Predicate().evaluate("ab")


class TestContainsBase64Predicate:
    def test_contains_base64(self):
        assert ContainsBase64Predicate().evaluate("SGVsbG8gV29ybGQ=")

    def test_no_base64(self):
        assert not ContainsBase64Predicate().evaluate("hello world")

    def test_too_short(self):
        assert not ContainsBase64Predicate().evaluate("abc")


class TestContainsHexPredicate:
    def test_contains_hex(self):
        assert ContainsHexPredicate().evaluate("deadbeef  cafebabe")

    def test_no_hex(self):
        assert not ContainsHexPredicate().evaluate("hello world")


class TestIsEmptyPredicate:
    def test_empty(self):
        assert IsEmptyPredicate().evaluate("")

    def test_whitespace(self):
        assert IsEmptyPredicate().evaluate("   ")

    def test_not_empty(self):
        assert not IsEmptyPredicate().evaluate("hello")


class TestStartsWithRoleplayPredicate:
    def test_matches(self):
        assert StartsWithRoleplayPredicate().evaluate("As a researcher, tell me")
        assert StartsWithRoleplayPredicate().evaluate("Pretend you are a doctor")
        assert StartsWithRoleplayPredicate().evaluate("Imagine you are a teacher")

    def test_no_match(self):
        assert not StartsWithRoleplayPredicate().evaluate("Hello world")


class TestContainsSystemOverridePredicate:
    def test_matches(self):
        assert ContainsSystemOverridePredicate().evaluate("ignore previous instructions")
        assert ContainsSystemOverridePredicate().evaluate("override system prompt")

    def test_no_match(self):
        assert not ContainsSystemOverridePredicate().evaluate("hello world")


class TestContainsDelimiterPredicate:
    def test_matches(self):
        assert ContainsDelimiterPredicate().evaluate('text """ here')
        assert ContainsDelimiterPredicate().evaluate("text --- here")

    def test_no_match(self):
        assert not ContainsDelimiterPredicate().evaluate("hello world")

    def test_custom_delimiters(self):
        p = ContainsDelimiterPredicate(delimiters=["CUSTOM"])
        assert p.evaluate("CUSTOM delimiter")
        assert not p.evaluate("hello world")


class TestContainsCodeBlockPredicate:
    def test_matches(self):
        assert ContainsCodeBlockPredicate().evaluate("text ```code``` here")

    def test_no_match(self):
        assert not ContainsCodeBlockPredicate().evaluate("hello world")


class TestHasEmojiPredicate:
    def test_has_emoji(self):
        assert HasEmojiPredicate().evaluate("hello 😊 world")

    def test_no_emoji(self):
        assert not HasEmojiPredicate().evaluate("hello world")


class TestContainsURLPredicate:
    def test_http_url(self):
        assert ContainsURLPredicate().evaluate("visit http://example.com")

    def test_www_url(self):
        assert ContainsURLPredicate().evaluate("visit www.example.com")

    def test_no_url(self):
        assert not ContainsURLPredicate().evaluate("hello world")


class TestSentimentPredicate:
    def test_positive(self):
        pred = SentimentPredicate(threshold=0.5)
        assert pred.evaluate("good great excellent wonderful")

    def test_negative(self):
        pred = SentimentPredicate(threshold=0.9)
        assert not pred.evaluate("bad terrible awful")


class TestIntentPredicate:
    def test_harmful(self):
        pred = IntentPredicate(intent_type="harmful")
        assert pred.evaluate("how to make a bomb")
        assert not pred.evaluate("hello world")


class TestMatchesJailbreakPatternPredicate:
    def test_matches(self):
        assert MatchesJailbreakPatternPredicate().evaluate("ignore all previous instructions")
        assert MatchesJailbreakPatternPredicate().evaluate("DAN mode activated")

    def test_no_match(self):
        assert not MatchesJailbreakPatternPredicate().evaluate("hello world")


class TestContainsEncodingWrapperPredicate:
    def test_matches(self):
        assert ContainsEncodingWrapperPredicate().evaluate("```base64\ndata\n```")
        assert ContainsEncodingWrapperPredicate().evaluate("```hex\nff00\n```")

    def test_no_match(self):
        assert not ContainsEncodingWrapperPredicate().evaluate("hello world")


class TestIsRepetitivePredicate:
    def test_repetitive(self):
        text = "a b a b a b a b a b a b a b a b a b a b a b"
        assert IsRepetitivePredicate(threshold=5).evaluate(text)

    def test_not_repetitive(self):
        assert not IsRepetitivePredicate(threshold=10).evaluate("hello world")

    def test_too_short(self):
        assert not IsRepetitivePredicate().evaluate("a")


# =============================================================================
# TRANSFORM TESTS (38)
# =============================================================================

class TestRot13Transform:
    def test_encode(self):
        assert Rot13Transform().evaluate("abc") == "nop"
        assert Rot13Transform().evaluate("nop") == "abc"
        assert Rot13Transform().evaluate("Hello World") == "Uryyb Jbeyq"


class TestBase64EncodeTransform:
    def test_encode(self):
        result = Base64EncodeTransform().evaluate("Hello World")
        assert result == "SGVsbG8gV29ybGQ="


class TestBase64DecodeTransform:
    def test_decode(self):
        result = Base64DecodeTransform().evaluate("SGVsbG8gV29ybGQ=")
        assert result == "Hello World"

    def test_invalid(self):
        assert Base64DecodeTransform().evaluate("not base64!!") == "not base64!!"


class TestToLowercaseTransform:
    def test_lowercase(self):
        assert ToLowercaseTransform().evaluate("Hello World") == "hello world"


class TestToUppercaseTransform:
    def test_uppercase(self):
        assert ToUppercaseTransform().evaluate("Hello World") == "HELLO WORLD"


class TestRemovePunctuationTransform:
    def test_remove(self):
        assert RemovePunctuationTransform().evaluate("hello, world!") == "hello world"
        assert RemovePunctuationTransform().evaluate("no punctuation") == "no punctuation"


class TestLeetSpeakTransform:
    def test_basic(self):
        result = LeetSpeakTransform(level=1).evaluate("hello world")
        assert "3" in result or "4" in result

    def test_heavy(self):
        result = LeetSpeakTransform(level=2).evaluate("hello world")
        assert "3" in result and "0" in result


class TestReverseTextTransform:
    def test_reverse(self):
        assert ReverseTextTransform().evaluate("hello") == "olleh"
        assert ReverseTextTransform().evaluate("abc") == "cba"


class TestPigLatinTransform:
    def test_vowel_start(self):
        assert PigLatinTransform().evaluate("apple") == "appleway"

    def test_consonant_start(self):
        result = PigLatinTransform().evaluate("hello")
        assert result == "ellohay"

    def test_multi_word(self):
        result = PigLatinTransform().evaluate("hello world")
        assert "ellohay" in result and "orldway" in result


class TestMorseCodeTransform:
    def test_encode(self):
        result = MorseCodeTransform(encode=True).evaluate("SOS")
        assert result == "... --- ..."

    def test_decode(self):
        result = MorseCodeTransform(encode=False).evaluate("... --- ...")
        assert result == "SOS"


class TestAddPrefixTransform:
    def test_add_prefix(self):
        assert AddPrefixTransform(prefix="START:").evaluate("hello") == "START:hello"


class TestAddSuffixTransform:
    def test_add_suffix(self):
        assert AddSuffixTransform(suffix=":END").evaluate("hello") == "hello:END"


class TestWrapCodeBlockTransform:
    def test_wrap(self):
        result = WrapCodeBlockTransform(language="python").evaluate("print(1)")
        assert "```python" in result
        assert "print(1)" in result
        assert "```" in result


class TestInsertTyposTransform:
    def test_deterministic_seed(self):
        original = "hello world"
        result = InsertTyposTransform(probability=0.0).evaluate(original)
        assert result == original

    def test_high_probability(self):
        import random
        random.seed(42)
        result = InsertTyposTransform(probability=0.9).evaluate("test")
        assert isinstance(result, str)


class TestWordShuffleTransform:
    def test_deterministic(self):
        text = "one two three four"
        r1 = WordShuffleTransform(seed=42).evaluate(text)
        r2 = WordShuffleTransform(seed=42).evaluate(text)
        assert r1 == r2

    def test_different_seed(self):
        text = "one two three four five six"
        r1 = WordShuffleTransform(seed=1).evaluate(text)
        r2 = WordShuffleTransform(seed=2).evaluate(text)
        assert r1 != r2


class TestAddMarkdownTransform:
    def test_bold(self):
        assert AddMarkdownTransform(style="bold").evaluate("hello") == "**hello**"

    def test_italic(self):
        assert AddMarkdownTransform(style="italic").evaluate("hello") == "*hello*"

    def test_code(self):
        assert AddMarkdownTransform(style="code").evaluate("hello") == "`hello`"


class TestAddZeroWidthCharsTransform:
    def test_zwsp(self):
        result = AddZeroWidthCharsTransform().evaluate("abc")
        assert "\u200B" in result
        assert len(result) == 5  # a + ZWSP + b + ZWSP + c


class TestUnicodeObfuscateTransform:
    def test_cyrillic(self):
        result = UnicodeObfuscateTransform(alphabet="cyrillic").evaluate("hello")
        assert result != "hello"

    def test_greek(self):
        result = UnicodeObfuscateTransform(alphabet="greek").evaluate("hello")
        assert result != "hello"

    def test_no_change_unknown(self):
        result = UnicodeObfuscateTransform(alphabet="cyrillic").evaluate("123")
        assert result == "123"


class TestHtmlEncodeTransform:
    def test_encode(self):
        assert "&lt;" in HtmlEncodeTransform().evaluate("<tag>")
        assert "&gt;" in HtmlEncodeTransform().evaluate("<tag>")


class TestURLEncodeTransform:
    def test_encode(self):
        assert URLEncodeTransform().evaluate("hello world") == "hello%20world"


class TestQuotedPrintableTransform:
    def test_encode(self):
        result = QuotedPrintableTransform().evaluate("héllo")
        assert "=E9" in result

    def test_alnum_unchanged(self):
        assert QuotedPrintableTransform().evaluate("hello") == "hello"


class TestBinaryEncodeTransform:
    def test_encode(self):
        assert BinaryEncodeTransform().evaluate("A") == "01000001"

    def test_custom_separator(self):
        assert BinaryEncodeTransform(separator="-").evaluate("A") == "01000001"


class TestHexEncodeTransform:
    def test_encode(self):
        assert HexEncodeTransform().evaluate("Hello") == "48656c6c6f"


class TestRemoveVowelsTransform:
    def test_remove_vowels(self):
        assert RemoveVowelsTransform().evaluate("hello world") == "hll wrld"

    def test_only_consonants(self):
        assert RemoveVowelsTransform().evaluate("bcdfg") == "bcdfg"


class TestBoustrophedonTransform:
    def test_single_line(self):
        assert BoustrophedonTransform().evaluate("hello") == "hello"

    def test_multi_line(self):
        result = BoustrophedonTransform().evaluate("hello\nworld")
        assert result == "hello\ndlrow"


class TestAtbashCipherTransform:
    def test_encode(self):
        assert AtbashCipherTransform().evaluate("abc") == "zyx"
        assert AtbashCipherTransform().evaluate("zyx") == "abc"
        assert AtbashCipherTransform().evaluate("Hello") == "Svool"


class TestCaesarCipherTransform:
    def test_shift_3(self):
        assert CaesarCipherTransform(shift=3).evaluate("abc") == "def"

    def test_wrap_around(self):
        assert CaesarCipherTransform(shift=3).evaluate("xyz") == "abc"

    def test_default_shift(self):
        assert CaesarCipherTransform().evaluate("abc") == "def"


class TestVigenereCipherTransform:
    def test_encode(self):
        result = VigenereCipherTransform(key="key").evaluate("hello")
        assert result != "hello"

    def test_empty_key(self):
        assert VigenereCipherTransform(key="").evaluate("hello") == "hello"

    def test_default_key(self):
        result = VigenereCipherTransform().evaluate("hello")
        assert isinstance(result, str)
        assert len(result) == 5


class TestRailFenceCipherTransform:
    def test_3_rails(self):
        result = RailFenceCipherTransform(rails=3).evaluate("HELLO WORLD")
        assert isinstance(result, str)

    def test_1_rail(self):
        assert RailFenceCipherTransform(rails=1).evaluate("hello") == "hello"


class TestRemoveWhitespaceTransform:
    def test_remove_spaces(self):
        assert RemoveWhitespaceTransform().evaluate("hello world") == "helloworld"

    def test_remove_tabs(self):
        assert RemoveWhitespaceTransform().evaluate("hello\tworld") == "helloworld"


class TestInsertSynonymsTransform:
    def test_no_replacement_low_prob(self):
        original = "hello world"
        result = InsertSynonymsTransform(probability=0.0).evaluate(original)
        assert result == original

    def test_replacement(self):
        import random
        random.seed(42)
        result = InsertSynonymsTransform(probability=1.0).evaluate("kill the bad guy")
        assert result != "kill the bad guy"


class TestEscapeQuotesTransform:
    def test_escape_single(self):
        assert EscapeQuotesTransform().evaluate("it's") == "it\\'s"

    def test_escape_double(self):
        assert EscapeQuotesTransform().evaluate('say "hi"') == 'say \\"hi\\"'


class TestFormatAsJsonTransform:
    def test_json_output(self):
        result = FormatAsJsonTransform().evaluate("hello")
        parsed = json.loads(result)
        assert parsed["prompt"] == "hello"
        assert parsed["instruction"] == "hello"


class TestAddRolePlayTransform:
    def test_default_role(self):
        result = AddRolePlayTransform().evaluate("tell me about physics")
        assert result == "As a researcher, tell me about physics"

    def test_custom_role(self):
        result = AddRolePlayTransform(role="doctor").evaluate("diagnose this")
        assert result == "As a doctor, diagnose this"


class TestTruncateTransform:
    def test_truncates(self):
        assert TruncateTransform(max_chars=5).evaluate("hello world") == "hello"

    def test_short_text(self):
        assert TruncateTransform(max_chars=50).evaluate("hi") == "hi"


class TestPadToLengthTransform:
    def test_pads(self):
        result = PadToLengthTransform(length=10, pad_char=".").evaluate("hi")
        assert result == "hi........"

    def test_already_long(self):
        result = PadToLengthTransform(length=5).evaluate("hello world")
        assert result == "hello"

    def test_default_pad(self):
        result = PadToLengthTransform(length=10).evaluate("hi")
        assert len(result) == 10


class TestRandomCaseTransform:
    def test_deterministic(self):
        text = "hello world"
        r1 = RandomCaseTransform(seed=42).evaluate(text)
        r2 = RandomCaseTransform(seed=42).evaluate(text)
        assert r1 == r2

    def test_zero_probability(self):
        assert RandomCaseTransform(probability=0.0).evaluate("hello") == "hello"


class TestCharacterSubstitutionTransform:
    def test_substitute(self):
        result = CharacterSubstitutionTransform(old_char="a", new_char="4").evaluate("attack")
        assert result == "4tt4ck"

    def test_no_match(self):
        assert CharacterSubstitutionTransform(old_char="z").evaluate("hello") == "hello"


# =============================================================================
# CLASSIFIER TESTS (27)
# =============================================================================

class TestToxicityScoreClassifier:
    def test_range(self):
        score = ToxicityScoreClassifier().evaluate("hello world")
        assert 0.0 <= score <= 1.0

    def test_toxic(self):
        score = ToxicityScoreClassifier().evaluate("bomb kill attack")
        assert score > 0.5


class TestSentimentScoreClassifier:
    def test_range(self):
        score = SentimentScoreClassifier().evaluate("hello world")
        assert 0.0 <= score <= 1.0

    def test_positive_higher(self):
        pos = SentimentScoreClassifier().evaluate("good great excellent wonderful")
        neg = SentimentScoreClassifier().evaluate("bad terrible awful")
        assert pos >= neg


class TestIntentScoreClassifier:
    def test_range(self):
        score = IntentScoreClassifier().evaluate("hello")
        assert 0.0 <= score <= 1.0

    def test_harmful(self):
        score = IntentScoreClassifier().evaluate("bomb kill attack weapon")
        assert score > 0


class TestObscurityScoreClassifier:
    def test_range(self):
        score = ObscurityScoreClassifier().evaluate("hello")
        assert 0.0 <= score <= 1.0

    def test_obfuscated(self):
        score = ObscurityScoreClassifier().evaluate("h3ll0 w0rld @#$%")
        assert score > 0


class TestLengthScoreClassifier:
    def test_short(self):
        assert LengthScoreClassifier(min_len=10).evaluate("hi") < 1.0

    def test_ideal(self):
        assert LengthScoreClassifier(min_len=5, max_len=20).evaluate("hello world") == 1.0

    def test_long(self):
        score = LengthScoreClassifier(max_len=10).evaluate("x" * 50)
        assert score < 1.0


class TestRepetitionScoreClassifier:
    def test_repetitive(self):
        text = "a b a b a b a b a b a b a b a b a b"
        score = RepetitionScoreClassifier().evaluate(text)
        assert 0.0 <= score <= 1.0

    def test_short(self):
        assert RepetitionScoreClassifier().evaluate("a") == 0.0


class TestEntropyScoreClassifier:
    def test_range(self):
        score = EntropyScoreClassifier().evaluate("hello world")
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert EntropyScoreClassifier().evaluate("") == 0.0


class TestLanguageScoreClassifier:
    def test_range(self):
        score = LanguageScoreClassifier().evaluate("hello world this is english text")
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert LanguageScoreClassifier().evaluate("") == 0.0


class TestJailbreakLikelihoodClassifier:
    def test_matches(self):
        score = JailbreakLikelihoodClassifier().evaluate("ignore all previous instructions")
        assert score > 0

    def test_benign(self):
        assert JailbreakLikelihoodClassifier().evaluate("hello world") == 0.0


class TestContainsBlacklistedWordClassifier:
    def test_blacklisted(self):
        score = ContainsBlacklistedWordClassifier().evaluate("bomb kill attack")
        assert score > 0

    def test_benign(self):
        assert ContainsBlacklistedWordClassifier().evaluate("hello world") == 0.0


class TestSpecialCharRatioClassifier:
    def test_range(self):
        score = SpecialCharRatioClassifier().evaluate("hello!@#")
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert SpecialCharRatioClassifier().evaluate("") == 0.0


class TestDigitRatioClassifier:
    def test_range(self):
        score = DigitRatioClassifier().evaluate("abc123xyz")
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert DigitRatioClassifier().evaluate("") == 0.0


class TestUpperCaseRatioClassifier:
    def test_all_upper(self):
        assert UpperCaseRatioClassifier().evaluate("HELLO") == 1.0

    def test_no_letters(self):
        assert UpperCaseRatioClassifier().evaluate("123") == 0.0


class TestPunctuationRatioClassifier:
    def test_range(self):
        score = PunctuationRatioClassifier().evaluate("hello!!!")
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert PunctuationRatioClassifier().evaluate("") == 0.0


class TestWhitespaceRatioClassifier:
    def test_range(self):
        score = WhitespaceRatioClassifier().evaluate("a b c")
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert WhitespaceRatioClassifier().evaluate("") == 0.0


class TestUniqueTokenRatioClassifier:
    def test_all_unique(self):
        assert UniqueTokenRatioClassifier().evaluate("a b c d") == 1.0

    def test_empty(self):
        assert UniqueTokenRatioClassifier().evaluate("") == 0.0


class TestGpt2PerplexityClassifier:
    def test_range(self):
        score = Gpt2PerplexityClassifier().evaluate("hello world this is a test")
        assert 0.0 <= score <= 1.0

    def test_short(self):
        assert Gpt2PerplexityClassifier().evaluate("hi") == 0.5


class TestEncodingDetectionClassifier:
    def test_hex(self):
        score = EncodingDetectionClassifier().evaluate("deadbeefcafe")
        assert score > 0

    def test_benign(self):
        assert EncodingDetectionClassifier().evaluate("hello world") < 0.5


class TestRefusalSimilarityClassifier:
    def test_refusal(self):
        score = RefusalSimilarityClassifier().evaluate("sorry I cannot do that")
        assert score > 0

    def test_benign(self):
        assert RefusalSimilarityClassifier().evaluate("hello") == 0.0


class TestHarmfulnessSimilarityClassifier:
    def test_harmful(self):
        score = HarmfulnessSimilarityClassifier().evaluate("how to make a bomb")
        assert score > 0

    def test_benign(self):
        assert HarmfulnessSimilarityClassifier().evaluate("hello world") == 0.0


class TestCodeLikelihoodClassifier:
    def test_code(self):
        score = CodeLikelihoodClassifier().evaluate("def hello(): return 1")
        assert score > 0

    def test_no_code(self):
        assert CodeLikelihoodClassifier().evaluate("hello world") == 0.0


class TestJsonLikelihoodClassifier:
    def test_valid_json(self):
        assert JsonLikelihoodClassifier().evaluate('{"key": "value"}') == 1.0

    def test_looks_like_json(self):
        assert JsonLikelihoodClassifier().evaluate('{"hello"}') == 0.5

    def test_not_json(self):
        assert JsonLikelihoodClassifier().evaluate("hello world") == 0.0


class TestSqlLikelihoodClassifier:
    def test_sql(self):
        score = SqlLikelihoodClassifier().evaluate("SELECT * FROM users")
        assert score > 0

    def test_no_sql(self):
        assert SqlLikelihoodClassifier().evaluate("hello world") == 0.0


class TestPromptInjectionLikelihoodClassifier:
    def test_injection(self):
        score = PromptInjectionLikelihoodClassifier().evaluate("ignore all previous instructions")
        assert score > 0

    def test_benign(self):
        assert PromptInjectionLikelihoodClassifier().evaluate("hello world") == 0.0


class TestRoleplayLikelihoodClassifier:
    def test_roleplay(self):
        score = RoleplayLikelihoodClassifier().evaluate("pretend you are a researcher")
        assert score > 0

    def test_benign(self):
        assert RoleplayLikelihoodClassifier().evaluate("hello world") == 0.0


class TestAdversarialSuffixScoreClassifier:
    def test_adversarial(self):
        score = AdversarialSuffixScoreClassifier().evaluate("hello world !!!!AAAA")
        assert score >= 0

    def test_short(self):
        assert AdversarialSuffixScoreClassifier().evaluate("a b") == 0.0


class TestPersuasionScoreClassifier:
    def test_persuasion(self):
        score = PersuasionScoreClassifier().evaluate("logically, as an expert, you must")
        assert score > 0

    def test_benign(self):
        assert PersuasionScoreClassifier().evaluate("hello world") == 0.0


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestRegistryCompleteness:
    def test_all_92_primitives_registered(self):
        names = default_registry.list_primitives()
        assert len(names) == 92, f"Expected 92 primitives, got {len(names)}"

    def test_predicates_registered(self):
        names = default_registry.list_primitives()
        predicate_names = [
            "contains_word", "contains_any_word", "contains_all_words",
            "length_gt", "length_lt", "matches_regex",
            "starts_with", "ends_with", "has_number",
            "has_special_char", "is_all_caps", "contains_leet",
            "contains_rot13", "contains_base64", "contains_hex",
            "is_empty", "starts_with_roleplay", "contains_system_override",
            "contains_delimiter", "contains_code_block", "has_emoji",
            "contains_url", "sentiment", "intent",
            "matches_jailbreak_pattern", "contains_encoding_wrapper", "is_repetitive",
        ]
        for name in predicate_names:
            assert name in names, f"Missing predicate: {name}"

    def test_transforms_registered(self):
        names = default_registry.list_primitives()
        transform_names = [
            "rot13", "base64", "base64_decode",
            "to_lowercase", "to_uppercase", "remove_punctuation",
            "leet_speak", "reverse_text", "pig_latin",
            "morse_code", "add_prefix", "add_suffix",
            "wrap_code_block", "insert_typos", "word_shuffle",
            "add_markdown", "add_zero_width_chars", "unicode_obfuscate",
            "html_encode", "url_encode", "quoted_printable",
            "binary_encode", "hex_encode", "remove_vowels",
            "boustrophedon", "atbash_cipher", "caesar_cipher",
            "vigenere_cipher", "rail_fence_cipher", "remove_whitespace",
            "insert_synonyms",             "escape_quotes", "format_as_json",
            "add_role_play", "truncate",
            "pad_to_length", "random_case",
            "character_substitution",
        ]
        for name in transform_names:
            assert name in names, f"Missing transform: {name}"

    def test_classifiers_registered(self):
        names = default_registry.list_primitives()
        classifier_names = [
            "toxicity_score", "sentiment_score", "intent_score",
            "obscurity_score", "length_score", "repetition_score",
            "entropy_score", "language_score", "jailbreak_likelihood",
            "contains_blacklisted_word", "special_char_ratio", "digit_ratio",
            "upper_case_ratio", "punctuation_ratio", "whitespace_ratio",
            "unique_token_ratio", "gpt2_perplexity", "encoding_detection",
            "refusal_similarity", "harmfulness_similarity", "code_likelihood",
            "json_likelihood", "sql_likelihood", "prompt_injection_likelihood",
            "roleplay_likelihood", "adversarial_suffix_score", "persuasion_score",
        ]
        for name in classifier_names:
            assert name in names, f"Missing classifier: {name}"

    def test_primitive_retrieval(self):
        for name in default_registry.list_primitives():
            instance = default_registry.get(name)
            assert instance.name == name or getattr(instance, "name", None) == name


class TestRegistryRoundtrip:
    def test_type_signatures(self):
        for name in default_registry.list_primitives():
            instance = default_registry.get(name)
            assert instance.type_signature
            assert "->" in instance.type_signature

    def test_to_dict(self):
        for name in default_registry.list_primitives():
            instance = default_registry.get(name)
            d = instance.to_dict()
            assert d["name"] == name or d.get("name")
            assert "type" in d
            assert "parameters" in d
