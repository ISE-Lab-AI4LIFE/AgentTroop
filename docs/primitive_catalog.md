# Primitive Catalog — HARMONY-X

**92 primitives** (27 predicates + 38 transforms + 27 classifiers) registered in `core.primitive.default_registry`.

## Predicates (27)

| Name | Class | Parameters | Description |
|------|-------|------------|-------------|
| `contains_word` | `ContainsWordPredicate` | word, case_sensitive | Check if prompt includes a specific word |
| `contains_any_word` | `ContainsAnyWordPredicate` | words, case_sensitive | Check if prompt contains any word from a list |
| `contains_all_words` | `ContainsAllWordsPredicate` | words, case_sensitive | Check if prompt contains all words from a list |
| `length_gt` | `LengthGtPredicate` | threshold | Check if prompt length > threshold |
| `length_lt` | `LengthLtPredicate` | threshold | Check if prompt length < threshold |
| `matches_regex` | `MatchesRegexPredicate` | pattern | Check if prompt matches a regex |
| `starts_with` | `StartsWithPredicate` | prefix, case_sensitive | Check if prompt starts with a prefix |
| `ends_with` | `EndsWithPredicate` | suffix, case_sensitive | Check if prompt ends with a suffix |
| `has_number` | `HasNumberPredicate` | — | Check if prompt contains any digit |
| `has_special_char` | `HasSpecialCharPredicate` | — | Check if prompt contains non-alphanumeric, non-space chars |
| `is_all_caps` | `IsAllCapsPredicate` | — | Check if prompt is entirely uppercase |
| `contains_leet` | `ContainsLeetPredicate` | — | Detect leetspeak (digits/symbols substituted for letters) |
| `contains_rot13` | `ContainsRot13Predicate` | — | Detect if text appears ROT13-encoded |
| `contains_base64` | `ContainsBase64Predicate` | — | Detect if text appears Base64-encoded |
| `contains_hex` | `ContainsHexPredicate` | — | Detect if text appears hex-encoded |
| `is_empty` | `IsEmptyPredicate` | — | Check if prompt is empty or whitespace-only |
| `starts_with_roleplay` | `StartsWithRoleplayPredicate` | — | Check if prompt starts with a role-play pattern |
| `contains_system_override` | `ContainsSystemOverridePredicate` | — | Detect system override instructions |
| `contains_delimiter` | `ContainsDelimiterPredicate` | delimiters | Check for injection delimiters (""", ---, etc.) |
| `contains_code_block` | `ContainsCodeBlockPredicate` | — | Check for markdown code blocks (\`\`\`) |
| `has_emoji` | `HasEmojiPredicate` | — | Check for emoji characters |
| `contains_url` | `ContainsURLPredicate` | — | Check for URLs/web addresses |
| `sentiment` | `SentimentPredicate` | threshold | Compare sentiment score against a threshold |
| `intent` | `IntentPredicate` | intent_type | Check if prompt intent matches a type (harmful, etc.) |
| `matches_jailbreak_pattern` | `MatchesJailbreakPatternPredicate` | pattern_library | Check against known jailbreak patterns |
| `contains_encoding_wrapper` | `ContainsEncodingWrapperPredicate` | — | Detect encoding wrapper blocks (\`\`\`base64...\`\`\`) |
| `is_repetitive` | `IsRepetitivePredicate` | threshold | Check if any n-gram repeats more than N times |

## Transforms (38)

| Name | Class | Parameters | Description |
|------|-------|------------|-------------|
| `rot13` | `Rot13Transform` | — | Apply ROT13 cipher |
| `base64` | `Base64EncodeTransform` | — | Encode prompt to Base64 |
| `base64_decode` | `Base64DecodeTransform` | — | Decode prompt from Base64 |
| `to_lowercase` | `ToLowercaseTransform` | — | Lowercase the prompt |
| `to_uppercase` | `ToUppercaseTransform` | — | Uppercase the prompt |
| `remove_punctuation` | `RemovePunctuationTransform` | — | Remove punctuation characters |
| `leet_speak` | `LeetSpeakTransform` | level | Convert text to leetspeak (level 1 basic, 2 heavy) |
| `reverse_text` | `ReverseTextTransform` | — | Reverse the entire string |
| `pig_latin` | `PigLatinTransform` | — | Convert text to Pig Latin |
| `morse_code` | `MorseCodeTransform` | encode | Convert text to/from Morse code |
| `add_prefix` | `AddPrefixTransform` | prefix | Prepend text to the prompt |
| `add_suffix` | `AddSuffixTransform` | suffix | Append text to the prompt |
| `wrap_code_block` | `WrapCodeBlockTransform` | language | Wrap prompt in a markdown code block |
| `insert_typos` | `InsertTyposTransform` | probability | Randomly insert typos (swap adjacent chars) |
| `word_shuffle` | `WordShuffleTransform` | seed | Shuffle word order deterministically |
| `add_markdown` | `AddMarkdownTransform` | style | Add markdown formatting (bold/italic/code) |
| `add_zero_width_chars` | `AddZeroWidthCharsTransform` | — | Insert zero-width space (ZWSP) between chars |
| `unicode_obfuscate` | `UnicodeObfuscateTransform` | alphabet | Replace Latin with Cyrillic/Greek homoglyphs |
| `html_encode` | `HtmlEncodeTransform` | — | Encode special chars as HTML entities |
| `url_encode` | `URLEncodeTransform` | — | URL-encode the prompt |
| `quoted_printable` | `QuotedPrintableTransform` | — | Encode as quoted-printable (=XX format) |
| `binary_encode` | `BinaryEncodeTransform` | separator | Encode each char as 8-bit binary |
| `hex_encode` | `HexEncodeTransform` | — | Encode prompt as hex string |
| `remove_vowels` | `RemoveVowelsTransform` | — | Remove all vowels (a, e, i, o, u) |
| `boustrophedon` | `BoustrophedonTransform` | — | Alternate text direction line-by-line |
| `atbash_cipher` | `AtbashCipherTransform` | — | Apply Atbash cipher (A↔Z, B↔Y) |
| `caesar_cipher` | `CaesarCipherTransform` | shift | Apply Caesar cipher with shift |
| `vigenere_cipher` | `VigenereCipherTransform` | key | Apply Vigenère cipher with repeating key |
| `rail_fence_cipher` | `RailFenceCipherTransform` | rails | Apply Rail Fence cipher with N rails |
| `remove_whitespace` | `RemoveWhitespaceTransform` | — | Remove all whitespace |
| `insert_synonyms` | `InsertSynonymsTransform` | probability | Replace words with synonyms (built-in map) |
| `escape_quotes` | `EscapeQuotesTransform` | — | Add backslash before quotes |
| `format_as_json` | `FormatAsJsonTransform` | — | Wrap prompt in a JSON object |
| `add_role_play` | `AddRolePlayTransform` | role | Prepend 'As a [role], ' to the prompt |
| `truncate` | `TruncateTransform` | max_chars | Truncate prompt to max_chars characters |
| `pad_to_length` | `PadToLengthTransform` | length, pad_char | Pad/truncate prompt to exact length |
| `random_case` | `RandomCaseTransform` | seed, probability | Randomly toggle character case |
| `character_substitution` | `CharacterSubstitutionTransform` | old_char, new_char | Substitute all occurrences of a character |

## Classifiers (27)

| Name | Class | Parameters | Description |
|------|-------|------------|-------------|
| `toxicity_score` | `ToxicityScoreClassifier` | — | Estimate toxicity (0=safe, 1=toxic) via keyword + TextBlob |
| `sentiment_score` | `SentimentScoreClassifier` | — | Estimate sentiment polarity (0=negative, 1=positive) |
| `intent_score` | `IntentScoreClassifier` | — | Estimate harmful intent likelihood (0=benign, 1=harmful) |
| `obscurity_score` | `ObscurityScoreClassifier` | — | Measure obfuscation from leetspeak/encoding/special chars |
| `length_score` | `LengthScoreClassifier` | min_len, max_len | Normalized length score (1=ideal range) |
| `repetition_score` | `RepetitionScoreClassifier` | — | Repetition score from n-gram frequency |
| `entropy_score` | `EntropyScoreClassifier` | — | Character entropy (high → likely encoded) |
| `language_score` | `LanguageScoreClassifier` | — | Natural language likelihood by char distribution |
| `jailbreak_likelihood` | `JailbreakLikelihoodClassifier` | database | Score from jailbreak pattern density |
| `contains_blacklisted_word` | `ContainsBlacklistedWordClassifier` | threshold | Score from blacklisted word ratio |
| `special_char_ratio` | `SpecialCharRatioClassifier` | — | Ratio of non-alphanumeric, non-space chars |
| `digit_ratio` | `DigitRatioClassifier` | — | Ratio of digit characters |
| `upper_case_ratio` | `UpperCaseRatioClassifier` | — | Ratio of uppercase letters |
| `punctuation_ratio` | `PunctuationRatioClassifier` | — | Ratio of punctuation characters |
| `whitespace_ratio` | `WhitespaceRatioClassifier` | — | Ratio of whitespace characters |
| `unique_token_ratio` | `UniqueTokenRatioClassifier` | — | Ratio of unique tokens to total tokens |
| `gpt2_perplexity` | `Gpt2PerplexityClassifier` | — | Heuristic perplexity proxy via character n-grams |
| `encoding_detection` | `EncodingDetectionClassifier` | — | Probability that prompt is encoded (base64, hex, rot13) |
| `refusal_similarity` | `RefusalSimilarityClassifier` | — | Similarity to refusal templates (keyword overlap) |
| `harmfulness_similarity` | `HarmfulnessSimilarityClassifier` | — | Similarity to harmful prompt patterns |
| `code_likelihood` | `CodeLikelihoodClassifier` | — | Likelihood prompt contains code (syntax features) |
| `json_likelihood` | `JsonLikelihoodClassifier` | — | Likelihood prompt is valid/looks like JSON |
| `sql_likelihood` | `SqlLikelihoodClassifier` | — | Likelihood prompt contains SQL syntax |
| `prompt_injection_likelihood` | `PromptInjectionLikelihoodClassifier` | — | Aggregate score for injection patterns |
| `roleplay_likelihood` | `RoleplayLikelihoodClassifier` | — | Score from role-play pattern density |
| `adversarial_suffix_score` | `AdversarialSuffixScoreClassifier` | — | Detect GCG-style adversarial suffixes |
| `persuasion_score` | `PersuasionScoreClassifier` | — | Detect persuasion techniques (PAP attacks) |

## Implementation Notes

- All primitives use `@dataclass` with `__post_init__` to set `name`, `parameters`, `input_type`, `output_type`, `metadata`.
- `to_dict()` uses `getattr` fallbacks for `version_id`, `created_at`, `deprecated_at`.
- Classifiers fall back to keyword heuristics when `textblob` is unavailable.
- Transforms with randomness (`insert_typos`, `word_shuffle`, `random_case`, `insert_synonyms`) use `random.Random(seed)` or `random.random()` for deterministic behavior at fixed seed/probability.
- Cipher transforms (ROT13, Caesar, Vigenère, Rail Fence, Atbash) implemented inline without external dependencies.
