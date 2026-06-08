from core.grammar import Grammar
from core.primitive import PrimitiveRegistry


def test_grammar_smtlib_contains_primitive_names():
    registry = PrimitiveRegistry()
    grammar = Grammar(registry)
    smtlib = grammar.to_smtlib()

    assert isinstance(smtlib, str)
    assert smtlib.strip() != ""
    assert any(
        name in smtlib for name in registry.list_primitives() + ["contains_word", "contains_any_word"]
    )
