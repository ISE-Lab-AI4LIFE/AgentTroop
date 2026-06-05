from core.intervention import Intervention
from core.primitive import PrimitiveRegistry, Rot13Transform


def test_intervention_apply_applies_transforms_in_order():
    intervention = Intervention(base_prompt="hello", transforms=[Rot13Transform()])
    assert intervention.apply() == "uryyb"
    assert intervention.final_prompt == "uryyb"


def test_intervention_serialization_roundtrip():
    registry = PrimitiveRegistry()
    intervention = Intervention(base_prompt="safe", transforms=[Rot13Transform()], metadata={"reason": "test"})
    data = intervention.to_dict()
    restored = Intervention.from_dict(data)

    assert restored.base_prompt == intervention.base_prompt
    assert restored.metadata == intervention.metadata
    assert restored.id == intervention.id
    assert restored.final_prompt == intervention.final_prompt
