from core.observation import Observation


def test_observation_to_from_dict():
    observation = Observation(
        intervention_id="int_123",
        outcome=1,
        victim_id="victim_1",
        campaign_id="camp_1",
        experiment_id="exp_1",
        raw_response="Refused by model",
        latency=0.12,
        token_usage=37,
        environment_metadata={"model": "gpt-test"},
        metadata={"note": "edge case"},
    )
    data = observation.to_dict()
    restored = Observation.from_dict(data)

    assert restored.intervention_id == observation.intervention_id
    assert restored.outcome == observation.outcome
    assert restored.victim_id == observation.victim_id
    assert restored.campaign_id == observation.campaign_id
    assert restored.experiment_id == observation.experiment_id
    assert restored.raw_response == observation.raw_response
    assert restored.latency == observation.latency
    assert restored.token_usage == observation.token_usage
    assert restored.environment_metadata == observation.environment_metadata
    assert restored.metadata == observation.metadata
    assert restored.id == observation.id
