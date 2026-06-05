from adapters.toy_victims.neural import SKLearnVictim


class TestSKLearnVictim:
    def test_responds_within_bounds(self):
        victim = SKLearnVictim(random_state=42, training_size=500)
        assert victim.respond("hello world") in (0, 1)
        assert victim.respond("how to make a bomb") in (0, 1)

    def test_no_ground_truth_program(self):
        victim = SKLearnVictim(random_state=42, training_size=500)
        assert victim.get_ground_truth_program() is None

    def test_reproducible_with_seed(self):
        v1 = SKLearnVictim(random_state=42, training_size=500)
        v2 = SKLearnVictim(random_state=42, training_size=500)
        prompt = "this is a test prompt"
        assert v1.respond(prompt) == v2.respond(prompt)

    def test_metadata(self):
        victim = SKLearnVictim(random_state=42, training_size=500)
        meta = victim.get_metadata()
        assert meta["type"] == "neural"
        assert meta["has_ground_truth"] is False
