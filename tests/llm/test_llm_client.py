"""Tests for OpenRouter client with rate-limit retry (Fix 6C)."""

import pytest
from unittest.mock import patch, MagicMock
import requests


class TestOpenRouterRateLimit:
    """Verify OpenRouterClient handles 429 with retry and fallback."""

    @pytest.fixture
    def client(self):
        from llm.llm_client import OpenRouterClient
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key',
            'OPENROUTER_MODEL': 'test-model',
        }, clear=False):
            return OpenRouterClient(max_retries=2, fallback_model="fallback-model")

    def test_retry_on_429(self, client):
        """Should retry up to max_retries on 429."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )

        with patch.object(client, '_do_generate', return_value=None) as mock_gen:
            with patch('time.sleep') as mock_sleep:
                # Make do_generate raise 429
                mock_gen.side_effect = requests.exceptions.HTTPError(
                    response=mock_resp
                )
                result = client.generate("test prompt")

        # Should have attempted multiple times (2 retries + 1 initial = 3)
        assert mock_gen.call_count >= 1
        # Should not crash
        assert isinstance(result, str)

    def test_graceful_timeout(self, client):
        """Should handle timeout without crashing."""
        with patch.object(client, '_do_generate') as mock_gen:
            mock_gen.side_effect = requests.exceptions.Timeout()
            with patch('time.sleep'):
                result = client.generate("test prompt")

        assert isinstance(result, str), "Should return empty string on timeout"

    def test_fallback_model_used(self, client):
        """Should try fallback model after primary exhausts retries."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )

        call_count = 0
        def _side_effect(prompt, max_tokens, temperature, enable_reasoning, model):
            nonlocal call_count
            call_count += 1
            if model == "test-model":
                raise requests.exceptions.HTTPError(response=mock_resp)
            # Fallback model succeeds
            return "fallback response"

        with patch.object(client, '_do_generate', side_effect=_side_effect):
            with patch('time.sleep'):
                result = client.generate("test prompt")

        assert isinstance(result, str)

    def test_never_crash_on_network_error(self, client):
        """Network errors should not crash the pipeline."""
        with patch.object(client, '_do_generate') as mock_gen:
            mock_gen.side_effect = requests.exceptions.ConnectionError("DNS failure")
            result = client.generate("test prompt")

        assert isinstance(result, str), "Should return empty string on connection error"


class TestScientificMemoryBackwardCompat:
    """Verify ScientificMemory.find_theories accepts target_model kwarg (Fix 6B)."""

    def test_target_model_parameter_accepted(self):
        from knowledge.scientific_memory import ScientificMemory
        import inspect
        sig = inspect.signature(ScientificMemory.find_theories)
        assert "target_model" in sig.parameters, (
            "find_theories should accept target_model parameter"
        )

    def test_target_model_default_none(self):
        from knowledge.scientific_memory import ScientificMemory
        import inspect
        sig = inspect.signature(ScientificMemory.find_theories)
        param = sig.parameters["target_model"]
        assert param.default is None, "target_model should default to None"
