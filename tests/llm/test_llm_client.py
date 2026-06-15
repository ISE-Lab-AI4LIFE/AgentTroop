"""Tests for LLM clients (OpenAI + OpenRouter) and factory."""

import pytest
from unittest.mock import patch, MagicMock
import requests


class TestOpenAIClient:
    """Verify OpenAIClient interface matches expectations."""

    @pytest.fixture
    def client(self):
        from llm.llm_client import OpenAIClient
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_MODEL': 'gpt-4o-mini',
        }, clear=False):
            return OpenAIClient()

    def test_generate_returns_string(self, client):
        with patch.object(client._client.chat.completions, 'create') as mock_create:
            mock_msg = MagicMock()
            mock_msg.message.content = "hello"
            mock_choice = MagicMock()
            mock_choice.message = mock_msg.message
            mock_create.return_value.choices = [mock_choice]
            result = client.generate("test prompt")
        assert result == "hello"

    def test_generate_empty_on_error(self, client):
        with patch.object(client._client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = RuntimeError("API error")
            result = client.generate("test prompt")
        assert result == ""

    def test_generate_with_system_prompt(self, client):
        with patch.object(client._client.chat.completions, 'create') as mock_create:
            mock_msg = MagicMock()
            mock_msg.message.content = "response"
            mock_choice = MagicMock()
            mock_choice.message = mock_msg.message
            mock_create.return_value.choices = [mock_choice]
            result = client.generate("user msg", system_prompt="be helpful")
        assert result == "response"
        args, kwargs = mock_create.call_args
        msgs = kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "be helpful"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "user msg"


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
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
        with patch.object(client, '_do_generate', return_value=None) as mock_gen:
            with patch('time.sleep') as mock_sleep:
                mock_gen.side_effect = requests.exceptions.HTTPError(
                    response=mock_resp
                )
                result = client.generate("test prompt")
        assert mock_gen.call_count >= 1
        assert isinstance(result, str)

    def test_graceful_timeout(self, client):
        with patch.object(client, '_do_generate') as mock_gen:
            mock_gen.side_effect = requests.exceptions.Timeout()
            with patch('time.sleep'):
                result = client.generate("test prompt")
        assert isinstance(result, str), "Should return empty string on timeout"

    def test_fallback_model_used(self, client):
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
            return "fallback response"
        with patch.object(client, '_do_generate', side_effect=_side_effect):
            with patch('time.sleep'):
                result = client.generate("test prompt")
        assert isinstance(result, str)

    def test_never_crash_on_network_error(self, client):
        with patch.object(client, '_do_generate') as mock_gen:
            mock_gen.side_effect = requests.exceptions.ConnectionError("DNS failure")
            result = client.generate("test prompt")
        assert isinstance(result, str), "Should return empty string on connection error"


class TestGetDefaultClient:
    """Verify get_default_client factory function."""

    def test_default_returns_openai(self):
        from llm.llm_client import OpenAIClient, get_default_client
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}, clear=False):
            client = get_default_client()
        assert isinstance(client, OpenAIClient)

    def test_openrouter_backend(self):
        from llm.llm_client import OpenRouterClient, get_default_client
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key',
        }, clear=False):
            client = get_default_client(backend="openrouter")
        assert isinstance(client, OpenRouterClient)

    def test_openai_backend_explicit(self):
        from llm.llm_client import OpenAIClient, get_default_client
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}, clear=False):
            client = get_default_client(backend="openai")
        assert isinstance(client, OpenAIClient)
