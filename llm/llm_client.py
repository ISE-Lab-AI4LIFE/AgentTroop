import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


class OpenAIClient:
    """LLM client backed by OpenAI SDK.

    Uses ``openai.OpenAI()`` with key from ``OPENAI_API_KEY`` env var.
    Shares the same ``generate()`` interface as ``OpenRouterClient``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        timeout_ms: int = 180000,
    ):
        from openai import OpenAI

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_ms = timeout_ms
        self._client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Send a prompt and return model response text.

        Returns empty string on failure so the pipeline can continue.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model_name = model or self.model
        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout_ms / 1000,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("OpenAI API error (model=%s): %s", model_name, e)
            return ""


class OpenRouterClient:
    """LLM client backed by OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        timeout_ms: int = 180000,
        max_retries: int = 3,
        fallback_model: str = "",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

        self.model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.fallback_model = fallback_model or os.environ.get(
            "OPENROUTER_FALLBACK_MODEL", ""
        )
        self._reasoning_history: Optional[dict] = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Send a prompt and return model response text.

        Retries with exponential backoff on 429 rate-limit responses.
        Falls back to ``fallback_model`` after exhausting retries with
        the primary model.

        Never raises an exception on rate-limit; returns empty string
        as last resort so the pipeline can continue.
        """
        enable_reasoning = kwargs.pop("reasoning", False)
        last_error: Optional[str] = None

        models_to_try = [model or self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        for attempt_model in models_to_try:
            for attempt in range(self.max_retries + 1):
                try:
                    result = self._do_generate(
                        prompt, max_tokens, temperature,
                        enable_reasoning, attempt_model,
                        system_prompt=system_prompt,
                    )
                    if result is not None:
                        return result
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else 0
                    if status == 429:
                        wait = min(2 ** attempt * 5, 120)
                        logger.warning(
                            "OpenRouter 429 (model=%s, attempt=%d/%d): "
                            "retrying in %ds",
                            attempt_model, attempt + 1, self.max_retries, wait,
                        )
                        time.sleep(wait)
                        last_error = f"429 rate-limited after {attempt + 1} attempts"
                    else:
                        logger.warning(
                            "OpenRouter HTTP %d (model=%s): %s",
                            status, attempt_model, str(e),
                        )
                        last_error = str(e)
                        break
                except requests.exceptions.Timeout:
                    logger.warning(
                        "OpenRouter timeout (model=%s, attempt=%d/%d)",
                        attempt_model, attempt + 1, self.max_retries,
                    )
                    last_error = f"timeout after {attempt + 1} attempts"
                    continue
                except requests.exceptions.ConnectionError as e:
                    logger.warning(
                        "OpenRouter connection error (model=%s): %s",
                        attempt_model, e,
                    )
                    last_error = str(e)
                    break
                except Exception as e:
                    logger.warning(
                        "OpenRouter unexpected error (model=%s): %s",
                        attempt_model, e,
                    )
                    last_error = str(e)
                    break

        logger.error(
            "OpenRouter all retries exhausted: %s. "
            "Pipeline will continue without LLM signal.",
            last_error,
        )
        return ""

    def _do_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        enable_reasoning: bool,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Internal: performs a single generate request."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if enable_reasoning:
            body["reasoning"] = {"enabled": True}
            if self._reasoning_history is not None:
                body["messages"].insert(0, self._reasoning_history)

        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body),
            timeout=self.timeout_ms / 1000,
        )

        resp.raise_for_status()
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        if enable_reasoning and "reasoning_details" in message:
            self._reasoning_history = {
                "role": "assistant",
                "content": content,
                "reasoning_details": message["reasoning_details"],
            }
        else:
            self._reasoning_history = None

        return content or ""

    def reset_reasoning(self) -> None:
        """Clear cached reasoning state."""
        self._reasoning_history = None


def get_default_client(backend: str = "openai") -> Any:
    """Return a default LLM client for the requested backend.

    Parameters
    ----------
    backend : str
        ``"openai"`` (default) or ``"openrouter"``.

    Returns
    -------
    ``OpenAIClient`` or ``OpenRouterClient``.
    """
    if backend == "openrouter":
        return OpenRouterClient()
    return OpenAIClient()


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "openai"
    client = get_default_client(backend)
    print(f"Client ready — backend={backend} model={client.model}")
