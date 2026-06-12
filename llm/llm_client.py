import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import requests

# Load .env from project root (idempotent if already loaded by caller)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """LLM client backed by OpenRouter API.

    Fix 6C: Added exponential backoff retry for 429 rate-limit responses,
    configurable fallback model, and graceful degradation (returns empty
    string on persistent failure instead of crashing the pipeline).
    """

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

        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        for attempt_model in models_to_try:
            for attempt in range(self.max_retries + 1):
                try:
                    result = self._do_generate(
                        prompt, max_tokens, temperature,
                        enable_reasoning, attempt_model,
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
    ) -> Optional[str]:
        """Internal: performs a single generate request.

        Returns None if the model used was not the primary and the
        request failed (so the outer loop can try the fallback).
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
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

        # store reasoning context if available
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


def get_default_client() -> OpenRouterClient:
    """Return OpenRouter client (only backend now)."""
    return OpenRouterClient()


if __name__ == "__main__":
    client = get_default_client()
    print("Client ready — model:", client.model)