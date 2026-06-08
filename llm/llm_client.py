import os
import json
from typing import Any, Optional

import requests


class OpenRouterClient:
	"""LLM client backed by OpenRouter API."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		model: str = "",
		timeout_ms: int = 180000,
	):
		self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
		if not self.api_key:
			raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

		self.model = model or os.environ.get("OPENROUTER_MODEL", "openrouter/free")
		self.timeout_ms = timeout_ms
		self._reasoning_history: Optional[dict] = None

	def generate(
		self,
		prompt: str,
		max_tokens: int = 4096,
		temperature: float = 0.0,
		**kwargs,
	) -> str:
		"""Send a prompt and return model response text."""

		enable_reasoning = kwargs.pop("reasoning", False)

		body: dict[str, Any] = {
			"model": self.model,
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