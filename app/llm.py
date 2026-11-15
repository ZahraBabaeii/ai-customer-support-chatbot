"""LLM client abstraction and the Gemini implementation."""

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


class LLMError(RuntimeError):
    """The LLM provider failed or returned no usable text."""


class LLMClient(Protocol):
    model: str

    def generate(self, system_prompt: str, messages: list[Message]) -> str: ...


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        timeout_seconds: int = 30,
    ) -> None:
        # Imported lazily so demo mode does not need the SDK loaded.
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )
        self.model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def generate(self, system_prompt: str, messages: list[Message]) -> str:
        types = self._types
        contents = [
            types.Content(
                role="user" if m.role == "user" else "model",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in messages
        ]
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response")
        return text
