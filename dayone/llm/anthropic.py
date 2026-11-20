"""Claude behind the `LLM` contract."""

from __future__ import annotations

import anthropic

from dayone.config import Settings

TIMEOUT_ATTEMPTS = 2  # one retry, on timeout only


class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    pass


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )
        self.calls = 0  # answered requests and their billed tokens, read by the benchmarks
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        for attempt in range(1, TIMEOUT_ATTEMPTS + 1):
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.APITimeoutError as error:
                if attempt == TIMEOUT_ATTEMPTS:
                    raise LLMTimeout(f"{self._model} timed out {TIMEOUT_ATTEMPTS} times") from error
                continue
            except anthropic.APIError as error:
                raise LLMError(f"{self._model} request failed: {error}") from error
            self.calls += 1
            self.input_tokens += message.usage.input_tokens
            self.output_tokens += message.usage.output_tokens
            return "".join(block.text for block in message.content if block.type == "text")
        raise AssertionError("unreachable")
