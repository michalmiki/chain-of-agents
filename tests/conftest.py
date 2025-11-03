"""Common fixtures and utilities for the test suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pytest

from chain_of_agents.providers.base_embedding_provider import BaseEmbeddingProvider
from chain_of_agents.providers.base_llm_provider import BaseLLMProvider


class RecordingLLMProvider(BaseLLMProvider):
    """LLM provider used in tests that records prompts and returns stubbed outputs."""

    def __init__(self, response_builder: Optional[Callable[[str, int], str]] = None):
        self.response_builder = response_builder
        self.prompts: List[str] = []

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        self.prompts.append(prompt)
        if self.response_builder is not None:
            return self.response_builder(prompt, len(self.prompts) - 1)
        return f"response-{len(self.prompts)}"

    def count_tokens(self, text: str) -> int:
        # Lightweight token approximation based on whitespace separated words.
        return max(1, len(text.split()))


@dataclass
class KeywordEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider that encodes presence of a keyword as a 1-D vector."""

    keyword: str

    def embed(self, text: str) -> List[float]:
        token = self.keyword.lower()
        return [1.0 if token in text.lower() else 0.0]


@pytest.fixture
def long_text() -> str:
    """Sample long text used in integration tests to exercise chunking."""

    sentences = [
        "This is sentence number {i} providing detailed context for testing the chain of agents pipeline.".format(i=i)
        for i in range(1, 13)
    ]
    # Insert blank lines periodically to produce friendlier chunk boundaries.
    paragraphs = [" ".join(sentences[i : i + 3]) for i in range(0, len(sentences), 3)]
    return "\n\n".join(paragraphs)
