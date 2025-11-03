import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chain_of_agents import ChainOfAgents


class DummyEmbeddingProvider:
    def embed(self, text):
        return [0.0]


class DummyLLMProvider:
    def __init__(self):
        self.count_tokens_calls = []
        self.generate_calls = []

    def generate(self, prompt, temperature: float = 0.7, max_tokens: int = 1024):
        self.generate_calls.append(prompt)
        return "dummy-response"

    def count_tokens(self, text: str) -> int:
        self.count_tokens_calls.append(text)
        return max(1, len(text))


def test_chain_of_agents_requires_generation_provider():
    embedding_provider = DummyEmbeddingProvider()

    with pytest.raises(ValueError, match="requires at least one generation-capable provider"):
        ChainOfAgents(embedding_provider=embedding_provider)


def test_chain_of_agents_falls_back_to_worker_provider_for_token_counting():
    worker_provider = DummyLLMProvider()
    coa = ChainOfAgents(worker_llm_provider=worker_provider)

    text = "Short text for processing."
    query = "What is this?"
    result = coa.process(text=text, query=query)

    assert result["answer"] == "dummy-response"
    assert worker_provider.count_tokens_calls, "Worker provider should be used for token counting."
    # Worker should process chunks and manager should generate final answer
    assert len(worker_provider.generate_calls) >= 1
