"""Unit tests for the chunking utilities."""
import re

import pytest

from chain_of_agents.chunking.chunker import Chunker

from tests.conftest import KeywordEmbeddingProvider, RecordingLLMProvider


@pytest.fixture
def simple_token_counter():
    """Token counter that approximates tokens from character count for testing."""

    def _counter(text: str) -> int:
        # keep the budget generous enough for small strings while scaling with length
        return max(1, len(text) // 50)

    return _counter


def _clean_text(text: str) -> str:
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def test_create_chunks_respects_budget(simple_token_counter):
    chunker = Chunker(token_budget=100)
    base_sentence = (
        "This is a fairly long sentence used to validate chunk creation over repeated text snippets. "
    )
    text = "\n".join(base_sentence for _ in range(20))
    query = "What is described?"
    instruction = "Answer based on the provided text."

    chunks = chunker.create_chunks(text, query, instruction, simple_token_counter)

    assert len(chunks) > 1, "Chunker should split the long text into multiple chunks"

    reserved_for_cu = int(chunker.token_budget * 0.2)
    chunk_budget = chunker.token_budget - simple_token_counter(query) - simple_token_counter(instruction) - reserved_for_cu
    char_budget = chunk_budget * 4

    for chunk in chunks:
        assert len(chunk) <= char_budget
        assert chunk, "Chunk should not be empty"


def test_create_chunks_invalid_budget(simple_token_counter):
    chunker = Chunker(token_budget=10)
    with pytest.raises(ValueError):
        chunker.create_chunks("short text", "a" * 500, "b" * 500, simple_token_counter)


def test_create_and_filter_chunks_filters_relevant_text(simple_token_counter):
    chunker = Chunker(token_budget=80)
    llm_provider = RecordingLLMProvider()
    embedding_provider = KeywordEmbeddingProvider(keyword="alpha")
    text = "alpha content here. beta content elsewhere. alpha again for relevance."

    filtered = chunker.create_and_filter_chunks(
        text=text,
        query="alpha question",
        instruction_prompt="Answer based on the provided text.",
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        similarity_threshold=0.5,
    )

    assert filtered, "At least one relevant chunk should be kept"
    assert all("alpha" in chunk.lower() for chunk in filtered), "Filtered chunks must contain the keyword"
