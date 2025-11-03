import unittest
from unittest import mock

from chain_of_agents.chunking.chunker import Chunker


class FakeTokenizer:
    """Simple character-level tokenizer for deterministic testing."""

    def encode(self, text: str):
        return [ord(ch) for ch in text]

    def decode(self, tokens):
        return ''.join(chr(t) for t in tokens)


class DummyLLMProvider:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class DummyEmbeddingProvider:
    def embed(self, text: str):
        lower = text.lower()
        if "broken" in lower:
            return []
        if "noise" in lower or "irrelevant" in lower:
            return [0.0, 1.0]
        if "relevant" in lower:
            return [1.0, 0.0]
        return [0.5, 0.5]


class ChunkerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.llm_provider = DummyLLMProvider(self.tokenizer)
        self.chunker = Chunker(token_budget=120)
        self.query = "Relevant query"
        self.instruction = "Process the provided chunks carefully."

    def test_create_chunks_respects_token_budget(self):
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

        chunks = self.chunker.create_chunks(
            text=text,
            query=self.query,
            instruction_prompt=self.instruction,
            token_counter=self.llm_provider.count_tokens,
            tokenizer=self.tokenizer,
        )

        query_tokens = self.chunker._get_token_length(self.query, self.llm_provider.count_tokens, self.tokenizer)
        instruction_tokens = self.chunker._get_token_length(self.instruction, self.llm_provider.count_tokens, self.tokenizer)
        reserved = int(self.chunker.token_budget * 0.2)
        chunk_budget = self.chunker.token_budget - query_tokens - instruction_tokens - reserved

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            chunk_tokens = len(self.tokenizer.encode(chunk))
            self.assertLessEqual(chunk_tokens, chunk_budget)
            self.assertGreater(chunk_tokens, 0)

    def test_create_and_filter_chunks_filters_below_threshold(self):
        embedding_provider = DummyEmbeddingProvider()
        mock_chunks = [
            "Relevant chunk includes important signal data.",
            "This section is full of noise and irrelevant chatter.",
            "Broken chunk should not cause errors.",
        ]

        with mock.patch.object(self.chunker, "create_chunks", return_value=mock_chunks):
            filtered = self.chunker.create_and_filter_chunks(
                text="unused",
                query=self.query,
                instruction_prompt=self.instruction,
                embedding_provider=embedding_provider,
                llm_provider=self.llm_provider,
                similarity_threshold=0.6,
                verbose=True,
            )

        self.assertEqual(filtered, [mock_chunks[0]])

    def test_create_and_filter_chunks_returns_unfiltered_on_query_failure(self):
        class FailingEmbeddingProvider(DummyEmbeddingProvider):
            def __init__(self, query: str):
                self._query = query

            def embed(self, text: str):
                if text == self._query:
                    return []
                return super().embed(text)

        embedding_provider = FailingEmbeddingProvider(self.query)
        mock_chunks = ["Relevant data"]

        with mock.patch.object(self.chunker, "create_chunks", return_value=mock_chunks):
            filtered = self.chunker.create_and_filter_chunks(
                text="unused",
                query=self.query,
                instruction_prompt=self.instruction,
                embedding_provider=embedding_provider,
                llm_provider=self.llm_provider,
                similarity_threshold=0.6,
            )

        self.assertEqual(filtered, mock_chunks)


if __name__ == "__main__":
    unittest.main()
