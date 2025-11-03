"""Integration-style tests for the ChainOfAgents orchestration."""

from chain_of_agents import ChainOfAgents

from tests.conftest import KeywordEmbeddingProvider, RecordingLLMProvider


def _deterministic_response(prompt: str, index: int) -> str:
    if "Input Chunk:" in prompt:
        return f"chunk-response-{index + 1}"
    if "Answer:" in prompt or "Summary:" in prompt:
        return f"manager-response-{index + 1}"
    return f"response-{index + 1}"


def test_query_flow_produces_answer_and_metadata(long_text):
    provider = RecordingLLMProvider(response_builder=_deterministic_response)
    chain = ChainOfAgents(
        llm_provider=provider,
        embedding_provider=KeywordEmbeddingProvider(keyword="context"),
        token_budget=200,
    )

    result = chain.query(text=long_text, query="What context is provided?")

    assert result["answer"].startswith("manager-response"), "Manager response should be returned as answer"

    metadata = result["metadata"]
    assert metadata["num_chunks"] == len(metadata["communication_units"]) > 0
    assert metadata["final_cu"] == metadata["communication_units"][-1]
    assert isinstance(metadata["processing_time"], float)

    # Ensure both worker and manager prompts were invoked
    assert any("Input Chunk:" in prompt for prompt in provider.prompts)
    assert any(prompt.strip().endswith("Answer:") for prompt in provider.prompts)


def test_summarize_flow_uses_summary_prompts(long_text):
    provider = RecordingLLMProvider(response_builder=_deterministic_response)
    chain = ChainOfAgents(
        llm_provider=provider,
        embedding_provider=KeywordEmbeddingProvider(keyword="context"),
        token_budget=200,
    )

    result = chain.summarize(text=long_text)

    assert result["answer"].startswith("manager-response"), "Manager response should be returned as summary"

    metadata = result["metadata"]
    assert metadata["num_chunks"] == len(metadata["communication_units"]) > 0
    assert metadata["final_cu"] == metadata["communication_units"][-1]

    # Summarization prompts should not include question cues
    assert all("Question:" not in prompt for prompt in provider.prompts)
    assert any(prompt.strip().endswith("Summary:") for prompt in provider.prompts)
