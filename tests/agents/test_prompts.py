"""Tests covering prompt generation for worker and manager agents."""

from chain_of_agents.agents.manager_agent import ManagerAgent
from chain_of_agents.agents.worker_agent import WorkerAgent

from tests.conftest import RecordingLLMProvider


def test_worker_query_prompt_includes_required_sections():
    provider = RecordingLLMProvider()
    agent = WorkerAgent(llm_provider=provider)

    agent.process_chunk(
        chunk="Chunk text for testing.",
        previous_cu="Previous summary.",
        query="What happened?",
        task_desc="Answer clearly.",
        is_query_based=True,
    )

    prompt = provider.prompts[-1]
    assert prompt.startswith("Answer clearly."), "Task description should lead the prompt"
    assert "Input Chunk:\nChunk text for testing." in prompt
    assert "Previous summary." in prompt
    assert "Question: What happened?" in prompt
    assert "generate a summary" in prompt


def test_worker_non_query_prompt_omits_question():
    provider = RecordingLLMProvider()
    agent = WorkerAgent(llm_provider=provider)

    agent.process_chunk(
        chunk="Another chunk.",
        previous_cu=None,
        query=None,
        task_desc="Summarize the content.",
        is_query_based=False,
    )

    prompt = provider.prompts[-1]
    assert prompt.startswith("Summarize the content."), "Custom task description should be used"
    assert "There is no previous source text." in prompt
    assert "Question:" not in prompt
    assert prompt.strip().endswith("include all important information."), "Prompt should preserve the template guidance"


def test_manager_query_prompt_mentions_question():
    provider = RecordingLLMProvider()
    agent = ManagerAgent(llm_provider=provider)

    agent.generate_answer(
        final_cu="Final communication unit.",
        query="Explain the outcome.",
        task_desc="Provide the best answer.",
        is_query_based=True,
    )

    prompt = provider.prompts[-1]
    assert prompt.startswith("Provide the best answer."), "Manager should use the provided task description"
    assert "Final communication unit." in prompt
    assert "Question: Explain the outcome." in prompt
    assert prompt.strip().endswith("Answer:"), "Prompt should end with the answer cue"


def test_manager_non_query_prompt_requests_summary():
    provider = RecordingLLMProvider()
    agent = ManagerAgent(llm_provider=provider)

    agent.generate_answer(
        final_cu="Manager input.",
        query=None,
        task_desc="Summarize the findings.",
        is_query_based=False,
    )

    prompt = provider.prompts[-1]
    assert prompt.startswith("Summarize the findings.")
    assert "Manager input." in prompt
    assert "Question:" not in prompt
    assert prompt.strip().endswith("Summary:"), "Prompt should end with the summary cue"
