"""Pytest-based smoke test for package imports."""

import importlib


def test_package_imports():
    module = importlib.import_module("chain_of_agents")
    assert hasattr(module, "ChainOfAgents")

    from chain_of_agents.chunking.chunker import Chunker  # noqa: F401
    from chain_of_agents.agents.worker_agent import WorkerAgent  # noqa: F401
    from chain_of_agents.agents.manager_agent import ManagerAgent  # noqa: F401

    # The import statements above succeed if no ImportError is raised
