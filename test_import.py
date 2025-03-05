#!/usr/bin/env python
"""
Test script to verify that the package can be imported correctly.
"""

try:
    # Try importing from src (when installed in development mode)
    from src import ChainOfAgents, GeminiModel, Chunker, WorkerAgent, ManagerAgent
    print("Successfully imported from src package!")
except ImportError:
    try:
        # Try importing from chain_of_agents (when installed via pip)
        from chain_of_agents import ChainOfAgents, GeminiModel, Chunker, WorkerAgent, ManagerAgent
        print("Successfully imported from chain_of_agents package!")
    except ImportError as e:
        print(f"Error importing package: {e}")

print("\nPackage structure verification:")
print("------------------------------")
print(f"ChainOfAgents: {ChainOfAgents.__module__}")
print(f"GeminiModel: {GeminiModel.__module__}")
print(f"Chunker: {Chunker.__module__}")
print(f"WorkerAgent: {WorkerAgent.__module__}")
print(f"ManagerAgent: {ManagerAgent.__module__}")
