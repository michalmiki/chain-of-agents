# Chain of Agents

__version__ = "0.1.0"

# Import main classes for easy access
from .chain_of_agents import ChainOfAgents
from .models.gemini_model import GeminiModel
from .chunking.chunker import Chunker
from .agents.worker_agent import WorkerAgent
from .agents.manager_agent import ManagerAgent
