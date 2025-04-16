"""
Base interface for embedding providers.
"""
from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        """
        pass
