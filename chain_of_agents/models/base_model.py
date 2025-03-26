"""
Base model interface for Chain of Agents implementation.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def __init__(self, model_name: str):
        """
        Initialize the model.
        
        Args:
            model_name: The name of the model to use.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The prompt to send to the model.
            temperature: The temperature for generation.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The generated text response.
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a string.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The number of tokens.
        """
        pass
