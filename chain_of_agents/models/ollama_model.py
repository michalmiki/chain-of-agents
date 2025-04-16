"""
Ollama model interface for Chain of Agents implementation.
"""
import os
from dotenv import load_dotenv
from .base_model import BaseModel

try:
    from ollama import chat, ChatResponse
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class OllamaModel(BaseModel):
    """Wrapper for Ollama models."""
    
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize the Ollama model.
        
        Args:
            model_name: The name of the Ollama model to use (e.g., 'llama3.2', 'mistral', etc.).
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama package is not installed. Please install it with 'pip install ollama'"
            )
        
        # Load environment variables
        load_dotenv('/Users/mikemik/Documents/Projects/Python/qki_analytics/chain-of-agent/.env.prod')
        
        # Set the model name
        self.model_name = model_name
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt: The prompt to send to the model.
            temperature: The temperature for generation.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The generated text response.
        """
        try:
            # Create the message object
            messages = [
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
            
            # Set up parameters
            options = {
                'temperature': temperature,
                'num_predict': max_tokens
            }
            
            # Call the Ollama API
            kwargs = {'model': self.model_name, 'messages': messages, 'options': options}
            if self.api_base:
                kwargs['api_base'] = self.api_base
                
            response: ChatResponse = chat(**kwargs)
            
            # Extract the response text
            return response.message.content
        except Exception as e:
            print(f"Error generating content with Ollama: {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a string.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The number of tokens.
        """
        try:
            # Ollama doesn't provide a direct token counting method
            # Use a simple approximation: 4 characters per token
            return len(text) // 4
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback: estimate tokens as words / 0.75 (rough approximation)
            return int(len(text.split()) / 0.75)
