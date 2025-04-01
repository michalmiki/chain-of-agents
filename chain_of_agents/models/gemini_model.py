"""
Gemini model interface for Chain of Agents implementation.
"""
import os
from google import genai
from google.genai import types # Import types for EmbedContentConfig
from dotenv import load_dotenv
from .base_model import BaseModel
from typing import List


class GeminiModel(BaseModel):
    """Wrapper for Google Gemini 2.0 Flash model."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini model.
        
        Args:
            model_name: The name of the Gemini model to use.
        """
        # Load environment variables
        load_dotenv('/Users/mikemik/Documents/Projects/Python/qki_analytics/chain-of-agent/.env.prod')
        
        # Initialize the Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate a response from the Gemini model.
        
        Args:
            prompt: The prompt to send to the model.
            temperature: The temperature for generation.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            The generated text response.
        """
        try:
            # According to the documentation, we need to pass the configuration as separate parameters
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
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
            # For now, use a simple approximation as the client API doesn't have a direct count_tokens method
            # Approximate: 4 characters per token
            return len(text) // 4
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback: estimate tokens as words / 0.75 (rough approximation)
            return int(len(text.split()) / 0.75)

    def embed_content(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        try:
            # Use the specified embedding model with the correct config structure
            result = self.client.models.embed_content(
                model="gemini-embedding-exp-03-07",  # As requested
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY") # Correct usage
            )
            # The result.embeddings is a list containing the embedding object(s)
            if result.embeddings and len(result.embeddings) > 0:
                 embedding_object = result.embeddings[0]
                 # Check specifically for the 'values' attribute which should contain the List[float]
                 if hasattr(embedding_object, 'values') and isinstance(embedding_object.values, list):
                     return embedding_object.values
                 else:
                     # If 'values' attribute is missing or not a list, embedding failed or format is unexpected
                     print(f"Error: Unexpected embedding object format. Expected '.values' attribute with a list. Got type: {type(embedding_object)}")
                     return [] # Return empty list to indicate failure
            else:
                 print("Warning: Embedding generation returned no embeddings.")
                 return []
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
