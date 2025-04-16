"""
OpenAI Embedding Provider: Implements BaseEmbeddingProvider using OpenAI's embedding API.
"""
from ..base_embedding_provider import BaseEmbeddingProvider
import os
from typing import List, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None):
        """
        OpenAI embedding provider. Pass api_key directly or set OPENAI_API_KEY in environment.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed. Please install it with 'pip install openai'")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or constructor argument")
        openai.api_key = api_key
        self.model_name = model_name
        self.api_key = api_key

    def embed(self, text: str) -> List[float]:
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            # OpenAI returns a list of embeddings (one per input)
            embedding = response["data"][0]["embedding"]
            return embedding
        except Exception as e:
            print(f"Error generating embedding with OpenAI: {e}")
            return []
