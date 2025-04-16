"""
Gemini Embedding Provider: Implements BaseEmbeddingProvider for embeddings.
"""
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ..base_embedding_provider import BaseEmbeddingProvider
import os
from typing import List

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "gemini-embedding-exp-03-07"):
        load_dotenv('/Users/mikemik/Documents/Projects/Python/qki_analytics/chain-of-agent/.env.prod')
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            if result.embeddings and len(result.embeddings) > 0:
                embedding_object = result.embeddings[0]
                if hasattr(embedding_object, 'values') and isinstance(embedding_object.values, list):
                    return embedding_object.values
                else:
                    print(f"Error: Unexpected embedding object format. Expected '.values' attribute with a list. Got type: {type(embedding_object)}")
            print("Warning: Embedding generation returned no embeddings.")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
        return []
