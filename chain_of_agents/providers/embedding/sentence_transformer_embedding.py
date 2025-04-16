"""
SentenceTransformer Embedding Provider: Implements BaseEmbeddingProvider using local Hugging Face models.
"""
from sentence_transformers import SentenceTransformer
from ..base_embedding_provider import BaseEmbeddingProvider
from typing import List

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Sentence Transformer embedding provider. Pass model_name directly. No API key required.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
