"""
Chunking module for the Chain of Agents implementation.
Splits text based on accurate token counts.
"""
import logging
import re
from typing import List

import numpy as np

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency declared but guard for robustness
    tiktoken = None
from ..providers.base_embedding_provider import BaseEmbeddingProvider
from ..providers.base_llm_provider import BaseLLMProvider


logger = logging.getLogger(__name__)


class Chunker:
    """
    Splits long texts into manageable chunks according to token budget.
    Implements the chunking strategy described in the Chain of Agents paper.
    Can also filter chunks based on embedding similarity to a query.
    """
    
    def __init__(self, token_budget: int = 12000):
        """
        Initialize the chunker.
        
        Args:
            token_budget: The token budget for each chunk, default is 12k tokens.
        """
        self.token_budget = token_budget
        
    def _resolve_tokenizer(self, tokenizer=None):
        """Return a tokenizer compatible with the provided model if possible."""
        if tokenizer and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
            return tokenizer
        if tiktoken is None:
            return None
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - defensive, should not happen with valid install
            return None

    def _get_token_length(self, text: str, token_counter, tokenizer=None) -> int:
        """
        Get the token length of a text.
        
        Args:
            text: The text to count tokens for.
            token_counter: A callable that counts tokens.
            
        Returns:
            The number of tokens.
        """
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception:  # pragma: no cover - fallback to provider counter
                logger.debug("Tokenizer encode failed; falling back to provided token counter.")
        return token_counter(text)

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            logger.debug("Received None embedding when computing cosine similarity.")
            return 0.0
        if not isinstance(vec1, (list, np.ndarray)) or not isinstance(vec2, (list, np.ndarray)):
            logger.debug("Cosine similarity requires list or ndarray embeddings. Skipping computation.")
            return 0.0

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if vec1.ndim == 0 or vec2.ndim == 0 or vec1.size == 0 or vec2.size == 0:
            logger.debug("Encountered empty embedding vector while computing cosine similarity.")
            return 0.0
        if vec1.shape != vec2.shape:
            logger.debug("Embedding vectors have mismatched shapes; skipping cosine similarity computation.")
            return 0.0
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0  # Avoid division by zero
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def create_chunks(
        self,
        text: str,
        query: str, 
        instruction_prompt: str, 
        token_counter,
        tokenizer=None,
    ) -> List[str]:
        """
        Create chunks from text based on token budget.
        
        The chunk size is calculated as:
        B = token_budget - tokens(query) - tokens(instruction_prompt)
        
        Args:
            text: The source text to chunk.
            query: The query or task description.
            instruction_prompt: The instruction prompt for the worker agent.
            token_counter: A callable that counts tokens.
            
        Returns:
            A list of text chunks.
        """
        tokenizer = self._resolve_tokenizer(tokenizer)

        # Calculate available budget for each chunk
        query_tokens = self._get_token_length(query, token_counter, tokenizer)
        instruction_tokens = self._get_token_length(instruction_prompt, token_counter, tokenizer)
        
        # Reserve some space for the communication unit from the previous worker
        # Assuming the communication unit will be roughly 20% of the chunk size
        reserved_for_cu = int(self.token_budget * 0.2)
        
        # Calculate chunk budget
        chunk_budget = self.token_budget - query_tokens - instruction_tokens - reserved_for_cu
        
        if chunk_budget <= 0:
            raise ValueError(
                f"Token budget ({self.token_budget}) is too small to accommodate "
                f"query ({query_tokens}) and instruction ({instruction_tokens}) tokens."
            )

        # Clean the text slightly
        text = re.sub(r'\n+', '\n', text) # Keep single newlines
        text = re.sub(r'[ \t]+', ' ', text) # Consolidate whitespace

        chunks = []
        if tokenizer is not None:
            token_ids = tokenizer.encode(text)
            start_index = 0
            while start_index < len(token_ids):
                end_index = min(start_index + chunk_budget, len(token_ids))
                chunk_tokens = token_ids[start_index:end_index]
                chunk_text = tokenizer.decode(chunk_tokens).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                start_index = end_index
        else:
            # Fallback to character-based chunking if tokenizer is unavailable
            char_budget = chunk_budget * 4
            start_index = 0
            text_length = len(text)

            while start_index < text_length:
                end_index = min(start_index + char_budget, text_length)
                if end_index < text_length:
                    break_point = -1
                    newline_pos = text.rfind('\n', max(start_index, end_index - 50), end_index)
                    space_pos = text.rfind(' ', max(start_index, end_index - 50), end_index)

                    if newline_pos > start_index:
                        break_point = newline_pos + 1
                    elif space_pos > start_index:
                        break_point = space_pos + 1

                    if break_point != -1:
                        end_index = break_point

                chunk = text[start_index:end_index].strip()
                if chunk:
                    while self._get_token_length(chunk, token_counter) > chunk_budget and len(chunk) > 1:
                        trim_point = max(chunk.rfind('\n', 0, len(chunk) - 1), chunk.rfind(' ', 0, len(chunk) - 1))
                        if trim_point == -1:
                            chunk = chunk[:max(1, len(chunk) - 1)].strip()
                        else:
                            chunk = chunk[:trim_point].strip()
                    if chunk and self._get_token_length(chunk, token_counter) <= chunk_budget:
                        chunks.append(chunk)
                start_index = end_index
                while start_index < text_length and text[start_index].isspace():
                    start_index += 1

        return chunks

    def create_and_filter_chunks(
        self,
        text: str,
        query: str,
        instruction_prompt: str,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider,
        similarity_threshold: float = 0.7,
        verbose: bool = False
    ) -> List[str]:
        """
        Create chunks and filter them based on embedding similarity to the query.

        Args:
            text: The source text to chunk.
            query: The query or task description.
            instruction_prompt: The instruction prompt for the worker agent.
            embedding_provider: The embedding provider instance for generating embeddings.
            llm_provider: The LLM provider instance for counting tokens.
            similarity_threshold: The minimum cosine similarity score for a chunk to be considered relevant.
            verbose: Whether to print detailed logging during filtering.

        Returns:
            A list of relevant text chunks, preserving original order.
        """
        # Ensure we have an LLM provider for token counting
        if llm_provider is None:
            raise ValueError("llm_provider is required for create_and_filter_chunks")
        # 1. Create initial chunks based on token budget
        token_counter = llm_provider.count_tokens
        tokenizer = getattr(llm_provider, "tokenizer", None)
        initial_chunks = self.create_chunks(
            text=text,
            query=query,
            instruction_prompt=instruction_prompt,
            token_counter=token_counter,
            tokenizer=tokenizer,
        )

        if not initial_chunks:
            return []

        # 2. Embed the query
        try:
            query_embedding = embedding_provider.embed(query)
            if not query_embedding:
                logger.warning("Failed to embed query. Skipping similarity filtering.")
                return initial_chunks
        except Exception as e:
            logger.warning("Error embedding query: %s. Skipping similarity filtering.", e)
            return initial_chunks


        # 3. Embed and filter chunks
        relevant_chunks = []
        verbose_log = logger.info if verbose else logger.debug
        if verbose:
            verbose_log("--- Chunk Filtering Details ---")
        for i, chunk in enumerate(initial_chunks):
            try:
                chunk_embedding = embedding_provider.embed(chunk)
                if not chunk_embedding:
                    logger.debug("Failed to embed chunk %s; skipping.", i + 1)
                    continue # Skip chunk if embedding fails

                similarity = self._cosine_similarity(query_embedding, chunk_embedding)

                # 4. Keep chunk if similarity is above threshold
                if similarity >= similarity_threshold:
                    if verbose:
                        verbose_log(
                            "  Chunk %s/%s: Similarity = %.4f [KEPT]",
                            i + 1,
                            len(initial_chunks),
                            similarity,
                        )
                    relevant_chunks.append(chunk)
                else:
                    if verbose:
                        verbose_log(
                            "  Chunk %s/%s: Similarity = %.4f [FILTERED]",
                            i + 1,
                            len(initial_chunks),
                            similarity,
                        )

            except Exception as e:
                logger.debug(
                    "Error processing chunk %s embedding or similarity: %s. Skipping chunk.",
                    i + 1,
                    e,
                )
                if verbose:
                    verbose_log(
                        "  Chunk %s/%s: [ERROR DURING PROCESSING]",
                        i + 1,
                        len(initial_chunks),
                    )
                continue # Skip chunk on error

        if verbose:
            verbose_log("--- End Chunk Filtering ---")
        logger.info(
            "Filtered chunks: Kept %s out of %s based on similarity threshold %.2f",
            len(relevant_chunks),
            len(initial_chunks),
            similarity_threshold,
        )
        return relevant_chunks
