"""
Chunking module for the Chain of Agents implementation.
Splits text based on character length.
"""
import re
import numpy as np
from typing import List, Tuple, Optional
from ..providers.base_embedding_provider import BaseEmbeddingProvider
from ..providers.base_llm_provider import BaseLLMProvider


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
        
    def _get_token_length(self, text: str, token_counter) -> int:
        """
        Get the token length of a text.
        
        Args:
            text: The text to count tokens for.
            token_counter: A callable that counts tokens.
            
        Returns:
            The number of tokens.
        """
        return token_counter(text)

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        # --- Debugging Start ---
        print(f"DEBUG: _cosine_similarity received vec1 type: {type(vec1)}")
        print(f"DEBUG: _cosine_similarity received vec2 type: {type(vec2)}")
        if not isinstance(vec1, (list, np.ndarray)):
            print(f"ERROR: vec1 is not a list or ndarray! Value: {vec1}")
            return 0.0 # Cannot calculate similarity
        if not isinstance(vec2, (list, np.ndarray)):
            print(f"ERROR: vec2 is not a list or ndarray! Value: {vec2}")
            return 0.0 # Cannot calculate similarity
        # --- Debugging End ---

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if vec1.ndim == 0 or vec2.ndim == 0 or vec1.size == 0 or vec2.size == 0:
             print(f"Warning: One or both embedding vectors are empty or scalar. vec1: {vec1}, vec2: {vec2}")
             return 0.0
        if vec1.shape != vec2.shape:
             # This might happen if embedding fails for one or returns different dimensions
             print(f"Warning: Embedding vectors have different shapes: {vec1.shape} vs {vec2.shape}")
             return 0.0
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0 # Avoid division by zero
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def create_chunks(
        self,
        text: str,
        query: str, 
        instruction_prompt: str, 
        token_counter
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
        # Calculate available budget for each chunk
        query_tokens = self._get_token_length(query, token_counter)
        instruction_tokens = self._get_token_length(instruction_prompt, token_counter)
        
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

        # Estimate character budget (assuming ~4 chars/token)
        # This is a rough approximation. Consider making this configurable.
        char_budget = chunk_budget * 4 

        # Clean the text slightly
        text = re.sub(r'\n+', '\n', text) # Keep single newlines
        text = re.sub(r'[ \t]+', ' ', text) # Consolidate whitespace

        chunks = []
        start_index = 0
        text_length = len(text)

        while start_index < text_length:
            # Determine the end index for the chunk
            end_index = min(start_index + char_budget, text_length)

            # Try to find a natural break point (newline or space) near the end_index
            # Search backwards from end_index
            if end_index < text_length: # Don't adjust if it's the very end
                break_point = -1
                # Look for newline first, then space, within a reasonable window (e.g., 50 chars)
                newline_pos = text.rfind('\n', max(start_index, end_index - 50), end_index)
                space_pos = text.rfind(' ', max(start_index, end_index - 50), end_index)

                if newline_pos > start_index:
                    break_point = newline_pos + 1 # Include the newline in the previous chunk usually feels better
                elif space_pos > start_index:
                    break_point = space_pos + 1 # Split after the space

                # If a break point is found, adjust end_index
                if break_point != -1:
                    end_index = break_point
                # If no natural break found nearby, just cut at char_budget
                # (This case is handled by the initial end_index calculation)

            # Extract the chunk
            chunk = text[start_index:end_index].strip()
            if chunk: # Avoid adding empty chunks
                chunks.append(chunk)

            # Move to the next chunk's start
            start_index = end_index
            # Skip potential leading whitespace for the next chunk
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
        initial_chunks = self.create_chunks(
            text=text,
            query=query,
            instruction_prompt=instruction_prompt,
            token_counter=token_counter
        )

        if not initial_chunks:
            return []

        # 2. Embed the query
        try:
            query_embedding = embedding_provider.embed(query)
            if not query_embedding:
                print("Warning: Failed to embed query. Skipping filtering.")
                return initial_chunks
        except Exception as e:
            print(f"Error embedding query: {e}. Skipping filtering.")
            return initial_chunks


        # 3. Embed and filter chunks
        relevant_chunks = []
        if verbose: print("\n--- Chunk Filtering Details ---")
        for i, chunk in enumerate(initial_chunks):
            try:
                chunk_embedding = embedding_provider.embed(chunk)
                if not chunk_embedding:
                    print(f"Warning: Failed to embed chunk. Skipping this chunk:\n{chunk[:100]}...")
                    continue # Skip chunk if embedding fails

                similarity = self._cosine_similarity(query_embedding, chunk_embedding)

                # 4. Keep chunk if similarity is above threshold
                if similarity >= similarity_threshold:
                    if verbose: print(f"  Chunk {i+1}/{len(initial_chunks)}: Similarity = {similarity:.4f} [KEPT]")
                    relevant_chunks.append(chunk)
                else:
                    if verbose: print(f"  Chunk {i+1}/{len(initial_chunks)}: Similarity = {similarity:.4f} [FILTERED]")

            except Exception as e:
                 print(f"Error processing chunk {i+1} embedding or similarity: {e}. Skipping chunk:\n{chunk[:100]}...")
                 if verbose: print(f"  Chunk {i+1}/{len(initial_chunks)}: [ERROR DURING PROCESSING]")
                 continue # Skip chunk on error

        if verbose: print("--- End Chunk Filtering ---")
        print(f"Filtered chunks: Kept {len(relevant_chunks)} out of {len(initial_chunks)} based on similarity threshold {similarity_threshold}")
        return relevant_chunks
