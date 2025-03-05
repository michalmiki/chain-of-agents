"""
Chunking module for the Chain of Agents implementation.
"""
import re
import nltk
from typing import List, Tuple, Optional
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Chunker:
    """
    Splits long texts into manageable chunks according to token budget.
    Implements the chunking strategy described in the Chain of Agents paper.
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
    
    def split_text_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: The text to split.
            
        Returns:
            A list of sentences.
        """
        # Clean the text
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        return sentences
    
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
        
        # Split text into sentences
        sentences = self.split_text_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self._get_token_length(sentence, token_counter)
            
            # If adding this sentence would exceed the budget, finalize the current chunk
            if current_length + sentence_tokens > chunk_budget and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
