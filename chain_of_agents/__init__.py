"""
Chain of Agents (CoA) implementation.
"""
from typing import Optional, List, Dict, Any, Callable
import time

from chain_of_agents.chunking.chunker import Chunker
from chain_of_agents.agents.worker_agent import WorkerAgent
from chain_of_agents.agents.manager_agent import ManagerAgent


class ChainOfAgents:
    """
    Chain of Agents (CoA) implementation for processing long contexts.
    This class coordinates the chunking of input text, the sequential processing
    by worker agents, and the final answer generation by the manager agent.
    """
    
    def __init__(
        self,
        # core providers
        llm_provider=None,
        embedding_provider=None,
        # optional separate providers per agent
        worker_llm_provider=None,
        manager_llm_provider=None,
        token_budget: int = 12000,
        # per-agent thinking controls
        worker_thinking: bool = False,
        manager_thinking: bool = False,
        # fallback global thinking flag
        enable_thinking: bool = False,
        verbose: bool = False,
        show_worker_output: bool = False,
        use_embedding_filter: bool = False,
        similarity_threshold: float = 0.7,
        # The following args are DEPRECATED and only here for backward compatibility:
        model_name: str = None,
        ollama: bool = False,
        embedding_model: 'BaseModel' = None
    ):
        """
        Initialize the Chain of Agents with explicit LLM and embedding providers.
        enable_thinking: If True and the underlying LLM provider supports it, request chain-of-thought from worker and manager agents.
        Args:
            llm_provider: Object implementing BaseLLMProvider for generation.
            embedding_provider: Object implementing BaseEmbeddingProvider for embeddings.
            token_budget: The token budget for each chunk.
            verbose: Whether to print verbose output.
            show_worker_output: Whether to print worker agent outputs in verbose mode.
            use_embedding_filter: Whether to filter chunks based on embedding similarity to the query.
            similarity_threshold: The cosine similarity threshold for filtering chunks.
            model_name, ollama, embedding_model: DEPRECATED. Use explicit providers instead.
        """
        # DEPRECATED: model_name, ollama, embedding_model are no longer supported.
        if model_name or ollama or embedding_model:
            raise ValueError("[DEPRECATION ERROR] 'model_name', 'ollama', and 'embedding_model' are no longer supported. Please use 'llm_provider' and 'embedding_provider' arguments with provider objects instead.")
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.chunker = Chunker(token_budget=token_budget)

        # Resolve providers per agent
        worker_provider = worker_llm_provider or llm_provider
        manager_provider = manager_llm_provider or llm_provider

        self.worker_agent = WorkerAgent(
            llm_provider=worker_provider,
            enable_thinking=(worker_thinking or enable_thinking)
        )
        self.manager_agent = ManagerAgent(
            llm_provider=manager_provider,
            enable_thinking=(manager_thinking or enable_thinking)
        )
        self.verbose = verbose
        self.show_worker_output = show_worker_output
        self.use_embedding_filter = use_embedding_filter
        self.similarity_threshold = similarity_threshold
        self.enable_thinking = enable_thinking
        # Check if filtering is requested but provider doesn't support embedding
        provider_for_embedding = self.embedding_provider or self.llm_provider
        if self.use_embedding_filter and not hasattr(provider_for_embedding, 'embed'):
            print(f"Warning: 'use_embedding_filter' is True, but the selected embedding provider ({type(provider_for_embedding).__name__}) does not implement 'embed'. Filtering will be disabled.")
            self.use_embedding_filter = False

    def _log(self, message: str):
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def process(
        self, 
        text: str, 
        query: Optional[str] = None,
        worker_task_desc: Optional[str] = None,
        manager_task_desc: Optional[str] = None,
        is_query_based: bool = True
    ) -> Dict[str, Any]:
        """
        Process a long text using the Chain of Agents approach.
        
        Args:
            text: The long text to process.
            query: The query or question (for query-based tasks).
            worker_task_desc: Task description for worker agents.
            manager_task_desc: Task description for the manager agent.
            is_query_based: Whether this is a query-based task.
            
        Returns:
            A dictionary containing the final answer and metadata.
        """
        start_time = time.time()
        
        # Create worker instruction prompt
        worker_instruction = worker_task_desc or (
            "Answer the question based on the provided text." if is_query_based 
            else "Generate a summary of the provided text."
        )
        
        # Create chunks (potentially filtering them)
        self._log("Creating chunks...")
        provider_for_embedding = self.embedding_provider or self.llm_provider
        if is_query_based and self.use_embedding_filter:
            self._log(f"Filtering chunks with similarity threshold: {self.similarity_threshold}")
            chunks = self.chunker.create_and_filter_chunks(
                text=text,
                query=query,  # Query must exist for query-based filtering
                instruction_prompt=worker_instruction,
                embedding_provider=provider_for_embedding,  # Use embedding provider if provided
                llm_provider=self.llm_provider,  # Use LLM provider for token counting
                similarity_threshold=self.similarity_threshold,
                verbose=self.verbose  # Pass verbose flag for detailed logging
            )
        else:
            chunks = self.chunker.create_chunks(
                text=text,
                query=query or "",
                instruction_prompt=worker_instruction,
                token_counter=self.llm_provider.count_tokens,
                tokenizer=getattr(self.llm_provider, "tokenizer", None),
            )
        
        if self.use_embedding_filter and not is_query_based:
            self._log("Note: Embedding filter is enabled but task is not query-based. Skipping filtering.")
            self._log(f"Created {len(chunks)} chunks (no filtering applied).")

        if not chunks:
             self._log("No chunks were created or remained after filtering. Cannot proceed.")
             return {
                 "answer": "Could not process the text as no relevant chunks were found or created.",
                 "metadata": {
                     "num_chunks": 0,
                     "processing_time": time.time() - start_time,
                     "communication_units": [],
                     "final_cu": None
                 }
             }

        # Process chunks with worker agents
        self._log("Processing chunks with worker agents...")
        communication_units = []
        previous_cu = None
        
        for i, chunk in enumerate(chunks):
            self._log(f"Processing chunk {i+1}/{len(chunks)}...")
            cu = self.worker_agent.process_chunk(
                chunk=chunk,
                previous_cu=previous_cu,
                query=query,
                task_desc=worker_task_desc,
                is_query_based=is_query_based
            )
            communication_units.append(cu)
            previous_cu = cu
            self._log(f"Processed chunk {i+1}/{len(chunks)}.")
            
            # Show worker output if enabled
            if self.verbose and self.show_worker_output:
                self._log(f"\n==== Worker {i+1} Output ====\n{cu}\n")
        
        # Generate final answer with manager agent
        self._log("Generating final answer with manager agent...")
        final_cu = communication_units[-1]
        answer = self.manager_agent.generate_answer(
            final_cu=final_cu,
            query=query,
            task_desc=manager_task_desc,
            is_query_based=is_query_based
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "answer": answer,
            "metadata": {
                "num_chunks": len(chunks),
                "processing_time": processing_time,
                "communication_units": communication_units,
                "final_cu": final_cu
            }
        }
    
    def query(
        self, 
        text: str, 
        query: str,
        worker_task_desc: Optional[str] = None,
        manager_task_desc: Optional[str] = None,
        show_worker_output: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process a query-based task (e.g., question answering).
        
        Args:
            text: The long text to process.
            query: The query or question.
            worker_task_desc: Task description for worker agents.
            manager_task_desc: Task description for the manager agent.
            
        Returns:
            A dictionary containing the final answer and metadata.
        """
        # Use the provided show_worker_output or the default from initialization
        if show_worker_output is not None:
            old_show_worker_output = self.show_worker_output
            self.show_worker_output = show_worker_output
        
        result = self.process(
            text=text,
            query=query,
            worker_task_desc=worker_task_desc or "Answer the question based on the provided text.",
            manager_task_desc=manager_task_desc or "Generate a comprehensive answer to the question based on the provided information.",
            is_query_based=True
        )
        
        # Restore the original setting if it was temporarily changed
        if show_worker_output is not None:
            self.show_worker_output = old_show_worker_output
            
        return result
    
    def summarize(
        self, 
        text: str,
        worker_task_desc: Optional[str] = None,
        manager_task_desc: Optional[str] = None,
        show_worker_output: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process a non-query-based task (e.g., summarization).
        
        Args:
            text: The long text to process.
            worker_task_desc: Task description for worker agents.
            manager_task_desc: Task description for the manager agent.
            
        Returns:
            A dictionary containing the final answer and metadata.
        """
        # Use the provided show_worker_output or the default from initialization
        if show_worker_output is not None:
            old_show_worker_output = self.show_worker_output
            self.show_worker_output = show_worker_output
        
        result = self.process(
            text=text,
            query=None,
            worker_task_desc=worker_task_desc or "Generate a summary of the provided text.",
            manager_task_desc=manager_task_desc or "Generate a comprehensive summary based on the provided information.",
            is_query_based=False
        )
        
        # Restore the original setting if it was temporarily changed
        if show_worker_output is not None:
            self.show_worker_output = old_show_worker_output
            
        return result
