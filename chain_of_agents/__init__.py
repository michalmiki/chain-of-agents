"""
Chain of Agents (CoA) implementation.
"""
from typing import Optional, List, Dict, Any, Callable
import time

from chain_of_agents.models.gemini_model import GeminiModel
from chain_of_agents.models.ollama_model import OllamaModel
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
        model_name: str = "gemini-2.0-flash",
        token_budget: int = 12000,
        verbose: bool = False,
        show_worker_output: bool = False,
        ollama: bool = False,
        use_embedding_filter: bool = False, # New: Flag to enable filtering
        similarity_threshold: float = 0.7   # New: Threshold for filtering
    ):
        """
        Initialize the Chain of Agents.
        
        Args:
            model_name: The name of the model to use.
            token_budget: The token budget for each chunk.
            verbose: Whether to print verbose output.
            show_worker_output: Whether to print worker agent outputs in verbose mode.
            ollama: Whether to use Ollama model instead of Gemini.
            use_embedding_filter: Whether to filter chunks based on embedding similarity to the query.
            similarity_threshold: The cosine similarity threshold for filtering chunks.
        """
        if ollama:
            self.model = OllamaModel(model_name=model_name)
        else:
            self.model = GeminiModel(model_name=model_name)
        self.chunker = Chunker(token_budget=token_budget)
        self.worker_agent = WorkerAgent(model=self.model)
        self.manager_agent = ManagerAgent(model=self.model)
        self.verbose = verbose
        self.show_worker_output = show_worker_output
        self.use_embedding_filter = use_embedding_filter
        self.similarity_threshold = similarity_threshold

        # Check if filtering is requested but model doesn't support embedding
        if self.use_embedding_filter and not hasattr(self.model, 'embed_content'):
             print(f"Warning: 'use_embedding_filter' is True, but the selected model ({type(self.model).__name__}) does not implement 'embed_content'. Filtering will be disabled.")
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
        if is_query_based and self.use_embedding_filter:
            self._log(f"Filtering chunks with similarity threshold: {self.similarity_threshold}")
            chunks = self.chunker.create_and_filter_chunks(
                text=text,
                query=query, # Query must exist for query-based filtering
                instruction_prompt=worker_instruction,
                model=self.model, # Pass the model for embedding
                similarity_threshold=self.similarity_threshold,
                verbose=self.verbose # Pass verbose flag for detailed logging
            )
            # The detailed logging is now inside create_and_filter_chunks
            # self._log(f"Created and filtered {len(chunks)} relevant chunks.") # Redundant now
        else:
            if is_query_based and self.use_embedding_filter:
                 self._log("Note: Embedding filter is enabled but task is not query-based. Skipping filtering.")
            chunks = self.chunker.create_chunks(
                text=text,
                query=query or "",
                instruction_prompt=worker_instruction,
                token_counter=self.model.count_tokens
            )
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
