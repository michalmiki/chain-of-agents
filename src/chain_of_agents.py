"""
Chain of Agents (CoA) implementation.
"""
from typing import Optional, List, Dict, Any, Callable
import time

from src.models.gemini_model import GeminiModel
from src.chunking.chunker import Chunker
from src.agents.worker_agent import WorkerAgent
from src.agents.manager_agent import ManagerAgent


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
        verbose: bool = False
    ):
        """
        Initialize the Chain of Agents.
        
        Args:
            model_name: The name of the model to use.
            token_budget: The token budget for each chunk.
            verbose: Whether to print verbose output.
        """
        self.model = GeminiModel(model_name=model_name)
        self.chunker = Chunker(token_budget=token_budget)
        self.worker_agent = WorkerAgent(model=self.model)
        self.manager_agent = ManagerAgent(model=self.model)
        self.verbose = verbose
    
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
        
        # Create chunks
        self._log("Creating chunks...")
        chunks = self.chunker.create_chunks(
            text=text,
            query=query or "",
            instruction_prompt=worker_instruction,
            token_counter=self.model.count_tokens
        )
        self._log(f"Created {len(chunks)} chunks.")
        
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
        manager_task_desc: Optional[str] = None
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
        return self.process(
            text=text,
            query=query,
            worker_task_desc=worker_task_desc or "Answer the question based on the provided text.",
            manager_task_desc=manager_task_desc or "Generate a comprehensive answer to the question based on the provided information.",
            is_query_based=True
        )
    
    def summarize(
        self, 
        text: str,
        worker_task_desc: Optional[str] = None,
        manager_task_desc: Optional[str] = None
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
        return self.process(
            text=text,
            query=None,
            worker_task_desc=worker_task_desc or "Generate a summary of the provided text.",
            manager_task_desc=manager_task_desc or "Generate a comprehensive summary based on the provided information.",
            is_query_based=False
        )
