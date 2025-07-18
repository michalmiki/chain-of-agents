"""
Worker agent for the Chain of Agents implementation.
"""
from typing import Optional, Dict, Any


class WorkerAgent:
    """
    Worker agent that processes chunks of text and generates communication units.
    """
    
    def __init__(self, llm_provider=None, model=None, enable_thinking: bool = False):
        """
        Initialize the worker agent.
        Args:
            llm_provider: An object implementing BaseLLMProvider for text generation.
            model: (DEPRECATED) legacy argument for backward compatibility.
        """
        if model is not None:
            print("[DEPRECATION WARNING] Use 'llm_provider' instead of 'model' for WorkerAgent.")
            llm_provider = model
        self.llm_provider = llm_provider
        # Configure provider-level thinking flag if supported
        if hasattr(self.llm_provider, "enable_thinking"):
            self.llm_provider.enable_thinking = enable_thinking
        if not hasattr(self.llm_provider, 'generate') or not callable(self.llm_provider.generate):
            raise TypeError(f"llm_provider must implement a callable 'generate' method. Got: {type(self.llm_provider).__name__}")
    
    def _create_query_based_prompt(
        self, 
        chunk: str, 
        previous_cu: Optional[str], 
        query: str,
        task_desc: str = "Answer the question based on the provided text."
    ) -> str:
        """
        Create a prompt for query-based tasks (e.g., question answering).
        
        Args:
            chunk: The current chunk of text to process.
            previous_cu: The communication unit from the previous worker, if any.
            query: The query or question.
            task_desc: Task-specific instruction.
            
        Returns:
            The formatted prompt.
        """
        prev_cu_text = f"Here is the summary of the previous source text: {previous_cu}" if previous_cu else "There is no previous source text."
        
        prompt = f"""{task_desc}

Input Chunk:
{chunk}

{prev_cu_text}

Question: {query}

You need to read the current source text and summary of previous source text (if any) and generate a summary to include them both.
Later, this summary will be used for other agents to answer the Question. So please write the summary that can include the evidence for answering the Question.
"""
        return prompt
    
    def _create_non_query_based_prompt(
        self, 
        chunk: str, 
        previous_cu: Optional[str],
        task_desc: str = "Generate a summary of the provided text."
    ) -> str:
        """
        Create a prompt for non-query-based tasks (e.g., summarization).
        
        Args:
            chunk: The current chunk of text to process.
            previous_cu: The communication unit from the previous worker, if any.
            task_desc: Task-specific instruction.
            
        Returns:
            The formatted prompt.
        """
        prev_cu_text = f"Here is the summary of the previous source text: {previous_cu}" if previous_cu else "There is no previous source text."
        
        prompt = f"""{task_desc}

Input Chunk:
{chunk}

{prev_cu_text}

You need to read the current source text and summary of previous source text (if any) and generate a summary to include them both.
Later, this summary will be used for other agents to generate a summary for the whole text.
Thus, your generated summary should be relatively long and include all important information.
"""
        return prompt
    
    def process_chunk(
        self, 
        chunk: str, 
        previous_cu: Optional[str] = None, 
        query: Optional[str] = None,
        task_desc: str = None,
        is_query_based: bool = True
    ) -> str:
        """
        Process a chunk of text and generate a communication unit.
        
        Args:
            chunk: The chunk of text to process.
            previous_cu: The communication unit from the previous worker, if any.
            query: The query or question (for query-based tasks).
            task_desc: Task-specific instruction.
            is_query_based: Whether this is a query-based task.
            
        Returns:
            The generated communication unit.
        """
        if is_query_based:
            if not query:
                raise ValueError("Query is required for query-based tasks")
            prompt = self._create_query_based_prompt(
                chunk=chunk,
                previous_cu=previous_cu,
                query=query,
                task_desc=task_desc or "Answer the question based on the provided text."
            )
        else:
            prompt = self._create_non_query_based_prompt(
                chunk=chunk,
                previous_cu=previous_cu,
                task_desc=task_desc or "Generate a summary of the provided text."
            )
        
        # Generate communication unit
        communication_unit = self.llm_provider.generate(prompt)
        return communication_unit
