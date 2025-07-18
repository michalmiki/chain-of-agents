"""
Manager agent for the Chain of Agents implementation.
"""
from typing import Optional


class ManagerAgent:
    """
    Manager agent that processes the final communication unit and generates the answer.
    """
    
    def __init__(self, llm_provider=None, model=None, enable_thinking: bool = False):
        """
        Initialize the manager agent.
        Args:
            llm_provider: An object implementing BaseLLMProvider for text generation.
            model: (DEPRECATED) legacy argument for backward compatibility.
        """
        if model is not None:
            print("[DEPRECATION WARNING] Use 'llm_provider' instead of 'model' for ManagerAgent.")
            llm_provider = model
        self.llm_provider = llm_provider
        if hasattr(self.llm_provider, "enable_thinking"):
            self.llm_provider.enable_thinking = enable_thinking
        if not hasattr(self.llm_provider, 'generate') or not callable(self.llm_provider.generate):
            raise TypeError(f"llm_provider must implement a callable 'generate' method. Got: {type(self.llm_provider).__name__}")
    
    def _create_query_based_prompt(
        self, 
        final_cu: str, 
        query: str,
        task_desc: str = "Generate a comprehensive answer to the question based on the provided information."
    ) -> str:
        """
        Create a prompt for query-based tasks (e.g., question answering).
        
        Args:
            final_cu: The final communication unit from the last worker.
            query: The query or question.
            task_desc: Task-specific instruction.
            
        Returns:
            The formatted prompt.
        """
        prompt = f"""{task_desc}

The following are given passages. However, the source text is too long and has been summarized.
You need to answer based on the summary:

{final_cu}

Question: {query}

Answer:
"""
        return prompt
    
    def _create_non_query_based_prompt(
        self, 
        final_cu: str,
        task_desc: str = "Generate a comprehensive summary based on the provided information."
    ) -> str:
        """
        Create a prompt for non-query-based tasks (e.g., summarization).
        
        Args:
            final_cu: The final communication unit from the last worker.
            task_desc: Task-specific instruction.
            
        Returns:
            The formatted prompt.
        """
        prompt = f"""{task_desc}

The following are given passages. However, the source text is too long and has been summarized.
You need to generate a final summary based on the provided summary:

{final_cu}

Summary:
"""
        return prompt
    
    def generate_answer(
        self, 
        final_cu: str, 
        query: Optional[str] = None,
        task_desc: str = None,
        is_query_based: bool = True
    ) -> str:
        """
        Generate the final answer based on the final communication unit.
        
        Args:
            final_cu: The final communication unit from the last worker.
            query: The query or question (for query-based tasks).
            task_desc: Task-specific instruction.
            is_query_based: Whether this is a query-based task.
            
        Returns:
            The generated answer.
        """
        if is_query_based:
            if not query:
                raise ValueError("Query is required for query-based tasks")
            prompt = self._create_query_based_prompt(
                final_cu=final_cu,
                query=query,
                task_desc=task_desc or "Generate a comprehensive answer to the question based on the provided information."
            )
        else:
            prompt = self._create_non_query_based_prompt(
                final_cu=final_cu,
                task_desc=task_desc or "Generate a comprehensive summary based on the provided information."
            )
        
        # Generate answer
        answer = self.llm_provider.generate(prompt)
        return answer
