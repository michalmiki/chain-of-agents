"""
OpenAI LLM Provider: Implements BaseLLMProvider for text generation using OpenAI's GPT models.
"""
from ..base_llm_provider import BaseLLMProvider
import os
from typing import Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed. Please install it with 'pip install openai'")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or as argument.")
        openai.api_key = self.api_key

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"Error generating content with OpenAI: {e}")
            return ""

    def count_tokens(self, text: str) -> int:
        # Approximate: 4 chars per token for English
        return len(text) // 4
