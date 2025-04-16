"""
Gemini LLM Provider: Implements BaseLLMProvider for text generation.
"""
from google import genai
from ..base_llm_provider import BaseLLMProvider
import os

class GeminiLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None):
        """
        Gemini LLM provider. Pass api_key directly or set GEMINI_API_KEY in environment.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or constructor argument")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return ""

    def count_tokens(self, text: str) -> int:
        try:
            return len(text) // 4
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return int(len(text.split()) / 0.75)
