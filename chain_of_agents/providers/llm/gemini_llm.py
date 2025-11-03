"""
Gemini LLM Provider: Implements BaseLLMProvider for text generation.
"""
from google import genai
from ..base_llm_provider import BaseLLMProvider
import os

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency declared but guard for robustness
    tiktoken = None

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
        self.tokenizer = None
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except Exception:
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.tokenizer = None

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
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return len(text.split()) if text else 0
