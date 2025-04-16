"""
Ollama LLM Provider: Implements BaseLLMProvider for text generation.
"""
from dotenv import load_dotenv
from ..base_llm_provider import BaseLLMProvider
import os

try:
    from ollama import chat, ChatResponse
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class OllamaLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "deepseek-llm:7b-instruct", base_url: str = "http://localhost:11434"):
        """
        Ollama LLM provider. Pass model_name and base_url directly. No API key required.
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama package is not installed. Please install it with 'pip install ollama'"
            )
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            messages = [
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
            options = {
                'temperature': temperature,
                'num_predict': max_tokens
            }
            kwargs = {'model': self.model_name, 'messages': messages, 'options': options}
            response: ChatResponse = chat(**kwargs)
            return response.message.content
        except Exception as e:
            print(f"Error generating content with Ollama: {e}")
            return ""

    def count_tokens(self, text: str) -> int:
        return len(text) // 4
