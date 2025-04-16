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
    def __init__(self, model_name: str = "llama3.2"):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama package is not installed. Please install it with 'pip install ollama'"
            )
        load_dotenv('/Users/mikemik/Documents/Projects/Python/qki_analytics/chain-of-agent/.env.prod')
        self.model_name = model_name

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
