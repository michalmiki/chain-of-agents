"""
Ollama LLM Provider: Implements BaseLLMProvider for text generation.
"""
from dotenv import load_dotenv
from ..base_llm_provider import BaseLLMProvider
import os

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency declared but guard for robustness
    tiktoken = None

try:
    from ollama import chat, ChatResponse
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class OllamaLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "deepseek-llm:7b-instruct", base_url: str = "http://localhost:11434", enable_thinking: bool = False, context_window: int = 32768):
        """
        Ollama LLM provider. Pass model_name and base_url directly. No API key required.
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama package is not installed. Please install it with 'pip install ollama'"
            )
        self.model_name = model_name
        self.base_url = base_url
        # Whether to request the model's chain-of-thought via Ollama's `think` flag
        self.enable_thinking = enable_thinking
        # Context window to send to Ollama (num_ctx option)
        self.context_window = context_window
        # Store the last raw "thinking" text (if any) for debugging purposes
        self.last_thinking: str = ""
        self.tokenizer = None
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except Exception:
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.tokenizer = None

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 8192) -> str:
        try:
            messages = [
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
                'num_ctx': self.context_window,
            }
            kwargs = {
                'model': self.model_name,
                'messages': messages,
                'options': options,
                'think': self.enable_thinking,
            }
            response = chat(**kwargs)

            # Ollama >= 0.2.0 returns a dict-like object; earlier versions return ChatResponse
            content = ""
            thinking_text = ""
            try:
                # Newer: dict style
                if isinstance(response, dict):
                    message = response.get("message", {})
                    content = message.get("content", "")
                    thinking_text = message.get("thinking", "")
                else:
                    # Fallback ChatResponse
                    content = response.message.content
                    thinking_text = getattr(response.message, "thinking", "")
            except Exception as parse_err:
                print(f"Warning: could not parse Ollama response: {parse_err}")
                content = str(response)

            # Persist thinking for external inspection if enabled
            if self.enable_thinking:
                self.last_thinking = thinking_text
            return content
        except Exception as e:
            print(f"Error generating content with Ollama: {e}")
            return ""

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return len(text.split()) if text else 0
