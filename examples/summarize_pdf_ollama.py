"""
Example script: Summarize a PDF document with Chain of Agents using an Ollama model.

This mirrors `summarize_pdf_gemini.py` but relies on a locally-served Ollama model. The
model name is read from the environment variable `OLLAMA_MODEL` (via `.env`) but can be
overridden with `--ollama-model`.

Usage (bash):
    python examples/summarize_pdf_ollama.py \
        --pdf-path examples/data/19782_Chain_of_Agents_Large_La.pdf \
        --ollama-model deepseek-r1:1.3b \
        --token-budget 10000 \
        --show-worker-output

Ollama must be running locally: https://github.com/ollama/ollama
"""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from markitdown import MarkItDown

# ---------------------------------------------------------------------------
# Load environment variables (expecting OLLAMA_MODEL, optional)
# ---------------------------------------------------------------------------
load_dotenv()
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")  # fallback model

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chain_of_agents import ChainOfAgents
from chain_of_agents.providers.llm.ollama_llm import OllamaLLMProvider


# ---------------------------------------------------------------------------
# Helper: convert PDF → plain text via MarkItDown
# ---------------------------------------------------------------------------

def pdf_to_text(pdf_path: str) -> str:
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"Reading and converting PDF: {pdf_path}")
    md = MarkItDown()
    result = md.convert(pdf_path)
    print("Conversion complete (PDF → text).")
    return result.text_content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarize a PDF with Chain of Agents (Ollama model)")
    parser.add_argument("--pdf-path", type=str, default="examples/data/19782_Chain_of_Agents_Large_La.pdf",
                        help="Path to the PDF file to summarise")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL,
                        help="Ollama model name (overrides $OLLAMA_MODEL)")
    parser.add_argument("--token-budget", type=int, default=10000,
                        help="Token budget for each chunk processed by ChainOfAgents")
    parser.add_argument("--show-worker-output", action="store_true",
                        help="Print intermediate summaries from worker agents")
    parser.add_argument("--worker-thinking", action="store_true",
                        help="Enable thinking in worker agents")
    parser.add_argument("--manager-thinking", action="store_true",
                        help="Enable thinking in manager agent")
    args = parser.parse_args()

    # Extract plain text from the PDF
    text = pdf_to_text(args.pdf_path)

    # Create separate LLM providers for workers and manager
    worker_provider = OllamaLLMProvider(model_name=args.ollama_model, enable_thinking=args.worker_thinking)
    manager_provider = OllamaLLMProvider(model_name=args.ollama_model, enable_thinking=args.manager_thinking)

    # Instantiate ChainOfAgents
    coa = ChainOfAgents(
        llm_provider=worker_provider,  # fallback default
        worker_llm_provider=worker_provider,
        manager_llm_provider=manager_provider,
        worker_thinking=args.worker_thinking,
        manager_thinking=args.manager_thinking,
        token_budget=args.token_budget,
        verbose=True,
        show_worker_output=args.show_worker_output,
    )

    # Summarize
    result = coa.summarize(text)

    # Output
    print("\n==== Final Summary ====")
    print(result["answer"])
    print("\n==== Metadata ====")
    print(f"Number of chunks: {result['metadata']['num_chunks']}")

    # Optional: show chain-of-thought if enabled
    if args.worker_thinking and getattr(worker_provider, "last_thinking", ""):
        print("\n==== Worker (last) chain-of-thought ====")
        print(worker_provider.last_thinking)
    if args.manager_thinking and getattr(manager_provider, "last_thinking", ""):
        print("\n==== Manager chain-of-thought ====")
        print(manager_provider.last_thinking)
    print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")


if __name__ == "__main__":
    main()
