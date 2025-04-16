"""
Example script for using Chain of Agents for question answering.
"""
import sys
import os
import argparse
from markitdown import MarkItDown
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Add the parent directory to sys.path to import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_agents import ChainOfAgents
from chain_of_agents.providers.llm.gemini_llm import GeminiLLMProvider
from chain_of_agents.providers.embedding.gemini_embedding import GeminiEmbeddingProvider


def main(args):
    # Load the long text - our paper PDF
    print("Loading the long text...")
    md = MarkItDown() # Set to True to enable plugins
    result = md.convert("examples/data/19782_Chain_of_Agents_Large_La.pdf")
    long_text = result.text_content
    print("Long text loaded successfully.")

    # Get query from args
    query = args.query
    print(f"Using Query: \"{query}\"")

    # Initialize Chain of Agents with args
    print(f"Initializing ChainOfAgents...")
    print(f"  Use Embedding Filter: {args.use_filter}")
    if args.use_filter:
        print(f"  Similarity Threshold: {args.threshold}")

    coa = ChainOfAgents(
        llm_provider=GeminiLLMProvider(model_name=args.gemini_llm, api_key=GEMINI_API_KEY),
        embedding_provider=GeminiEmbeddingProvider(model_name=args.embedding_model, api_key=GEMINI_API_KEY),
        verbose=True,
        token_budget= 10000,
        show_worker_output=True,
        use_embedding_filter=args.use_filter,
        similarity_threshold=args.threshold
    )

    # Process the query
    result = coa.query(long_text, query)
    
    # Print the result
    print("\n==== Final Answer ====")
    print(result["answer"])
    print("\n==== Metadata ====")
    print(f"Number of chunks: {result['metadata']['num_chunks']}")
    print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")
    if args.use_filter:
        print(f"Filtering Used: Yes (Threshold: {args.threshold})")
    else:
        print("Filtering Used: No")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chain of Agents QA on a long text with optional embedding filtering.")
    parser.add_argument('--gemini-llm', type=str, default='gemini-2.0-flash', help='Gemini LLM model name')
    parser.add_argument('--embedding-model', type=str, default='gemini-embedding-exp-03-07', help='Gemini embedding model name')
    parser.add_argument(
        '--use-filter',
        action='store_true',
        help='Enable embedding-based chunk filtering.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Cosine similarity threshold for chunk filtering (requires --use-filter).'
    )
    parser.add_argument(
        '--query',
        type=str,
        default="What does the 'CU' stands for?",
        help='The question to ask the Chain of Agents.'
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
