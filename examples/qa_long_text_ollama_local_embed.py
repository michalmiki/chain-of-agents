"""
Example script for using Chain of Agents with Ollama for QA and local Sentence Transformers embeddings.
"""
import sys
import os
import argparse
from markitdown import MarkItDown

# Add the parent directory to sys.path to import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_agents import ChainOfAgents
from chain_of_agents.providers.llm.ollama_llm import OllamaLLMProvider
from chain_of_agents.providers.embedding.sentence_transformer_embedding import SentenceTransformerEmbeddingProvider

def main(args):
    # Load the long text (PDF)
    print("Loading the long text...")
    md = MarkItDown()
    result = md.convert("examples/data/19782_Chain_of_Agents_Large_La.pdf")
    long_text = result.text_content
    print("Long text loaded successfully.")

    # Get query from args
    query = args.query
    print(f"Using Query: \"{query}\"")

    embedding_provider = None
    if args.use_local_embeddings:
        print(f"Using local SentenceTransformerEmbeddingProvider for embeddings: {args.embedding_model}")
        embedding_provider = SentenceTransformerEmbeddingProvider(model_name=args.embedding_model)

    print(f"Initializing ChainOfAgents with OllamaLLMProvider (model: {args.ollama_model})...")
    print(f"  Use Embedding Filter: {args.use_filter}")
    if args.use_filter:
        print(f"  Similarity Threshold: {args.threshold}")

    coa = ChainOfAgents(
        llm_provider=OllamaLLMProvider(model_name=args.ollama_model),
        embedding_provider=embedding_provider,
        verbose=True,
        token_budget=10000,
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
    parser = argparse.ArgumentParser(description="Run Chain of Agents QA on a long text with Ollama and optional local embedding model.")
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
    parser.add_argument(
        '--ollama-model',
        type=str,
        default="llama2",
        help='Name of the Ollama model to use for generation.'
    )
    parser.add_argument(
        '--use-local-embeddings',
        action='store_true',
        help='Use a local Sentence Transformers model for embeddings instead of the default.'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default="all-MiniLM-L6-v2",
        help='Name or path of the local Sentence Transformer embedding model (only used if --use-local-embeddings is set).'
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
