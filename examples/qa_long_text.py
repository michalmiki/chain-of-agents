"""
Example script for using Chain of Agents for question answering.
"""
import sys
import os
from markitdown import MarkItDown


# Add the parent directory to sys.path to import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_agents import ChainOfAgents


def main():
    # Load the long text - our paper PDF
    print("Loading the long text...")
    md = MarkItDown() # Set to True to enable plugins
    result = md.convert("examples/data/19782_Chain_of_Agents_Large_La.pdf")
    long_text = result.text_content
    print("Long text loaded successfully.")
    
    # Sample query
    query = "What does the 'CU' stands for?"
    
    # Initialize Chain of Agents
    coa = ChainOfAgents(verbose=True, show_worker_output=True)
    
    # Process the query
    result = coa.query(long_text, query)
    
    # Print the result
    print("\n==== Final Answer ====")
    print(result["answer"])
    print("\n==== Metadata ====")
    print(f"Number of chunks: {result['metadata']['num_chunks']}")
    print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")


if __name__ == "__main__":
    main()
