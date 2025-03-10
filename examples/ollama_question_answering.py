#!/usr/bin/env python
"""
Example of using Chain of Agents with Ollama for question answering.
"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ChainOfAgents, OllamaModel

# Sample text for demonstration
SAMPLE_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go).

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
"""

def main():
    """Run the example."""
    # Create an instance of the Ollama model
    # You can specify a different model name if desired
    model = OllamaModel(model_name="llama3.2")
    
    # Create an instance of the Chain of Agents
    chain = ChainOfAgents(
        model=model,
        num_workers=3,
        show_worker_output=True  # Set to True to see worker agent outputs
    )
    
    # Process a query using the Chain of Agents
    query = "What is the AI effect and why does it happen?"
    result = chain.process_query(query=query, context=SAMPLE_TEXT)
    
    print("\n=== Final Answer ===")
    print(result)


if __name__ == "__main__":
    main()
