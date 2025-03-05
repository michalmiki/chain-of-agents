"""
Example script for using Chain of Agents for question answering.
"""
import sys
import os

# Add the parent directory to sys.path to import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chain_of_agents import ChainOfAgents


def main():
    # Sample long text for QA
    long_text = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

    The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

    Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when obtaining funding for AI projects was difficult.

    In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began.

    Interest in neural networks and "connectionism" was revived by David Rumelhart and others in the middle of the 1980s. Artificial neural networks are modeled on the brain's neural structure, and its advent marked a paradigm shift in the field of AI from high-level symbolic reasoning to low-level machine learning.

    In the 1990s and early 21st century, AI achieved its greatest successes, albeit somewhat behind the scenes. Artificial intelligence is used for logistics, data mining, medical diagnosis and many other areas throughout the technology industry. The success was due to increasing computational power (see Moore's law), greater emphasis on solving specific problems, new ties between AI and other fields working on similar problems, and a new commitment by researchers to solid mathematical methods and rigorous scientific standards.

    Deep learning began to make an impact c. 2010. Deep learning is a branch of machine learning that makes use of multi-layer neural networks. It has enhanced abilities to detect and utilize patterns and to learn from large amounts of data, which works well for image detection, speech recognition, and has been used extensively in autonomous vehicles. In a 2017 survey, 26% of respondents said that AI has a "long way to go" before it reaches a high level and 28% said that AI is "just getting started." But 42% said that AI is "somewhat functional" or better. Despite GPT, DALL-E and other advances c. 2022, deep learning still has significant limitations, such as a lack of straightforward methods for incorporating common-sense reasoning, limitations of the "training data" approach, and issues with transparency and an inability to explain reasoning processes.
    """
    
    # Sample query
    query = "What were the major milestones in AI development since the 1980s?"
    
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
