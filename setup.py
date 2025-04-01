"""
Chain of Agents
--------------
A Python implementation of the Chain of Agents framework for processing long-context tasks
using large language models.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chain_of_agents",
    version="0.1.0",
    author="QKI Analytics",
    author_email="your.email@example.com",
    description="Chain of Agents framework for processing long-context tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chain-of-agents",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "google-generativeai>=0.3.0",
        "python-dotenv>=1.0.0",
        # nltk removed
        "tiktoken>=0.5.0",
        "ollama>=0.1.5",
        "numpy>=1.21.0", # Added numpy for cosine similarity in chunker
    ],
    entry_points={
        "console_scripts": [
            # coa-setup-nltk removed
        ],
    },
)
