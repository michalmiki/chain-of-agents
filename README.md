# Chain of Agents Implementation

[![PyPI version](https://img.shields.io/pypi/v/chain-of-agents.svg)](https://pypi.org/project/chain-of-agents/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains a Python implementation of the "Chain of Agents" (CoA) approach for handling long-context tasks with Large Language Models (LLMs). CoA is based on the research paper "Chain of Agents: Large Language Models Collaborating on Long-Context Tasks."

For more details, you can read the [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://openreview.net/pdf?id=LuCLf4BJsr) paper.

## Overview

Chain of Agents is a novel framework that enables multiple LLMs to collaborate sequentially to process long texts. The approach consists of:

1. **Worker Agents**: A chain of LLMs that process chunks of text sequentially, with each agent building upon the previous agent's summary.
2. **Manager Agent**: An LLM that synthesizes the final output from the accumulated information passed through the worker chain.

This implementation uses explicit LLM and embedding providers. Built-in LLM providers include Gemini, OpenAI, and Ollama; embedding providers include OpenAI, SentenceTransformers, and Gemini.

## Features

- Support for both query-based tasks (e.g., question answering) and non-query-based tasks (e.g., summarization)
- Efficient chunking of long texts based on token budget
- Sequential processing of chunks with information passing between agents
- Customizable task descriptions and prompts
- Detailed metadata and logging
- Support for pluggable LLM providers (e.g., Gemini, OpenAI, Ollama) via the BaseLLMProvider interface
- Support for pluggable embedding providers (e.g., OpenAI, SentenceTransformers, Gemini) via the BaseEmbeddingProvider interface

## Project Structure

```
chain-of-agent/
├── .gitignore
├── build_package.sh
├── chain-of-agents.md       # Paper summary
├── LICENSE
├── MANIFEST.in
├── README.md                # This file
├── requirements.txt         # Core dependencies (also listed in setup.py)
├── setup_nltk.py            # NLTK data download script
├── setup.py                 # Package setup script
├── test_import.py           # Simple import test
├── build/                   # Build artifacts (generated)
├── chain_of_agents.egg-info/ # Build artifacts (generated)
├── examples/                # Example scripts
│   ├── ollama_question_answering.py
│   ├── qa_long_text.py
│   ├── question_answering.py
│   ├── summarization.py
│   ├── test_import.py       # Simple import test (from root)
│   └── data/                # Example data (e.g., PDFs)
└── chain_of_agents/         # Source code for the package
    ├── __init__.py          # Main ChainOfAgents class definition
    ├── agents/              # Worker and manager agent implementations
    │   ├── __init__.py
    │   ├── manager_agent.py
    │   └── worker_agent.py
    ├── chunking/            # Text chunking logic
    │   ├── __init__.py
    │   └── chunker.py
    ├── providers/           # LLM and embedding providers
    │   ├── __init__.py
    │   ├── llm/
    │   │   ├── __init__.py
    │   │   ├── base_llm.py
    │   │   ├── gemini_llm.py
    │   │   ├── openai_llm.py
    │   │   └── ollama_llm.py
    │   └── embedding/
    │       ├── __init__.py
    │       ├── base_embedding.py
    │       ├── openai_embedding.py
    │       ├── sentence_transformers_embedding.py
    │       └── gemini_embedding.py
    └── models/              # LLM model interfaces
        ├── __init__.py
        ├── base_model.py
        ├── gemini_model.py
        └── ollama_model.py
```

## Installation

### Prerequisites

- Python 3.8 or higher
- **Optional (for Gemini):** A Gemini API key (obtain one from [Google AI Studio](https://ai.google.dev/)). Set it as an environment variable `GEMINI_API_KEY`.
- **Optional (for Ollama):** A running Ollama instance. Ensure the Ollama server is accessible. You might need to set `OLLAMA_HOST` if it's not running on the default `http://localhost:11434`.

### Option 1: Install from PyPI

```bash
pip install chain-of-agents
```

### Option 2: Install from source

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/chain-of-agent.git
   cd chain-of-agent
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Set necessary environment variables:
   - **For Gemini:**
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```
   - **For Ollama:** Ensure your Ollama server is running. If it's not on the default host, you might need:
     ```bash
     # Example if Ollama is running elsewhere
     # export OLLAMA_HOST="http://your-ollama-server:11434" 
     ```
     The `OllamaModel` class uses the `ollama` library, which typically checks `OLLAMA_HOST` or defaults to localhost.

### Dependencies

The core dependencies are listed in `setup.py` and include:
- `google-generativeai`
- `python-dotenv`
- `nltk`
- `tiktoken`
- `ollama`
- `sentence-transformers`

**Optional:** For processing specific file types like PDFs (as shown in `examples/qa_long_text.py`), you might need additional libraries like `markitdown`:
```bash
pip install markitdown
```

If you want to run the PDF question answering example:
```bash
python examples/qa_long_text.py
```

### NLTK Setup

After installation, you need to download the required NLTK data:

```bash
# If installed via PyPI
coa-setup-nltk

# Or run the setup script directly
python setup_nltk.py
```

## Usage

### Basic Usage

```python
from chain_of_agents import ChainOfAgents
from chain_of_agents.providers.llm.openai_llm import OpenAILLMProvider
from chain_of_agents.providers.embedding.openai_embedding import OpenAIEmbeddingProvider

# Initialize providers
llm_provider = OpenAILLMProvider(model_name="gpt-3.5-turbo")
embedding_provider = OpenAIEmbeddingProvider(model_name="text-embedding-ada-002")

# Initialize Chain of Agents
coa = ChainOfAgents(
    llm_provider=llm_provider,
    embedding_provider=embedding_provider,
    token_budget=12000,
    verbose=True,
    show_worker_output=False,      # Set to True to see intermediate worker outputs
    use_embedding_filter=True,      # Set to True to filter chunks by embedding similarity
    similarity_threshold=0.75       # Cosine similarity threshold for filtering
)

# Query-based task (Question Answering)
long_text = "Your very long document text goes here..."
query = "What are the main points discussed in the document?"
result = coa.query(text=long_text, query=query)
print(result["answer"])

# Summarization
result_summary = coa.summarize(text=long_text)
print(result_summary["answer"])
```

### Question Answering

Both Gemini and Ollama instances can be used for question answering via the `.query()` method.

```python
# Using the initialized coa from above...

long_text = "..." # Your long text
question = "..." # Your question

# Process the query (example with Gemini instance)
result = coa.query(text=long_text, query=question)

# Get the answer
print(result["answer"])
```

### Summarization

Both Gemini and Ollama instances can be used for summarization via the `.summarize()` method.

```python
# Using the initialized coa from above...

long_text = "..." # Your long text

# Generate a summary (example with Ollama instance)
result = coa.summarize(text=long_text)

# Get the summary
print(result["answer"])
```

## Examples

The repository includes example scripts for both question answering and summarization:

```
python examples/question_answering.py
python examples/summarization.py
```

## Key Considerations

- **Model Selection**:
    - **Gemini (Default):** The package defaults to using Gemini models (`GeminiModel`). Specify the model name (e.g., `"gemini-1.5-flash"`) during `ChainOfAgents` initialization. Ensure your `GEMINI_API_KEY` environment variable is set.
    - **Ollama:** To use models served by Ollama, initialize `ChainOfAgents` with the `ollama=True` flag and provide the desired Ollama `model_name` (e.g., `"llama3"`). Ensure your Ollama server is running and accessible (check `OLLAMA_HOST` if needed).
- **Token Budget**: The `token_budget` parameter determines the approximate size of text chunks processed by worker agents. The default is 12,000 for Gemini. You might want to adjust this (e.g., lower it to 4000) when using Ollama models, depending on their context window size and the task complexity. This budget accounts for the chunk text, query, instructions, and estimated previous summary size.
- **Performance**: Effectiveness depends on the worker agents' ability to create useful summaries and the manager agent's synthesis capability.
- **Latency**: Sequential LLM calls increase overall processing time compared to single-call methods.

## Troubleshooting

### Common Issues

- **API Key Error (Gemini)**: Ensure your `GEMINI_API_KEY` environment variable is correctly set and exported.
- **Connection Error (Ollama)**: Ensure your Ollama server is running and accessible. Check the `OLLAMA_HOST` environment variable if it's not on localhost.
- **Token Limit Errors**: If the model rejects prompts due to length, try reducing the `token_budget` when initializing `ChainOfAgents`. This gives more room for prompts and intermediate summaries.
- **NLTK Resource Errors**: If you see `LookupError` related to `punkt`, ensure you ran the NLTK setup: `coa-setup-nltk` or `python setup_nltk.py`. If issues persist, try running `import nltk; nltk.download('punkt')` in a Python interpreter.
- **Memory Issues**: Processing very large documents can consume significant memory. Consider system resources or processing the document in smaller sections if needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the existing coding style.

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{chainofagents2023,
  title={Chain of Agents: Large Language Models Collaborating on Long-Context Tasks},
  author={Author Name et al.},
  journal={Conference/Journal Name},
  year={2023}
}
```

## Related Work

- [LangChain](https://github.com/hwchase17/langchain) - Framework for developing applications with LLMs
- [LlamaIndex](https://github.com/jerryjliu/llama_index) - Data framework for LLM applications
- [AutoGen](https://github.com/microsoft/autogen) - Framework for building agent-based systems

## Limitations

- The current implementation only supports text-based tasks.
- The effectiveness of the approach depends on the quality of the chunking strategy and the ability of worker agents to generate informative communication units.
- As the number of chunks increases, there might be some information loss in the chain.

## Future Improvements

- Support for multi-modal inputs and outputs
- Optimization of chunking strategies
- Fine-tuning of prompts for specific tasks
- Parallel processing of non-dependent chunks
- Integration with other LLM providers

## License

[MIT License](LICENSE)
