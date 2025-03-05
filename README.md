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

This implementation uses Google's Gemini 2.0 Flash model as the underlying LLM for both worker and manager agents.

## Features

- Support for both query-based tasks (e.g., question answering) and non-query-based tasks (e.g., summarization)
- Efficient chunking of long texts based on token budget
- Sequential processing of chunks with information passing between agents
- Customizable task descriptions and prompts
- Detailed metadata and logging

## Project Structure

```
chain-of-agent/
├── chain-of-agents.md       # Paper summary
├── README.md                # This file
├── requirements.txt         # Dependencies
├── pyproject.toml           # Package configuration
├── src/                     # Source code
│   ├── agents/              # Worker and manager agent implementations
│   ├── chunking/            # Text chunking logic
│   ├── models/              # Model interfaces
│   └── chain_of_agents.py   # Main coordinator
├── tests/                   # Unit and integration tests
└── examples/                # Example scripts
    ├── question_answering.py
    ├── summarization.py
    └── data/                # Example datasets
```

## Installation

### Prerequisites

- Python 3.8 or higher
- A Gemini API key (obtain one from [Google AI Studio](https://ai.google.dev/))

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

3. Set up an `.env.prod` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
   
   Alternatively, set it as an environment variable:
   ```bash
   export GEMINI_API_KEY=your-api-key-here
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

# Initialize Chain of Agents with custom parameters
coa = ChainOfAgents(
    token_budget=8000,     # Tokens per chunk
    model_name="gemini-2.0-flash",
    verbose=True
)

# Process a document with a query
result = coa.query(
    text="Your long document here...",
    query="What are the main points discussed?"
)

print(result["answer"])
```

### Question Answering

```python
# If installed via pip
from chain_of_agents import ChainOfAgents

# If installed from source
# from src.chain_of_agents import ChainOfAgents

# Initialize Chain of Agents
coa = ChainOfAgents(
    token_budget=8000,     # Tokens per chunk
    model_name="gemini-2.0-flash",
    verbose=True
)

# Process a query
result = coa.query(
    text="Your long text here...",
    query="Your question here?"
)

# Get the answer
print(result["answer"])
```

### Summarization

```python
# If installed via pip
from chain_of_agents import ChainOfAgents

# If installed from source
# from src.chain_of_agents import ChainOfAgents

# Initialize Chain of Agents
coa = ChainOfAgents(
    token_budget=8000,     # Tokens per chunk
    model_name="gemini-2.0-flash",
    verbose=True
)

# Generate a summary
result = coa.summarize(
    text="Your long text here..."
)

# Get the summary
print(result["answer"])
```

### Advanced Configuration

```python
# Custom chunking strategy
coa = ChainOfAgents(
    token_budget=10000,
    chunk_overlap=200,     # Token overlap between chunks
    model_name="gemini-2.0-flash",
)
```

## Examples

The repository includes example scripts for both question answering and summarization:

```
python examples/question_answering.py
python examples/summarization.py
```

## Key Considerations

- **Token Budget**: The implementation uses a default token budget of 12,000 tokens per chunk, which can be adjusted when initializing the `ChainOfAgents` class.
- **Performance**: The effectiveness of the Chain of Agents depends on the quality of the summaries generated by the worker agents and the ability of the manager agent to synthesize the final output.
- **Latency**: Multiple sequential calls to the LLM API can result in higher latency compared to single-call approaches.

## Troubleshooting

### Common Issues

- **API Key Error**: If you encounter authentication issues, ensure your Gemini API key is correctly set in `.env.prod` or as an environment variable.
  
- **Token Limit Errors**: If you receive errors about exceeding token limits, try reducing the `token_budget` parameter when initializing the `ChainOfAgents`.
  
- **NLTK Resource Errors**: If you see errors related to NLTK resources, run the setup script manually:
  ```python
  import nltk
  nltk.download('punkt')
  ```

- **Memory Issues**: For very large documents, you may encounter memory issues. Try processing smaller portions of the document or increase your system's available memory.

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
