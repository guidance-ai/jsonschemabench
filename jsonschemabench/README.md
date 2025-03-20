[![ArXiv](https://img.shields.io/badge/arXiv-2501.10868-b31b1b)](https://arxiv.org/abs/2501.10868)
[![Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-orange)](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench)

JSONSchemaBench is a library for benchmarking language models on their ability to generate valid JSON according to JSON Schema specifications. This tool allows researchers and developers to evaluate different language models on structured generation tasks.

## Installation

To install the library, follow these steps:

1. Create a conda environment:
    ```bash
    conda create -n "jsonschemabench" python=3.12
    conda activate jsonschemabench
    ```

2. Install the core dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install engines libraries:
   ```bash
    # Install OpenAI and Gemini
    pip install openai
    pip install google-generativeai

    # Install Guidance
    pip install git+https://github.com/guidance-ai/guidance.git@514a5eb16b9d29ad824d9357732ba66e5e767642
    
    # Install llama-cpp-python with CUDA support
    CMAKE_ARGS="-DGGML_CUDA=on" pip install git+https://github.com/abetlen/llama-cpp-python.git
    
    # Install Outlines and XGrammar
    pip install git+https://github.com/dottxt-ai/outlines.git
    pip install git+https://github.com/mlc-ai/xgrammar.git
   ```

For GPU acceleration, ensure you have the appropriate CUDA drivers installed on your system.

## Documentation

- **[Quickstart Guide](docs/quickstart.md)**: Learn how to run benchmarks using the CLI or Python API
- **[Custom Engine Tutorial](docs/custom_engine.md)**: Detailed instructions for implementing your own engine

For examples of how to use different engines, check the [tests/all.py](tests/all.py) file.
