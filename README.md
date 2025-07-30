# LangChain & LangGraph Examples

This repository contains sample projects and examples for working with LangChain and LangGraph frameworks.

## Projects

### 🦜 LangChain with Ollama (`/langchain`)

A comprehensive example project demonstrating how to use LangChain with Ollama and the llama3.2:latest model. This project includes:

- **Simple Chat**: Basic question-answering with LangChain and Ollama
- **Document Q&A**: Retrieval-Augmented Generation (RAG) for document-based questions
- **Conversation Memory**: Maintaining context across multiple interactions
- **Streaming Chat**: Real-time response streaming for better user experience

**Key Features:**
- Local LLM execution with Ollama
- Document processing and vector embeddings
- Memory management for conversations
- Real-time streaming responses
- Complete setup guide and examples

[📖 View LangChain Project Documentation](./langchain/README.md)

### 🕸️ LangGraph Examples (`/langgraph`)

*Coming soon - Advanced workflow and agent examples using LangGraph*

## Getting Started

1. Choose the project you're interested in
2. Navigate to the respective folder
3. Follow the setup instructions in the project's README
4. Run the examples to see the frameworks in action

## Prerequisites

- Python 3.8+
- Ollama (for LangChain examples)
- Git

## Quick Start

For the LangChain project:

```bash
# Navigate to the langchain directory
cd langchain

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running and has llama3.2:latest
ollama pull llama3.2:latest

# Run a simple example
python simple_chat.py
```

## Contributing

Feel free to contribute more examples, improvements, or new project templates!