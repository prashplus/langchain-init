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

### 🕸️ LangGraph with Ollama (`/langgraph`)

A comprehensive collection of LangGraph examples demonstrating advanced AI workflows and multi-agent systems using Ollama and the llama3.2:latest model. This project includes:

- **Simple Agent**: Basic agent workflow with conditional routing
- **Multi-Step Reasoning**: Complex problem-solving through multiple reasoning stages
- **Conditional Workflow**: Dynamic routing based on content analysis
- **Research Assistant**: Multi-stage research and summarization workflow
- **Code Reviewer**: Automated code review through multiple analysis stages
- **Creative Writing**: Multi-stage creative content generation and refinement
- **Chat with Tools**: Conversational agent with tool integration capabilities

**Advanced Features:**
- Stateful workflows with persistent context
- Conditional branching and dynamic routing
- Multi-agent collaboration patterns
- Tool integration and execution
- Streaming workflow updates
- Complex state management

[📖 View LangGraph Project Documentation](./langgraph/README.md)

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

For the LangGraph project:

```bash
# Navigate to the langgraph directory
cd langgraph

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running and has llama3.2:latest
ollama pull llama3.2:latest

# Run a workflow example
python simple_agent.py
```

## Contributing

Feel free to contribute more examples, improvements, or new project templates!

## Authors

* **Prashant Piprotar** - - [Prash+](https://github.com/prashplus)

Visit my blog for more Tech Stuff
### http://prashplus.blogspot.com