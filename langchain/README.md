# LangChain Ollama Sample Project

This project demonstrates how to use LangChain with Ollama and the llama3.2:latest model for various AI tasks including simple chat, document Q&A, and conversation memory.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running
3. **llama3.2:latest** model downloaded

## Setup Instructions

### 1. Install Ollama

Visit [https://ollama.ai](https://ollama.ai) and download Ollama for your operating system.

### 2. Install the llama3.2:latest model

```bash
ollama pull llama3.2:latest
```

### 3. Verify Ollama is running

```bash
ollama list
```

You should see `llama3.2:latest` in the list of available models.

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
langchain/
├── requirements.txt          # Python dependencies
├── simple_chat.py           # Basic chat example
├── document_qa.py           # Document Q&A example
├── conversation_memory.py   # Conversation with memory
├── streaming_chat.py        # Streaming responses
└── sample_document.txt      # Sample document for Q&A
```

## Examples

### 1. Simple Chat (`simple_chat.py`)

A basic example showing how to use LangChain with Ollama for simple question-answering.

```bash
python simple_chat.py
```

### 2. Document Q&A (`document_qa.py`)

Demonstrates how to use LangChain to answer questions about a specific document.

```bash
python document_qa.py
```

### 3. Conversation Memory (`conversation_memory.py`)

Shows how to maintain conversation context across multiple interactions.

```bash
python conversation_memory.py
```

### 4. Streaming Chat (`streaming_chat.py`)

Example of how to stream responses from the model in real-time.

```bash
python streaming_chat.py
```

## Key Features Demonstrated

- **Basic LLM Integration**: How to connect LangChain with Ollama
- **Document Loading**: Loading and processing text documents
- **Text Splitting**: Breaking documents into manageable chunks
- **Vector Embeddings**: Creating embeddings for document search
- **Retrieval-Augmented Generation (RAG)**: Combining document retrieval with generation
- **Conversation Memory**: Maintaining context across conversations
- **Streaming Responses**: Real-time response streaming

## Configuration

The examples use the following default configuration:
- **Model**: `llama3.2:latest`
- **Ollama Base URL**: `http://localhost:11434`
- **Temperature**: `0.7` (adjustable for creativity vs consistency)

## Troubleshooting

### Common Issues

1. **Ollama not running**: Make sure Ollama service is started
   ```bash
   ollama serve
   ```

2. **Model not found**: Ensure llama3.2:latest is downloaded
   ```bash
   ollama pull llama3.2:latest
   ```

3. **Connection refused**: Check if Ollama is running on the correct port (default: 11434)

4. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

## Customization

You can modify the examples to:
- Use different Ollama models
- Adjust temperature and other model parameters
- Load different types of documents
- Implement custom prompt templates
- Add more sophisticated memory systems

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [LangChain Community](https://python.langchain.com/docs/integrations/llms/ollama)

## Contributing

Feel free to add more examples or improve existing ones. Some ideas for additional examples:
- Function calling with tools
- Multi-agent conversations
- Custom chains and prompts
- Integration with external APIs
