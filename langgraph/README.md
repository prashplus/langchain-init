# LangGraph Ollama Demo Project

This project demonstrates how to use LangGraph with Ollama and the llama3.2:latest model for building sophisticated AI workflows, agents, and multi-step reasoning systems.

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain's expression language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.

Key concepts:
- **Nodes**: Individual processing units (functions or chains)
- **Edges**: Connections between nodes that define the flow
- **State**: Shared data structure that persists across the graph
- **Conditional Edges**: Dynamic routing based on conditions

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running
3. **llama3.2:latest** model downloaded

## Setup Instructions

### 1. Install Ollama and Model

```bash
# Install the model
ollama pull llama3.2:latest

# Verify installation
ollama list
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter dependency conflicts, try creating a virtual environment first:

```bash
# Create virtual environment
python -m venv langgraph_env

# Activate virtual environment
# On Windows:
langgraph_env\Scripts\activate
# On Mac/Linux:
source langgraph_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Test the Setup

```bash
python test_setup.py
```

## Project Structure

```
langgraph/
├── requirements.txt              # Python dependencies (compatible versions)
├── test_setup.py                 # Test script to verify setup
├── simple_agent.py              # Basic agent workflow
├── multi_step_reasoning.py      # Multi-step problem solving
├── conditional_workflow.py      # Conditional branching
├── research_assistant.py        # Research and summarization workflow
├── code_reviewer.py             # Code review workflow
├── creative_writing.py          # Creative writing assistant
└── chat_with_tools.py           # Chat agent with tool usage
```

## Examples

### 1. Simple Agent (`simple_agent.py`)

A basic LangGraph workflow demonstrating how to create a simple agent that can handle different types of queries.

```bash
python simple_agent.py
```

### 2. Multi-Step Reasoning (`multi_step_reasoning.py`)

Demonstrates how to break down complex problems into multiple reasoning steps using LangGraph.

```bash
python multi_step_reasoning.py
```

### 3. Conditional Workflow (`conditional_workflow.py`)

Shows how to create workflows with conditional branching based on content analysis.

```bash
python conditional_workflow.py
```

### 4. Research Assistant (`research_assistant.py`)

A more complex example that demonstrates a research workflow with multiple stages.

```bash
python research_assistant.py
```

### 5. Code Reviewer (`code_reviewer.py`)

An automated code review workflow that analyzes code through multiple stages.

```bash
python code_reviewer.py
```

### 6. Creative Writing (`creative_writing.py`)

A creative writing assistant that uses multiple stages to generate and refine stories.

```bash
python creative_writing.py
```

### 7. Chat with Tools (`chat_with_tools.py`)

An interactive chat agent that can use various tools and maintain conversation state.

```bash
python chat_with_tools.py
```

## Key Features Demonstrated

- **Stateful Workflows**: Maintaining state across multiple processing steps
- **Conditional Logic**: Dynamic routing based on content and conditions
- **Multi-Agent Patterns**: Coordinating multiple AI agents
- **Tool Integration**: Using external tools within workflows
- **Error Handling**: Robust error handling and recovery
- **Streaming Support**: Real-time updates during workflow execution
- **Memory Management**: Persistent state across workflow executions

## LangGraph Concepts Covered

### 1. **Graph Construction**
- Creating nodes and edges
- Defining state schemas
- Setting up conditional edges

### 2. **State Management**
- Shared state across nodes
- State updates and transformations
- State persistence

### 3. **Workflow Patterns**
- Sequential processing
- Parallel execution
- Conditional branching
- Loops and cycles

### 4. **Agent Patterns**
- ReAct (Reasoning + Acting) pattern
- Plan-and-Execute pattern
- Multi-agent collaboration

## Configuration

The examples use the following default configuration:
- **Model**: `llama3.2:latest`
- **Ollama Base URL**: `http://localhost:11434`
- **Temperature**: `0.7` (adjustable per use case)
- **Max Iterations**: Configurable for each workflow

## Troubleshooting

### Common Issues

1. **Ollama not running**: Ensure Ollama service is started
   ```bash
   ollama serve
   ```

2. **Model not found**: Ensure llama3.2:latest is downloaded
   ```bash
   ollama pull llama3.2:latest
   ```

3. **Graph execution errors**: Check node implementations and state schema
4. **Memory issues**: Adjust chunk sizes and state management
5. **Dependency conflicts**: Use a virtual environment
   ```bash
   python -m venv langgraph_env
   langgraph_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
6. **Import errors**: Test your setup
   ```bash
   python test_setup.py
   ```

## Advanced Usage

### Custom State Schema

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph

class CustomState(TypedDict):
    messages: List[str]
    current_step: str
    results: dict
```

### Adding Custom Tools

```python
def custom_tool(query: str) -> str:
    # Your custom tool implementation
    return f"Result for: {query}"
```

### Streaming Workflows

```python
for chunk in graph.stream(initial_state):
    print(f"Step: {chunk}")
```

## Performance Tips

1. **Optimize Node Functions**: Keep individual nodes focused and efficient
2. **State Management**: Only store necessary data in the state
3. **Parallel Execution**: Use parallel edges where possible
4. **Caching**: Implement caching for expensive operations
5. **Memory Management**: Clear unnecessary state data between runs

## Comparison with Traditional Chains

| Feature | LangChain Chains | LangGraph |
|---------|------------------|-----------|
| **Flow Control** | Linear/Sequential | Cyclic/Conditional |
| **State Management** | Limited | Full State Persistence |
| **Branching** | Basic | Advanced Conditional |
| **Multi-Agent** | Complex Setup | Native Support |
| **Debugging** | Limited | Built-in Visualization |

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama Documentation](https://github.com/jmorganca/ollama)

## Contributing

Ideas for additional examples:
- Multi-modal workflows (text + images)
- Integration with external APIs
- Custom tool creation
- Workflow optimization patterns
- Error recovery strategies

Feel free to contribute new examples or improvements to existing ones!
