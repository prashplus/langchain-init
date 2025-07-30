# Contributing to LangChain & LangGraph Examples

Thank you for your interest in contributing! This repository contains example code for LangChain and LangGraph frameworks.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/prashplus/langchain-init.git
   cd langchain-init
   ```

2. **Install dependencies**
   ```bash
   # For LangChain examples
   pip install -r langchain/requirements.txt
   
   # For LangGraph examples
   pip install -r langgraph/requirements.txt
   ```

3. **Set up Ollama (for running examples)**
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull the required model
   ollama pull llama3.2:latest
   ```

## Testing

Before submitting a pull request, make sure your changes pass all tests:

### Local Testing

```bash
# Test dependencies and imports
python test_dependencies.py

# Run CI tests (syntax, structure, imports)
python ci_test.py

# Validate example files
python validate_examples.py
```

### Automated Testing

The repository includes GitHub Actions workflows that automatically:

- **CI Tests**: Run on every push and pull request
  - Test Python syntax across multiple Python versions (3.9, 3.10, 3.11)
  - Validate imports and dependencies
  - Check file structure and examples
  - Run linting with flake8

- **Quick Status Check**: Fast validation for basic functionality
- **Health Check**: Weekly automated dependency and health monitoring

## Code Guidelines

1. **Python Code Style**
   - Follow PEP 8 guidelines
   - Use meaningful variable names
   - Add docstrings to functions and classes
   - Keep functions focused and concise

2. **Example Structure**
   - Each example should have a `main()` function
   - Include clear comments explaining the code
   - Add error handling for external dependencies (like Ollama)
   - Provide informative print statements

3. **Dependencies**
   - Keep dependencies minimal and well-documented
   - Update `requirements.txt` files when adding new dependencies
   - Test that examples work with the specified dependency versions

## Submitting Changes

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the guidelines above
3. **Test locally** using the commands above
4. **Submit a pull request** with a clear description of changes

### Pull Request Requirements

- [ ] All CI tests pass
- [ ] Code follows the style guidelines
- [ ] New examples include proper documentation
- [ ] Dependencies are updated if needed
- [ ] Examples can run successfully (when Ollama is available)

## Example Categories

### LangChain Examples (`/langchain`)
- Focus on core LangChain functionality
- Integration with Ollama for local LLM usage
- Document processing and RAG implementations
- Memory and conversation management

### LangGraph Examples (`/langgraph`)
- Multi-step workflows and agent patterns
- State management and complex reasoning
- Tool integration and function calling
- Creative and research applications

## Questions or Issues?

- Check existing issues before creating new ones
- Provide clear reproduction steps for bugs
- Include environment details (Python version, OS, etc.)
- Use the issue templates when available

Thank you for contributing! 🎉
