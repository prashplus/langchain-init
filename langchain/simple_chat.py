"""
Simple Chat Example with LangChain and Ollama

This example demonstrates basic usage of LangChain with Ollama
using the llama3.2:latest model for simple question-answering.
"""

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

def main():
    # Initialize Ollama with llama3.2:latest model
    llm = Ollama(model="llama3.2:latest")
    
    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the following question: {question}"
    )
    
    # Create a chain
    chain = prompt | llm | StrOutputParser()
    
    print("🤖 LangChain + Ollama Simple Chat")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_question = input("You: ")
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        try:
            # Generate response
            print("Assistant: ", end="", flush=True)
            response = chain.invoke({"question": user_question})
            print(response)
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
