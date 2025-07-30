"""
Streaming Chat Example with LangChain and Ollama

This example demonstrates how to stream responses from Ollama
in real-time for a more interactive chat experience.
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys

def main():
    # Initialize Ollama with streaming callback
    llm = Ollama(
        model="llama3.2:latest",
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=True
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Please provide a detailed response to: {question}"
    )
    
    # Create a chain
    chain = prompt | llm | StrOutputParser()
    
    print("🤖 LangChain + Ollama Streaming Chat")
    print("Responses will stream in real-time!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_question = input("You: ")
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        try:
            print("Assistant: ", end="", flush=True)
            
            # Stream the response
            response = ""
            for chunk in chain.stream({"question": user_question}):
                response += chunk
                print(chunk, end="", flush=True)
            
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
