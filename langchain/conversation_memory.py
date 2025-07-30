"""
Conversation Memory Example with LangChain and Ollama

This example demonstrates how to maintain conversation context
across multiple interactions using LangChain's memory capabilities.
"""

from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

def main():
    # Initialize Ollama
    llm = Ollama(model="llama3.2:latest")
    
    # Create memory to store conversation history
    memory = ConversationBufferMemory()
    
    # Create a custom prompt template that includes conversation history
    template = """The following is a friendly conversation between a human and an AI assistant. 
    The AI assistant is helpful, creative, clever, and very friendly.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Create conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True  # Set to False to hide chain details
    )
    
    print("🤖 LangChain + Ollama Conversation with Memory")
    print("The AI will remember our conversation context!")
    print("Type 'quit' to exit, 'memory' to see conversation history\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'memory':
            print("\n📚 Conversation History:")
            print(memory.buffer)
            print("-" * 50)
            continue
        
        try:
            # Generate response with memory
            response = conversation.predict(input=user_input)
            print(f"Assistant: {response}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
