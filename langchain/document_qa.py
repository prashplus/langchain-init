"""
Document Q&A Example with LangChain and Ollama

This example demonstrates how to use LangChain with Ollama to answer
questions about a specific document using Retrieval-Augmented Generation (RAG).
"""

from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def load_document(file_path):
    """Load document from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating a sample document...")
        return create_sample_document(file_path)

def create_sample_document(file_path):
    """Create a sample document if it doesn't exist"""
    sample_content = """
    LangChain is a framework for developing applications powered by language models. 
    It enables developers to build context-aware and reasoning applications that can 
    connect language models to other sources of data and interact with their environment.

    Key Features of LangChain:
    1. LLMs and Prompts: Management and optimization of prompts and interface with LLMs
    2. Chains: Sequences of calls to LLMs or other utilities
    3. Data Augmented Generation: Chains that interact with external data sources
    4. Agents: Chains that use LLMs to determine actions to take
    5. Memory: Persist state between calls of a chain/agent

    Ollama is a tool that allows you to run large language models locally on your machine.
    It supports various models including Llama, Mistral, and others. Ollama makes it easy
    to get up and running with LLMs without needing cloud services or API keys.

    Benefits of using Ollama:
    - Privacy: Your data stays on your machine
    - Speed: No network latency for inference
    - Cost: No per-token charges
    - Offline: Works without internet connection
    - Customization: Fine-tune models for specific use cases
    """
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(sample_content)
    
    return sample_content

def main():
    # Initialize Ollama
    llm = Ollama(model="llama3.2:latest")
    
    # Load document
    document_path = "sample_document.txt"
    document_content = load_document(document_path)
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(document_content)
    
    print(f"Document split into {len(texts)} chunks")
    
    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model="llama3.2:latest")
    
    # Create vector store
    print("Creating vector store...")
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Create custom prompt template
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer based on the context, just say that you don't know, 
    don't try to make up an answer.

    Context: {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    print("\n🤖 LangChain + Ollama Document Q&A")
    print(f"Loaded document: {document_path}")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("Question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        try:
            # Get answer
            result = qa_chain({"query": question})
            
            print(f"\nAnswer: {result['result']}")
            
            # Show source documents
            print("\nSource chunks used:")
            for i, doc in enumerate(result['source_documents']):
                print(f"{i+1}. {doc.page_content[:100]}...")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
