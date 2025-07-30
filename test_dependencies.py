"""
Test script to verify LangChain and LangGraph installations work correctly
"""

def test_langchain():
    """Test LangChain imports"""
    try:
        from langchain_community.llms import Ollama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        print("✅ LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ LangChain import error: {e}")
        return False

def test_langgraph():
    """Test LangGraph imports"""
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        print("✅ LangGraph imports successful")
        return True
    except ImportError as e:
        print(f"❌ LangGraph import error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3.2:latest")
        # Just test the initialization, don't make a call yet
        print("✅ Ollama initialization successful")
        print("⚠️  Note: Make sure Ollama is running and llama3.2:latest is pulled")
        return True
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Dependencies...")
    print("=" * 40)
    
    langchain_ok = test_langchain()
    langgraph_ok = test_langgraph()
    ollama_ok = test_ollama_connection()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"LangChain: {'✅ OK' if langchain_ok else '❌ FAILED'}")
    print(f"LangGraph: {'✅ OK' if langgraph_ok else '❌ FAILED'}")
    print(f"Ollama: {'✅ OK' if ollama_ok else '❌ FAILED'}")
    
    if all([langchain_ok, langgraph_ok, ollama_ok]):
        print("\n🎉 All dependencies are working correctly!")
        print("You can now run the examples.")
    else:
        print("\n⚠️  Some dependencies have issues. Please check the errors above.")
        print("Make sure to run: pip install -r requirements.txt")
        print("And ensure Ollama is running with: ollama pull llama3.2:latest")
