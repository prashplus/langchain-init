"""
Test script for LangChain project dependencies
"""

def test_imports():
    """Test all required imports"""
    try:
        from langchain_community.llms import Ollama
        from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA, ConversationChain
        from langchain.memory import ConversationBufferMemory
        from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        print("✅ All LangChain imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic LangChain functionality"""
    try:
        from langchain_community.llms import Ollama
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = Ollama(model="llama3.2:latest")
        prompt = ChatPromptTemplate.from_template("Hello, this is a test: {input}")
        chain = prompt | llm
        
        print("✅ Basic chain creation successful")
        print("⚠️  To test full functionality, ensure Ollama is running")
        return True
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing LangChain Dependencies...")
    print("=" * 40)
    
    imports_ok = test_imports()
    basic_ok = test_basic_functionality()
    
    print("\n" + "=" * 40)
    if imports_ok and basic_ok:
        print("🎉 LangChain setup is working correctly!")
        print("You can now run: python simple_chat.py")
    else:
        print("⚠️  Issues detected. Please run: pip install -r requirements.txt")
