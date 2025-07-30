"""
Test script for LangGraph project dependencies
"""

def test_imports():
    """Test all required imports"""
    try:
        from langgraph.graph import StateGraph, END
        from langchain_community.llms import Ollama
        from typing import TypedDict, List, Dict
        import json
        import random
        from datetime import datetime
        print("✅ All LangGraph imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_graph():
    """Test basic LangGraph functionality"""
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        class TestState(TypedDict):
            message: str
        
        def test_node(state: TestState) -> TestState:
            state["message"] = "Hello from LangGraph!"
            return state
        
        workflow = StateGraph(TestState)
        workflow.add_node("test", test_node)
        workflow.set_entry_point("test")
        workflow.add_edge("test", END)
        
        graph = workflow.compile()
        result = graph.invoke({"message": ""})
        
        if result["message"] == "Hello from LangGraph!":
            print("✅ Basic LangGraph workflow successful")
            return True
        else:
            print("❌ Basic workflow test failed")
            return False
    except Exception as e:
        print(f"❌ Basic graph error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing LangGraph Dependencies...")
    print("=" * 40)
    
    imports_ok = test_imports()
    graph_ok = test_basic_graph()
    
    print("\n" + "=" * 40)
    if imports_ok and graph_ok:
        print("🎉 LangGraph setup is working correctly!")
        print("You can now run: python simple_agent.py")
    else:
        print("⚠️  Issues detected. Please run: pip install -r requirements.txt")
