"""
Simple Agent Example with LangGraph and Ollama

This example demonstrates how to create a basic agent workflow using LangGraph
that can handle different types of queries and route them appropriately.
"""

from typing import TypedDict, List
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
import json

# Define the state schema
class AgentState(TypedDict):
    messages: List[str]
    query_type: str
    response: str
    current_step: str

def classify_query(state: AgentState) -> AgentState:
    """Classify the type of query to determine processing path"""
    llm = Ollama(model="llama3.2:latest")
    
    query = state["messages"][-1] if state["messages"] else ""
    
    classification_prompt = f"""
    Analyze this query and classify it into one of these categories:
    - "question": General questions that need factual answers
    - "creative": Creative writing, stories, poems, etc.
    - "technical": Programming, technical explanations, how-to guides
    - "math": Mathematical problems or calculations
    
    Query: {query}
    
    Respond with just the category name.
    """
    
    query_type = llm.invoke(classification_prompt).strip().lower()
    
    state["query_type"] = query_type
    state["current_step"] = "classified"
    
    print(f"🔍 Query classified as: {query_type}")
    
    return state

def handle_question(state: AgentState) -> AgentState:
    """Handle general questions"""
    llm = Ollama(model="llama3.2:latest")
    
    query = state["messages"][-1]
    
    prompt = f"""
    You are a helpful assistant. Answer this question clearly and concisely:
    
    Question: {query}
    
    Provide a well-structured answer with relevant details.
    """
    
    response = llm.invoke(prompt)
    state["response"] = response
    state["current_step"] = "answered"
    
    print("💭 Generating factual answer...")
    
    return state

def handle_creative(state: AgentState) -> AgentState:
    """Handle creative writing requests"""
    llm = Ollama(model="llama3.2:latest")
    
    query = state["messages"][-1]
    
    prompt = f"""
    You are a creative writing assistant. Be imaginative and engaging:
    
    Request: {query}
    
    Create something creative, engaging, and well-written.
    """
    
    response = llm.invoke(prompt)
    state["response"] = response
    state["current_step"] = "created"
    
    print("🎨 Generating creative content...")
    
    return state

def handle_technical(state: AgentState) -> AgentState:
    """Handle technical queries"""
    llm = Ollama(model="llama3.2:latest")
    
    query = state["messages"][-1]
    
    prompt = f"""
    You are a technical expert. Provide detailed, accurate technical information:
    
    Technical Query: {query}
    
    Include examples, code snippets if relevant, and step-by-step explanations.
    """
    
    response = llm.invoke(prompt)
    state["response"] = response
    state["current_step"] = "explained"
    
    print("🔧 Generating technical explanation...")
    
    return state

def handle_math(state: AgentState) -> AgentState:
    """Handle mathematical problems"""
    llm = Ollama(model="llama3.2:latest")
    
    query = state["messages"][-1]
    
    prompt = f"""
    You are a math tutor. Solve this problem step by step:
    
    Math Problem: {query}
    
    Show your work clearly and explain each step.
    """
    
    response = llm.invoke(prompt)
    state["response"] = response
    state["current_step"] = "solved"
    
    print("🧮 Solving mathematical problem...")
    
    return state

def route_query(state: AgentState) -> str:
    """Route to appropriate handler based on query type"""
    query_type = state["query_type"]
    
    routing_map = {
        "question": "handle_question",
        "creative": "handle_creative", 
        "technical": "handle_technical",
        "math": "handle_math"
    }
    
    return routing_map.get(query_type, "handle_question")

def create_agent_graph():
    """Create the agent workflow graph"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("handle_question", handle_question)
    workflow.add_node("handle_creative", handle_creative)
    workflow.add_node("handle_technical", handle_technical)
    workflow.add_node("handle_math", handle_math)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional edges based on classification
    workflow.add_conditional_edges(
        "classify",
        route_query,
        {
            "handle_question": "handle_question",
            "handle_creative": "handle_creative",
            "handle_technical": "handle_technical", 
            "handle_math": "handle_math"
        }
    )
    
    # All handlers end the workflow
    workflow.add_edge("handle_question", END)
    workflow.add_edge("handle_creative", END)
    workflow.add_edge("handle_technical", END)
    workflow.add_edge("handle_math", END)
    
    return workflow.compile()

def main():
    print("🤖 LangGraph Simple Agent with Ollama")
    print("This agent will classify your query and route it to the appropriate handler")
    print("Types: question, creative, technical, math")
    print("Type 'quit' to exit\n")
    
    # Create the agent graph
    agent = create_agent_graph()
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        try:
            # Initialize state
            initial_state = {
                "messages": [user_input],
                "query_type": "",
                "response": "",
                "current_step": "start"
            }
            
            # Run the workflow
            result = agent.invoke(initial_state)
            
            print(f"\nAssistant: {result['response']}")
            print(f"📊 Processing path: {result['current_step']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
