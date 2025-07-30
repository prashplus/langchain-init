"""
Chat Agent with Tools Example using LangGraph and Ollama

This example demonstrates how to create a conversational agent that can
use various tools while maintaining conversation state and context.
"""

from typing import TypedDict, List, Dict, Any
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
import json
import random
from datetime import datetime
import math

# Define the state schema
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    available_tools: List[str]
    tool_results: Dict[str, Any]
    conversation_context: str
    needs_tool: bool
    selected_tool: str
    user_input: str
    response: str

# Define available tools
def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        # Simple math evaluation (be careful with eval in production!)
        result = eval(expression.replace('^', '**'))
        return f"Calculator result: {expression} = {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"

def random_number_tool(min_val: int = 1, max_val: int = 100) -> str:
    """Generate a random number within a range"""
    try:
        min_val = int(min_val)
        max_val = int(max_val)
        number = random.randint(min_val, max_val)
        return f"Random number between {min_val} and {max_val}: {number}"
    except Exception as e:
        return f"Random number error: {str(e)}"

def datetime_tool() -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def word_count_tool(text: str) -> str:
    """Count words in text"""
    word_count = len(text.split())
    char_count = len(text)
    return f"Text analysis - Words: {word_count}, Characters: {char_count}"

def reverse_text_tool(text: str) -> str:
    """Reverse the given text"""
    return f"Reversed text: {text[::-1]}"

# Tool registry
AVAILABLE_TOOLS = {
    "calculator": calculator_tool,
    "random_number": random_number_tool,
    "datetime": datetime_tool,
    "word_count": word_count_tool,
    "reverse_text": reverse_text_tool
}

def analyze_user_input(state: ChatState) -> ChatState:
    """Analyze user input to determine if tools are needed"""
    llm = Ollama(model="llama3.2:latest")
    
    user_input = state["user_input"]
    available_tools = list(AVAILABLE_TOOLS.keys())
    
    analysis_prompt = f"""
    Analyze this user message to determine if any tools should be used:
    
    User message: {user_input}
    
    Available tools:
    - calculator: For math calculations (e.g., "calculate 5+3", "what's 10*4")
    - random_number: For generating random numbers (e.g., "give me a random number", "random between 1 and 50")
    - datetime: For current date/time (e.g., "what time is it", "current date")
    - word_count: For counting words in text (e.g., "count words in this text")
    - reverse_text: For reversing text (e.g., "reverse this text", "backwards")
    
    Respond with:
    NEEDS_TOOL: yes/no
    TOOL_NAME: [tool name if needed, or "none"]
    REASONING: [brief explanation]
    
    If no tool is needed, just respond with conversational chat.
    """
    
    analysis = llm.invoke(analysis_prompt)
    
    # Parse analysis
    needs_tool = False
    selected_tool = "none"
    
    for line in analysis.split('\n'):
        if line.startswith("NEEDS_TOOL:"):
            needs_tool = "yes" in line.lower()
        elif line.startswith("TOOL_NAME:"):
            tool_name = line.split(":")[1].strip().lower()
            if tool_name in AVAILABLE_TOOLS:
                selected_tool = tool_name
    
    state["needs_tool"] = needs_tool
    state["selected_tool"] = selected_tool
    
    print(f"🔍 Analysis: Tool needed: {needs_tool}, Selected: {selected_tool}")
    
    return state

def execute_tool(state: ChatState) -> ChatState:
    """Execute the selected tool"""
    llm = Ollama(model="llama3.2:latest")
    
    tool_name = state["selected_tool"]
    user_input = state["user_input"]
    
    if tool_name not in AVAILABLE_TOOLS:
        state["tool_results"]["error"] = f"Tool {tool_name} not found"
        return state
    
    # Extract parameters for tool execution
    if tool_name == "calculator":
        # Extract mathematical expression
        extract_prompt = f"""
        Extract the mathematical expression from this message: {user_input}
        
        Examples:
        "calculate 5+3" -> "5+3"
        "what's 10*4-2" -> "10*4-2"
        "compute 15/3" -> "15/3"
        
        Respond with just the mathematical expression.
        """
        expression = llm.invoke(extract_prompt).strip()
        result = calculator_tool(expression)
        
    elif tool_name == "random_number":
        # Extract range if specified
        if "between" in user_input.lower():
            try:
                numbers = [int(s) for s in user_input.split() if s.isdigit()]
                if len(numbers) >= 2:
                    result = random_number_tool(numbers[0], numbers[1])
                else:
                    result = random_number_tool()
            except:
                result = random_number_tool()
        else:
            result = random_number_tool()
            
    elif tool_name == "datetime":
        result = datetime_tool()
        
    elif tool_name == "word_count":
        # Extract text to count
        extract_prompt = f"""
        Extract the text that should be word-counted from this message: {user_input}
        
        If the user is asking to count words in their own message, return the entire message.
        If they specify particular text, extract that text.
        
        Message: {user_input}
        
        Respond with just the text to be counted.
        """
        text_to_count = llm.invoke(extract_prompt).strip()
        result = word_count_tool(text_to_count)
        
    elif tool_name == "reverse_text":
        # Extract text to reverse
        extract_prompt = f"""
        Extract the text that should be reversed from this message: {user_input}
        
        Message: {user_input}
        
        Respond with just the text to be reversed.
        """
        text_to_reverse = llm.invoke(extract_prompt).strip()
        result = reverse_text_tool(text_to_reverse)
    
    else:
        result = "Tool execution failed"
    
    state["tool_results"][tool_name] = result
    
    print(f"🔧 Tool '{tool_name}' executed: {result[:50]}...")
    
    return state

def generate_response(state: ChatState) -> ChatState:
    """Generate conversational response, incorporating tool results if available"""
    llm = Ollama(model="llama3.2:latest")
    
    user_input = state["user_input"]
    needs_tool = state["needs_tool"]
    tool_results = state["tool_results"]
    
    # Build conversation context
    recent_messages = state["messages"][-5:] if state["messages"] else []
    context = ""
    for msg in recent_messages:
        context += f"{msg['role']}: {msg['content']}\n"
    
    if needs_tool and tool_results:
        # Generate response incorporating tool results
        tool_output = ""
        for tool, result in tool_results.items():
            if result:
                tool_output += f"{result}\n"
        
        response_prompt = f"""
        The user asked: {user_input}
        
        Tool results:
        {tool_output}
        
        Previous conversation context:
        {context}
        
        Generate a natural, conversational response that incorporates the tool results.
        Be friendly and helpful. Explain the results if needed.
        """
    else:
        # Generate regular conversational response
        response_prompt = f"""
        The user said: {user_input}
        
        Previous conversation context:
        {context}
        
        Generate a natural, helpful conversational response. Be friendly and engaging.
        """
    
    response = llm.invoke(response_prompt)
    state["response"] = response
    
    # Add messages to conversation history
    state["messages"].append({"role": "user", "content": user_input})
    state["messages"].append({"role": "assistant", "content": response})
    
    # Clear tool results for next interaction
    state["tool_results"] = {}
    
    print("💬 Response generated")
    
    return state

def route_after_analysis(state: ChatState) -> str:
    """Route based on whether tools are needed"""
    if state["needs_tool"]:
        return "execute_tool"
    else:
        return "generate_response"

def create_chat_agent_workflow():
    """Create the chat agent workflow graph"""
    
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_user_input)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("generate_response", generate_response)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Conditional routing based on tool needs
    workflow.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "execute_tool": "execute_tool",
            "generate_response": "generate_response"
        }
    )
    
    # After tool execution, always generate response
    workflow.add_edge("execute_tool", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()

def main():
    print("🤖 LangGraph Chat Agent with Tools using Ollama")
    print("Chat with me! I can use various tools to help you.")
    print("\nAvailable tools:")
    print("- 🧮 Calculator: math calculations")
    print("- 🎲 Random numbers: generate random numbers")
    print("- 🕐 Date/time: current date and time")
    print("- 📝 Word count: count words in text")
    print("- 🔄 Reverse text: reverse any text")
    print("\nType 'quit' to exit, 'tools' to see tool examples\n")
    
    # Create the chat agent
    agent = create_chat_agent_workflow()
    
    # Initialize conversation state
    conversation_state = {
        "messages": [],
        "available_tools": list(AVAILABLE_TOOLS.keys()),
        "tool_results": {},
        "conversation_context": "",
        "needs_tool": False,
        "selected_tool": "",
        "user_input": "",
        "response": ""
    }
    
    # Add welcome message
    conversation_state["messages"].append({
        "role": "assistant", 
        "content": "Hello! I'm your AI assistant with tool capabilities. How can I help you today?"
    })
    
    print("Assistant: Hello! I'm your AI assistant with tool capabilities. How can I help you today?")
    print("-" * 60)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'tools':
            print("\n🔧 Tool Examples:")
            print("- 'Calculate 15 * 4 + 7'")
            print("- 'Give me a random number between 1 and 100'")
            print("- 'What time is it?'")
            print("- 'Count words in: The quick brown fox jumps'")
            print("- 'Reverse this text: Hello World'")
            print()
            continue
        
        if user_input.strip() == "":
            continue
        
        try:
            # Update state with user input
            conversation_state["user_input"] = user_input
            conversation_state["needs_tool"] = False
            conversation_state["selected_tool"] = ""
            conversation_state["tool_results"] = {}
            
            # Run the workflow
            result = agent.invoke(conversation_state)
            
            # Update conversation state
            conversation_state.update(result)
            
            print(f"Assistant: {result['response']}")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
