"""
Conditional Workflow Example with LangGraph and Ollama

This example demonstrates how to create workflows with conditional branching
based on content analysis and dynamic routing.
"""

from typing import TypedDict, List, Dict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# Define the state schema
class WorkflowState(TypedDict):
    input_text: str
    content_type: str
    sentiment: str
    complexity: str
    processing_path: List[str]
    results: Dict[str, str]
    final_output: str

def analyze_content(state: WorkflowState) -> WorkflowState:
    """Analyze the input content for type, sentiment, and complexity"""
    llm = Ollama(model="llama3.2:latest")
    
    text = state["input_text"]
    
    analysis_prompt = f"""
    Analyze the following text and provide:
    1. Content Type: (question, request, complaint, compliment, information, other)
    2. Sentiment: (positive, negative, neutral)
    3. Complexity: (simple, moderate, complex)
    
    Text: {text}
    
    Respond in this exact format:
    Content Type: [type]
    Sentiment: [sentiment]  
    Complexity: [complexity]
    """
    
    analysis = llm.invoke(analysis_prompt)
    
    # Parse the analysis (simplified parsing)
    content_type = "other"
    sentiment = "neutral"
    complexity = "moderate"
    
    for line in analysis.split('\n'):
        if line.startswith("Content Type:"):
            content_type = line.split(":")[1].strip().lower()
        elif line.startswith("Sentiment:"):
            sentiment = line.split(":")[1].strip().lower()
        elif line.startswith("Complexity:"):
            complexity = line.split(":")[1].strip().lower()
    
    state["content_type"] = content_type
    state["sentiment"] = sentiment
    state["complexity"] = complexity
    state["processing_path"] = ["analysis"]
    
    print(f"📊 Analysis Complete:")
    print(f"   Type: {content_type}")
    print(f"   Sentiment: {sentiment}")
    print(f"   Complexity: {complexity}")
    
    return state

def handle_question(state: WorkflowState) -> WorkflowState:
    """Handle question-type content"""
    llm = Ollama(model="llama3.2:latest")
    
    text = state["input_text"]
    complexity = state["complexity"]
    
    if complexity == "simple":
        prompt = f"Provide a clear, concise answer to this question: {text}"
    elif complexity == "complex":
        prompt = f"Provide a detailed, comprehensive answer with examples to this complex question: {text}"
    else:
        prompt = f"Provide a well-structured answer to this question: {text}"
    
    response = llm.invoke(prompt)
    state["results"]["question_response"] = response
    state["processing_path"].append("question_handler")
    
    print("❓ Processing as question...")
    
    return state

def handle_request(state: WorkflowState) -> WorkflowState:
    """Handle request-type content"""
    llm = Ollama(model="llama3.2:latest")
    
    text = state["input_text"]
    
    prompt = f"""
    This is a request. Provide helpful guidance or steps to fulfill it:
    
    Request: {text}
    
    Provide actionable steps or information.
    """
    
    response = llm.invoke(prompt)
    state["results"]["request_response"] = response
    state["processing_path"].append("request_handler")
    
    print("📝 Processing as request...")
    
    return state

def handle_complaint(state: WorkflowState) -> WorkflowState:
    """Handle complaint-type content with empathy"""
    llm = Ollama(model="llama3.2:latest")
    
    text = state["input_text"]
    
    prompt = f"""
    This is a complaint. Respond with empathy and provide helpful solutions:
    
    Complaint: {text}
    
    Show understanding, acknowledge the issue, and suggest solutions.
    """
    
    response = llm.invoke(prompt)
    state["results"]["complaint_response"] = response
    state["processing_path"].append("complaint_handler")
    
    print("😟 Processing as complaint with empathy...")
    
    return state

def handle_compliment(state: WorkflowState) -> WorkflowState:
    """Handle compliment-type content"""
    llm = Ollama(model="llama3.2:latest")
    
    text = state["input_text"]
    
    prompt = f"""
    This is a compliment or positive feedback. Respond graciously:
    
    Compliment: {text}
    
    Show appreciation and engage positively.
    """
    
    response = llm.invoke(prompt)
    state["results"]["compliment_response"] = response
    state["processing_path"].append("compliment_handler")
    
    print("😊 Processing as compliment...")
    
    return state

def handle_information(state: WorkflowState) -> WorkflowState:
    """Handle information-sharing content"""
    llm = Ollama(model="llama3.2:latest")
    
    text = state["input_text"]
    
    prompt = f"""
    This appears to be information sharing. Acknowledge and add relevant insights:
    
    Information: {text}
    
    Acknowledge the information and provide relevant additional insights or questions.
    """
    
    response = llm.invoke(prompt)
    state["results"]["information_response"] = response
    state["processing_path"].append("information_handler")
    
    print("ℹ️ Processing as information sharing...")
    
    return state

def handle_other(state: WorkflowState) -> WorkflowState:
    """Handle other types of content"""
    llm = Ollama(model="llama3.2:latest")
    
    text = state["input_text"]
    
    prompt = f"""
    Respond appropriately to this content:
    
    Content: {text}
    
    Provide a helpful and relevant response.
    """
    
    response = llm.invoke(prompt)
    state["results"]["other_response"] = response
    state["processing_path"].append("other_handler")
    
    print("🔄 Processing as general content...")
    
    return state

def apply_sentiment_filter(state: WorkflowState) -> WorkflowState:
    """Apply sentiment-based modifications to the response"""
    sentiment = state["sentiment"]
    
    # Get the main response
    main_response = ""
    for key, value in state["results"].items():
        if value:
            main_response = value
            break
    
    if sentiment == "negative":
        # Add empathetic tone for negative sentiment
        prefix = "I understand this might be frustrating. "
        state["final_output"] = prefix + main_response
        print("💙 Applied empathetic tone for negative sentiment")
    elif sentiment == "positive":
        # Add enthusiastic tone for positive sentiment
        prefix = "I'm glad to help with this! "
        state["final_output"] = prefix + main_response
        print("✨ Applied positive tone for good sentiment")
    else:
        # Neutral tone
        state["final_output"] = main_response
        print("⚖️ Applied neutral tone")
    
    state["processing_path"].append("sentiment_filter")
    
    return state

def route_by_content_type(state: WorkflowState) -> str:
    """Route to appropriate handler based on content type"""
    content_type = state["content_type"]
    
    routing_map = {
        "question": "handle_question",
        "request": "handle_request",
        "complaint": "handle_complaint", 
        "compliment": "handle_compliment",
        "information": "handle_information",
        "other": "handle_other"
    }
    
    return routing_map.get(content_type, "handle_other")

def should_apply_sentiment_filter(state: WorkflowState) -> str:
    """Determine if sentiment filtering should be applied"""
    # Always apply sentiment filter for this demo
    return "apply_sentiment"

def create_conditional_workflow():
    """Create the conditional workflow graph"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_content)
    workflow.add_node("handle_question", handle_question)
    workflow.add_node("handle_request", handle_request)
    workflow.add_node("handle_complaint", handle_complaint)
    workflow.add_node("handle_compliment", handle_compliment)
    workflow.add_node("handle_information", handle_information)
    workflow.add_node("handle_other", handle_other)
    workflow.add_node("apply_sentiment", apply_sentiment_filter)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Conditional routing based on content type
    workflow.add_conditional_edges(
        "analyze",
        route_by_content_type,
        {
            "handle_question": "handle_question",
            "handle_request": "handle_request",
            "handle_complaint": "handle_complaint",
            "handle_compliment": "handle_compliment", 
            "handle_information": "handle_information",
            "handle_other": "handle_other"
        }
    )
    
    # All handlers go to sentiment filter
    for handler in ["handle_question", "handle_request", "handle_complaint", 
                   "handle_compliment", "handle_information", "handle_other"]:
        workflow.add_conditional_edges(
            handler,
            should_apply_sentiment_filter,
            {"apply_sentiment": "apply_sentiment"}
        )
    
    # End after sentiment filter
    workflow.add_edge("apply_sentiment", END)
    
    return workflow.compile()

def main():
    print("🔀 LangGraph Conditional Workflow with Ollama")
    print("This workflow analyzes content and routes it through different processing paths")
    print("Type 'quit' to exit\n")
    
    # Create the workflow
    workflow = create_conditional_workflow()
    
    # Example inputs for testing
    examples = [
        "How do I bake a chocolate cake?",  # Question
        "Can you help me write a resignation letter?",  # Request
        "Your service is terrible and slow!",  # Complaint
        "You did an amazing job helping me yesterday!",  # Compliment
        "I just learned that Python is great for AI development.",  # Information
        "Whatever, I don't care anymore."  # Other
    ]
    
    print("Example inputs you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print()
    
    while True:
        user_input = input("Input: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.strip() == "":
            continue
        
        try:
            # Initialize state
            initial_state = {
                "input_text": user_input,
                "content_type": "",
                "sentiment": "",
                "complexity": "",
                "processing_path": [],
                "results": {},
                "final_output": ""
            }
            
            print("\n🔄 Processing through conditional workflow...")
            print("=" * 50)
            
            # Run the workflow
            result = workflow.invoke(initial_state)
            
            print("\n" + "=" * 50)
            print("📋 WORKFLOW COMPLETE")
            print("=" * 50)
            print(f"🎯 Final Output: {result['final_output']}")
            print(f"🛤️ Processing Path: {' → '.join(result['processing_path'])}")
            print(f"📊 Analysis: {result['content_type']} | {result['sentiment']} | {result['complexity']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
