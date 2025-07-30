"""
Multi-Step Reasoning Example with LangGraph and Ollama

This example demonstrates how to break down complex problems into
multiple reasoning steps using LangGraph workflows.
"""

from typing import TypedDict, List, Dict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# Define the state schema
class ReasoningState(TypedDict):
    original_problem: str
    reasoning_steps: List[Dict[str, str]]
    current_step: int
    final_answer: str
    confidence: str

def analyze_problem(state: ReasoningState) -> ReasoningState:
    """Analyze the problem and break it down into steps"""
    llm = Ollama(model="llama3.2:latest")
    
    problem = state["original_problem"]
    
    analysis_prompt = f"""
    Analyze this complex problem and break it down into logical reasoning steps:
    
    Problem: {problem}
    
    Create a step-by-step approach to solve this problem. List 3-5 key steps needed.
    Format your response as a numbered list where each step is one clear action.
    
    Example format:
    1. Understand the core question
    2. Identify relevant information
    3. Apply logical reasoning
    4. Calculate or deduce the answer
    5. Verify the solution
    """
    
    analysis = llm.invoke(analysis_prompt)
    
    # Parse the steps (simplified parsing)
    steps = []
    for line in analysis.split('\n'):
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
            step_text = line.strip()
            # Remove numbering
            if '.' in step_text:
                step_text = step_text.split('.', 1)[1].strip()
            steps.append({
                "description": step_text,
                "result": "",
                "status": "pending"
            })
    
    state["reasoning_steps"] = steps
    state["current_step"] = 0
    
    print(f"🔍 Problem analyzed into {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step['description']}")
    
    return state

def execute_reasoning_step(state: ReasoningState) -> ReasoningState:
    """Execute the current reasoning step"""
    llm = Ollama(model="llama3.2:latest")
    
    current_idx = state["current_step"]
    if current_idx >= len(state["reasoning_steps"]):
        return state
    
    current_step = state["reasoning_steps"][current_idx]
    problem = state["original_problem"]
    
    # Build context from previous steps
    previous_context = ""
    for i, step in enumerate(state["reasoning_steps"][:current_idx]):
        if step["result"]:
            previous_context += f"Step {i+1} result: {step['result']}\n"
    
    step_prompt = f"""
    Original Problem: {problem}
    
    Previous reasoning:
    {previous_context}
    
    Current step to execute: {current_step['description']}
    
    Execute this reasoning step. Provide your analysis and findings for this specific step.
    Be specific and detailed in your reasoning.
    """
    
    result = llm.invoke(step_prompt)
    
    # Update the step with the result
    state["reasoning_steps"][current_idx]["result"] = result
    state["reasoning_steps"][current_idx]["status"] = "completed"
    
    print(f"✅ Step {current_idx + 1} completed: {current_step['description']}")
    print(f"   Result: {result[:100]}...")
    
    return state

def advance_step(state: ReasoningState) -> ReasoningState:
    """Advance to the next reasoning step"""
    state["current_step"] += 1
    return state

def synthesize_answer(state: ReasoningState) -> ReasoningState:
    """Synthesize all reasoning steps into a final answer"""
    llm = Ollama(model="llama3.2:latest")
    
    problem = state["original_problem"]
    
    # Build complete reasoning chain
    reasoning_chain = ""
    for i, step in enumerate(state["reasoning_steps"]):
        reasoning_chain += f"Step {i+1}: {step['description']}\n"
        reasoning_chain += f"Result: {step['result']}\n\n"
    
    synthesis_prompt = f"""
    Original Problem: {problem}
    
    Complete reasoning chain:
    {reasoning_chain}
    
    Based on all the reasoning steps above, provide:
    1. A clear, final answer to the original problem
    2. A brief summary of the key reasoning that led to this answer
    3. Your confidence level (High/Medium/Low) and why
    
    Format:
    FINAL ANSWER: [Your answer]
    
    REASONING SUMMARY: [Brief summary]
    
    CONFIDENCE: [High/Medium/Low] - [Explanation]
    """
    
    synthesis = llm.invoke(synthesis_prompt)
    
    # Parse the response (simplified)
    lines = synthesis.split('\n')
    final_answer = ""
    confidence = ""
    
    for line in lines:
        if line.startswith("FINAL ANSWER:"):
            final_answer = line.replace("FINAL ANSWER:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.replace("CONFIDENCE:", "").strip()
    
    state["final_answer"] = final_answer if final_answer else synthesis
    state["confidence"] = confidence if confidence else "Medium - Complete analysis performed"
    
    print("🎯 Final synthesis completed")
    
    return state

def should_continue_reasoning(state: ReasoningState) -> str:
    """Determine if more reasoning steps are needed"""
    current_step = state["current_step"]
    total_steps = len(state["reasoning_steps"])
    
    if current_step < total_steps:
        return "execute_step"
    else:
        return "synthesize"

def create_reasoning_graph():
    """Create the multi-step reasoning workflow graph"""
    
    workflow = StateGraph(ReasoningState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_problem)
    workflow.add_node("execute_step", execute_reasoning_step)
    workflow.add_node("advance", advance_step)
    workflow.add_node("synthesize", synthesize_answer)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Flow: analyze -> execute_step -> advance -> (continue or synthesize)
    workflow.add_edge("analyze", "execute_step")
    workflow.add_edge("execute_step", "advance")
    
    # Conditional edge to continue reasoning or synthesize
    workflow.add_conditional_edges(
        "advance",
        should_continue_reasoning,
        {
            "execute_step": "execute_step",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

def main():
    print("🧠 LangGraph Multi-Step Reasoning with Ollama")
    print("Enter complex problems that require step-by-step reasoning")
    print("Type 'quit' to exit\n")
    
    # Create the reasoning graph
    reasoner = create_reasoning_graph()
    
    # Example problems for testing
    examples = [
        "If a train travels 120 miles in 2 hours, and then speeds up by 25% for the next 3 hours, how far does it travel in total?",
        "A company's revenue increased by 15% in year 1, decreased by 10% in year 2, and increased by 20% in year 3. If they started with $100,000, what's their final revenue?",
        "How would you design a system to handle 1 million concurrent users for a social media app?"
    ]
    
    print("Example problems you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print()
    
    while True:
        user_input = input("Problem: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.strip() == "":
            continue
        
        try:
            # Initialize state
            initial_state = {
                "original_problem": user_input,
                "reasoning_steps": [],
                "current_step": 0,
                "final_answer": "",
                "confidence": ""
            }
            
            print("\n🔄 Starting multi-step reasoning process...")
            print("=" * 60)
            
            # Run the workflow
            result = reasoner.invoke(initial_state)
            
            print("\n" + "=" * 60)
            print("📋 REASONING COMPLETE")
            print("=" * 60)
            print(f"🎯 Final Answer: {result['final_answer']}")
            print(f"📊 Confidence: {result['confidence']}")
            print(f"🔢 Steps completed: {len(result['reasoning_steps'])}")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
