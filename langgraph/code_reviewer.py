"""
Code Reviewer Example with LangGraph and Ollama

This example demonstrates an automated code review workflow that analyzes
code through multiple stages: syntax, style, logic, security, and optimization.
"""

from typing import TypedDict, List, Dict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# Define the state schema
class CodeReviewState(TypedDict):
    code: str
    language: str
    review_stages: List[str]
    current_stage: int
    reviews: Dict[str, Dict[str, any]]
    overall_score: int
    final_summary: str

def detect_language(state: CodeReviewState) -> CodeReviewState:
    """Detect the programming language of the code"""
    llm = Ollama(model="llama3.2:latest")
    
    code = state["code"]
    
    detection_prompt = f"""
    Analyze this code and determine the programming language:
    
    Code:
    {code}
    
    Respond with just the programming language name (e.g., python, javascript, java, cpp, etc.)
    """
    
    language = llm.invoke(detection_prompt).strip().lower()
    
    # Set review stages based on language
    stages = ["syntax", "style", "logic", "security", "optimization"]
    
    state["language"] = language
    state["review_stages"] = stages
    state["current_stage"] = 0
    state["reviews"] = {}
    
    print(f"🔍 Detected language: {language}")
    print(f"📋 Review stages: {' → '.join(stages)}")
    
    return state

def review_syntax(state: CodeReviewState) -> CodeReviewState:
    """Review code syntax and basic structure"""
    llm = Ollama(model="llama3.2:latest")
    
    code = state["code"]
    language = state["language"]
    
    syntax_prompt = f"""
    Review this {language} code for syntax issues:
    
    Code:
    {code}
    
    Analyze for:
    1. Syntax errors
    2. Missing imports or dependencies
    3. Proper use of language constructs
    4. Basic structural issues
    
    Provide:
    - Issues found (if any)
    - Severity (High/Medium/Low)
    - Suggestions for fixes
    - Score out of 10
    
    Format:
    ISSUES: [List issues or "None found"]
    SEVERITY: [Overall severity]
    SUGGESTIONS: [Improvement suggestions]
    SCORE: [Number out of 10]
    """
    
    review = llm.invoke(syntax_prompt)
    
    # Parse review (simplified)
    score = 8  # Default
    for line in review.split('\n'):
        if line.startswith("SCORE:") and any(char.isdigit() for char in line):
            numbers = [int(s) for s in line.split() if s.isdigit()]
            if numbers:
                score = min(10, max(0, numbers[0]))
    
    state["reviews"]["syntax"] = {
        "review": review,
        "score": score,
        "stage": "syntax"
    }
    
    print(f"✅ Syntax review complete - Score: {score}/10")
    
    return state

def review_style(state: CodeReviewState) -> CodeReviewState:
    """Review code style and formatting"""
    llm = Ollama(model="llama3.2:latest")
    
    code = state["code"]
    language = state["language"]
    
    style_prompt = f"""
    Review this {language} code for style and formatting:
    
    Code:
    {code}
    
    Analyze for:
    1. Naming conventions
    2. Code formatting and indentation
    3. Comments and documentation
    4. Code organization
    5. Adherence to language style guides
    
    Provide:
    - Style issues found
    - Best practices not followed
    - Suggestions for improvement
    - Score out of 10
    
    Format:
    ISSUES: [List issues or "None found"]
    SUGGESTIONS: [Style improvement suggestions]
    SCORE: [Number out of 10]
    """
    
    review = llm.invoke(style_prompt)
    
    # Parse score
    score = 7
    for line in review.split('\n'):
        if line.startswith("SCORE:") and any(char.isdigit() for char in line):
            numbers = [int(s) for s in line.split() if s.isdigit()]
            if numbers:
                score = min(10, max(0, numbers[0]))
    
    state["reviews"]["style"] = {
        "review": review,
        "score": score,
        "stage": "style"
    }
    
    print(f"🎨 Style review complete - Score: {score}/10")
    
    return state

def review_logic(state: CodeReviewState) -> CodeReviewState:
    """Review code logic and functionality"""
    llm = Ollama(model="llama3.2:latest")
    
    code = state["code"]
    language = state["language"]
    
    logic_prompt = f"""
    Review this {language} code for logic and functionality:
    
    Code:
    {code}
    
    Analyze for:
    1. Logic errors or bugs
    2. Edge case handling
    3. Error handling
    4. Algorithm efficiency
    5. Correctness of implementation
    
    Provide:
    - Logic issues found
    - Potential bugs or edge cases
    - Suggestions for improvement
    - Score out of 10
    
    Format:
    ISSUES: [List issues or "None found"]
    BUGS: [Potential bugs]
    SUGGESTIONS: [Logic improvement suggestions]
    SCORE: [Number out of 10]
    """
    
    review = llm.invoke(logic_prompt)
    
    # Parse score
    score = 8
    for line in review.split('\n'):
        if line.startswith("SCORE:") and any(char.isdigit() for char in line):
            numbers = [int(s) for s in line.split() if s.isdigit()]
            if numbers:
                score = min(10, max(0, numbers[0]))
    
    state["reviews"]["logic"] = {
        "review": review,
        "score": score,
        "stage": "logic"
    }
    
    print(f"🧠 Logic review complete - Score: {score}/10")
    
    return state

def review_security(state: CodeReviewState) -> CodeReviewState:
    """Review code for security vulnerabilities"""
    llm = Ollama(model="llama3.2:latest")
    
    code = state["code"]
    language = state["language"]
    
    security_prompt = f"""
    Review this {language} code for security vulnerabilities:
    
    Code:
    {code}
    
    Analyze for:
    1. Input validation issues
    2. SQL injection vulnerabilities
    3. Cross-site scripting (XSS) risks
    4. Authentication/authorization flaws
    5. Data exposure risks
    6. Other security best practices
    
    Provide:
    - Security vulnerabilities found
    - Risk level (High/Medium/Low)
    - Security improvements needed
    - Score out of 10
    
    Format:
    VULNERABILITIES: [List vulnerabilities or "None found"]
    RISK_LEVEL: [Overall risk level]
    IMPROVEMENTS: [Security improvement suggestions]
    SCORE: [Number out of 10]
    """
    
    review = llm.invoke(security_prompt)
    
    # Parse score
    score = 9
    for line in review.split('\n'):
        if line.startswith("SCORE:") and any(char.isdigit() for char in line):
            numbers = [int(s) for s in line.split() if s.isdigit()]
            if numbers:
                score = min(10, max(0, numbers[0]))
    
    state["reviews"]["security"] = {
        "review": review,
        "score": score,
        "stage": "security"
    }
    
    print(f"🔒 Security review complete - Score: {score}/10")
    
    return state

def review_optimization(state: CodeReviewState) -> CodeReviewState:
    """Review code for performance and optimization"""
    llm = Ollama(model="llama3.2:latest")
    
    code = state["code"]
    language = state["language"]
    
    optimization_prompt = f"""
    Review this {language} code for performance and optimization:
    
    Code:
    {code}
    
    Analyze for:
    1. Performance bottlenecks
    2. Memory usage efficiency
    3. Algorithm complexity
    4. Resource utilization
    5. Scalability considerations
    
    Provide:
    - Performance issues found
    - Optimization opportunities
    - Scalability concerns
    - Score out of 10
    
    Format:
    ISSUES: [Performance issues or "None found"]
    OPTIMIZATIONS: [Optimization suggestions]
    SCALABILITY: [Scalability considerations]
    SCORE: [Number out of 10]
    """
    
    review = llm.invoke(optimization_prompt)
    
    # Parse score
    score = 7
    for line in review.split('\n'):
        if line.startswith("SCORE:") and any(char.isdigit() for char in line):
            numbers = [int(s) for s in line.split() if s.isdigit()]
            if numbers:
                score = min(10, max(0, numbers[0]))
    
    state["reviews"]["optimization"] = {
        "review": review,
        "score": score,
        "stage": "optimization"
    }
    
    print(f"⚡ Optimization review complete - Score: {score}/10")
    
    return state

def advance_review_stage(state: CodeReviewState) -> CodeReviewState:
    """Move to the next review stage"""
    state["current_stage"] += 1
    return state

def generate_final_summary(state: CodeReviewState) -> CodeReviewState:
    """Generate final code review summary"""
    llm = Ollama(model="llama3.2:latest")
    
    # Calculate overall score
    total_score = sum(review["score"] for review in state["reviews"].values())
    overall_score = total_score // len(state["reviews"])
    state["overall_score"] = overall_score
    
    # Compile all reviews
    all_reviews = ""
    for stage, review in state["reviews"].items():
        all_reviews += f"{stage.upper()} Review (Score: {review['score']}/10):\n"
        all_reviews += f"{review['review']}\n\n"
    
    summary_prompt = f"""
    Based on the following comprehensive code review:
    
    {all_reviews}
    
    Overall Score: {overall_score}/10
    
    Create a final summary that includes:
    1. OVERALL ASSESSMENT: High-level summary of code quality
    2. KEY STRENGTHS: What the code does well
    3. CRITICAL ISSUES: Most important issues to address
    4. PRIORITY RECOMMENDATIONS: Top 3-5 actionable improvements
    5. APPROVAL STATUS: (Approved/Needs Minor Changes/Needs Major Changes/Rejected)
    
    Be concise but comprehensive.
    """
    
    summary = llm.invoke(summary_prompt)
    state["final_summary"] = summary
    
    print("📊 Final summary generated")
    
    return state

def route_review_stage(state: CodeReviewState) -> str:
    """Route to the appropriate review stage"""
    current_stage = state["current_stage"]
    stages = state["review_stages"]
    
    if current_stage >= len(stages):
        return "summarize"
    
    stage_name = stages[current_stage]
    return f"review_{stage_name}"

def should_continue_review(state: CodeReviewState) -> str:
    """Determine if more review stages are needed"""
    if state["current_stage"] < len(state["review_stages"]):
        return "continue_review"
    else:
        return "summarize"

def create_code_review_workflow():
    """Create the code review workflow graph"""
    
    workflow = StateGraph(CodeReviewState)
    
    # Add nodes
    workflow.add_node("detect_language", detect_language)
    workflow.add_node("review_syntax", review_syntax)
    workflow.add_node("review_style", review_style)
    workflow.add_node("review_logic", review_logic)
    workflow.add_node("review_security", review_security)
    workflow.add_node("review_optimization", review_optimization)
    workflow.add_node("advance", advance_review_stage)
    workflow.add_node("summarize", generate_final_summary)
    
    # Set entry point
    workflow.set_entry_point("detect_language")
    
    # Flow: detect -> review stages -> advance -> (continue or summarize)
    workflow.add_edge("detect_language", "review_syntax")
    workflow.add_edge("review_syntax", "advance")
    
    # Conditional routing through review stages
    workflow.add_conditional_edges(
        "advance",
        should_continue_review,
        {
            "continue_review": "review_style",
            "summarize": "summarize"
        }
    )
    
    # Connect review stages
    workflow.add_edge("review_style", "advance")
    workflow.add_edge("review_logic", "advance") 
    workflow.add_edge("review_security", "advance")
    workflow.add_edge("review_optimization", "advance")
    
    # Add conditional routing for each stage
    for i, stage in enumerate(["style", "logic", "security", "optimization"]):
        if i < 3:  # Not the last stage
            workflow.add_conditional_edges(
                f"review_{stage}",
                lambda state, next_stage=["logic", "security", "optimization"][i]: f"review_{next_stage}",
                {f"review_{['logic', 'security', 'optimization'][i]}": f"review_{['logic', 'security', 'optimization'][i]}"}
            )
    
    # Simplify: direct connections
    workflow.add_edge("review_style", "review_logic")
    workflow.add_edge("review_logic", "review_security") 
    workflow.add_edge("review_security", "review_optimization")
    workflow.add_edge("review_optimization", "summarize")
    
    workflow.add_edge("summarize", END)
    
    return workflow.compile()

def main():
    print("🔍 LangGraph Code Reviewer with Ollama")
    print("Paste your code for comprehensive automated review")
    print("Type 'quit' to exit, 'example' for a sample code\n")
    
    # Create the code review workflow
    reviewer = create_code_review_workflow()
    
    # Example code snippets
    example_code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# Usage
data = [1, 2, 3, 4, 5]
avg = calculate_average(data)
print(f"Average: {avg}")
'''
    
    while True:
        print("Enter 'example' to use sample code, or paste your code:")
        user_input = input("Code (or 'example'/'quit'): ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'example':
            code_to_review = example_code
            print("Using example Python code...")
        else:
            if user_input.strip() == "":
                continue
            code_to_review = user_input
        
        try:
            # Initialize state
            initial_state = {
                "code": code_to_review,
                "language": "",
                "review_stages": [],
                "current_stage": 0,
                "reviews": {},
                "overall_score": 0,
                "final_summary": ""
            }
            
            print("\n🔄 Starting comprehensive code review...")
            print("=" * 60)
            
            # Run the workflow
            result = reviewer.invoke(initial_state)
            
            print("\n" + "=" * 60)
            print("📋 CODE REVIEW COMPLETE")
            print("=" * 60)
            print(f"🎯 Overall Score: {result['overall_score']}/10")
            print(f"💻 Language: {result['language']}")
            print(f"📊 Stages Reviewed: {len(result['reviews'])}")
            
            print("\n📝 Final Summary:")
            print(result['final_summary'])
            
            print("\n📋 Detailed Scores:")
            for stage, review in result['reviews'].items():
                print(f"  {stage.capitalize()}: {review['score']}/10")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
