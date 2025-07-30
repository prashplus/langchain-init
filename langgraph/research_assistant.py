"""
Research Assistant Example with LangGraph and Ollama

This example demonstrates a more complex workflow that acts as a research assistant,
gathering information, analyzing it, and producing comprehensive summaries.
"""

from typing import TypedDict, List, Dict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# Define the state schema
class ResearchState(TypedDict):
    research_topic: str
    research_questions: List[str]
    findings: List[Dict[str, str]]
    current_question_index: int
    summary: str
    recommendations: str
    confidence_score: int

def generate_research_questions(state: ResearchState) -> ResearchState:
    """Generate specific research questions for the topic"""
    llm = Ollama(model="llama3.2:latest")
    
    topic = state["research_topic"]
    
    questions_prompt = f"""
    You are a research assistant. For the topic "{topic}", generate 5 specific, focused research questions that would help create a comprehensive understanding of the subject.
    
    Make the questions:
    1. Specific and actionable
    2. Cover different aspects of the topic
    3. Progressive in complexity
    
    Format your response as a numbered list:
    1. [Question 1]
    2. [Question 2]
    ...
    """
    
    response = llm.invoke(questions_prompt)
    
    # Parse questions (simplified)
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove numbering
            if '.' in line:
                question = line.split('.', 1)[1].strip()
            else:
                question = line.strip('- ').strip()
            if question:
                questions.append(question)
    
    state["research_questions"] = questions[:5]  # Limit to 5 questions
    state["current_question_index"] = 0
    state["findings"] = []
    
    print(f"🔍 Generated {len(questions)} research questions:")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")
    
    return state

def research_question(state: ResearchState) -> ResearchState:
    """Research a specific question"""
    llm = Ollama(model="llama3.2:latest")
    
    if state["current_question_index"] >= len(state["research_questions"]):
        return state
    
    current_question = state["research_questions"][state["current_question_index"]]
    topic = state["research_topic"]
    
    research_prompt = f"""
    Research Topic: {topic}
    
    Specific Research Question: {current_question}
    
    Provide comprehensive information about this specific question. Include:
    1. Key facts and information
    2. Different perspectives if applicable
    3. Important details and context
    4. Any relevant examples or case studies
    
    Be thorough and accurate in your research response.
    """
    
    finding = llm.invoke(research_prompt)
    
    # Store the finding
    state["findings"].append({
        "question": current_question,
        "finding": finding,
        "index": state["current_question_index"]
    })
    
    print(f"📚 Researched question {state['current_question_index'] + 1}: {current_question[:50]}...")
    
    return state

def advance_research(state: ResearchState) -> ResearchState:
    """Move to the next research question"""
    state["current_question_index"] += 1
    return state

def synthesize_findings(state: ResearchState) -> ResearchState:
    """Synthesize all research findings into a comprehensive summary"""
    llm = Ollama(model="llama3.2:latest")
    
    topic = state["research_topic"]
    
    # Compile all findings
    all_findings = ""
    for i, finding in enumerate(state["findings"]):
        all_findings += f"Research Question {i+1}: {finding['question']}\n"
        all_findings += f"Findings: {finding['finding']}\n\n"
    
    synthesis_prompt = f"""
    Research Topic: {topic}
    
    All Research Findings:
    {all_findings}
    
    Based on all the research findings above, create:
    
    1. COMPREHENSIVE SUMMARY: A well-structured summary that integrates all findings
    2. KEY INSIGHTS: The most important insights discovered
    3. CONNECTIONS: How different aspects of the research connect to each other
    4. IMPLICATIONS: What this research means or implies
    
    Format your response with clear sections.
    """
    
    summary = llm.invoke(synthesis_prompt)
    state["summary"] = summary
    
    print("📊 Synthesizing all research findings...")
    
    return state

def generate_recommendations(state: ResearchState) -> ResearchState:
    """Generate actionable recommendations based on research"""
    llm = Ollama(model="llama3.2:latest")
    
    topic = state["research_topic"]
    summary = state["summary"]
    
    recommendations_prompt = f"""
    Research Topic: {topic}
    
    Research Summary:
    {summary}
    
    Based on this research, provide:
    
    1. ACTIONABLE RECOMMENDATIONS: 3-5 specific actions someone could take
    2. NEXT STEPS: What should be researched or explored further
    3. POTENTIAL CHALLENGES: What obstacles or challenges to be aware of
    4. CONFIDENCE ASSESSMENT: How confident are you in these findings (1-10 scale)
    
    Be practical and specific in your recommendations.
    """
    
    recommendations = llm.invoke(recommendations_prompt)
    
    # Extract confidence score (simplified)
    confidence = 7  # Default
    for line in recommendations.split('\n'):
        if 'confidence' in line.lower() and any(char.isdigit() for char in line):
            numbers = [int(s) for s in line.split() if s.isdigit()]
            if numbers:
                confidence = min(10, max(1, numbers[0]))
    
    state["recommendations"] = recommendations
    state["confidence_score"] = confidence
    
    print("💡 Generated recommendations and next steps...")
    
    return state

def should_continue_research(state: ResearchState) -> str:
    """Determine if more research questions need to be answered"""
    if state["current_question_index"] < len(state["research_questions"]):
        return "research_more"
    else:
        return "synthesize"

def create_research_workflow():
    """Create the research assistant workflow graph"""
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("generate_questions", generate_research_questions)
    workflow.add_node("research_question", research_question)
    workflow.add_node("advance", advance_research)
    workflow.add_node("synthesize", synthesize_findings)
    workflow.add_node("recommend", generate_recommendations)
    
    # Set entry point
    workflow.set_entry_point("generate_questions")
    
    # Flow: generate -> research -> advance -> (continue or synthesize) -> recommend
    workflow.add_edge("generate_questions", "research_question")
    workflow.add_edge("research_question", "advance")
    
    # Conditional edge to continue research or move to synthesis
    workflow.add_conditional_edges(
        "advance",
        should_continue_research,
        {
            "research_more": "research_question",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("synthesize", "recommend")
    workflow.add_edge("recommend", END)
    
    return workflow.compile()

def main():
    print("🔬 LangGraph Research Assistant with Ollama")
    print("Enter a research topic and get a comprehensive analysis")
    print("Type 'quit' to exit\n")
    
    # Create the research workflow
    researcher = create_research_workflow()
    
    # Example topics for testing
    examples = [
        "The impact of artificial intelligence on employment",
        "Sustainable energy solutions for urban areas",
        "The psychology of remote work productivity",
        "Blockchain technology in healthcare",
        "Climate change adaptation strategies"
    ]
    
    print("Example research topics you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print()
    
    while True:
        user_input = input("Research Topic: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.strip() == "":
            continue
        
        try:
            # Initialize state
            initial_state = {
                "research_topic": user_input,
                "research_questions": [],
                "findings": [],
                "current_question_index": 0,
                "summary": "",
                "recommendations": "",
                "confidence_score": 0
            }
            
            print(f"\n🔄 Starting comprehensive research on: {user_input}")
            print("=" * 70)
            
            # Run the workflow
            result = researcher.invoke(initial_state)
            
            print("\n" + "=" * 70)
            print("📋 RESEARCH COMPLETE")
            print("=" * 70)
            
            print(f"📊 Research Summary:")
            print(result['summary'])
            
            print(f"\n💡 Recommendations:")
            print(result['recommendations'])
            
            print(f"\n📈 Confidence Score: {result['confidence_score']}/10")
            print(f"🔢 Questions Researched: {len(result['findings'])}")
            print("-" * 70)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
