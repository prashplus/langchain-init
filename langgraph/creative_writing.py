"""
Creative Writing Assistant Example with LangGraph and Ollama

This example demonstrates a creative writing workflow that uses multiple stages
to generate, refine, and enhance creative content like stories, poems, and articles.
"""

from typing import TypedDict, List, Dict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# Define the state schema
class CreativeState(TypedDict):
    prompt: str
    content_type: str
    target_length: str
    target_audience: str
    tone: str
    draft_content: str
    refined_content: str
    enhanced_content: str
    final_content: str
    feedback: Dict[str, str]

def analyze_creative_request(state: CreativeState) -> CreativeState:
    """Analyze the creative writing request to understand requirements"""
    llm = Ollama(model="llama3.2:latest")
    
    prompt = state["prompt"]
    
    analysis_prompt = f"""
    Analyze this creative writing request and determine:
    
    Request: {prompt}
    
    Determine:
    1. Content Type: (story, poem, article, dialogue, description, other)
    2. Target Length: (short, medium, long)
    3. Target Audience: (children, teens, adults, general)
    4. Tone: (serious, humorous, dramatic, inspirational, mysterious, romantic, etc.)
    
    Respond in this format:
    Content Type: [type]
    Target Length: [length]
    Target Audience: [audience]
    Tone: [tone]
    """
    
    analysis = llm.invoke(analysis_prompt)
    
    # Parse analysis (simplified)
    content_type = "story"
    target_length = "medium"
    target_audience = "general"
    tone = "neutral"
    
    for line in analysis.split('\n'):
        if line.startswith("Content Type:"):
            content_type = line.split(":")[1].strip().lower()
        elif line.startswith("Target Length:"):
            target_length = line.split(":")[1].strip().lower()
        elif line.startswith("Target Audience:"):
            target_audience = line.split(":")[1].strip().lower()
        elif line.startswith("Tone:"):
            tone = line.split(":")[1].strip().lower()
    
    state["content_type"] = content_type
    state["target_length"] = target_length
    state["target_audience"] = target_audience
    state["tone"] = tone
    
    print(f"📝 Creative request analyzed:")
    print(f"   Type: {content_type}")
    print(f"   Length: {target_length}")
    print(f"   Audience: {target_audience}")
    print(f"   Tone: {tone}")
    
    return state

def generate_draft(state: CreativeState) -> CreativeState:
    """Generate the initial creative content draft"""
    llm = Ollama(model="llama3.2:latest")
    
    prompt = state["prompt"]
    content_type = state["content_type"]
    target_length = state["target_length"]
    target_audience = state["target_audience"]
    tone = state["tone"]
    
    # Determine word count based on length
    word_counts = {
        "short": "200-400 words",
        "medium": "500-800 words", 
        "long": "1000-1500 words"
    }
    word_count = word_counts.get(target_length, "500-800 words")
    
    draft_prompt = f"""
    Create a {content_type} based on this request: {prompt}
    
    Requirements:
    - Length: {word_count}
    - Audience: {target_audience}
    - Tone: {tone}
    - Content Type: {content_type}
    
    Focus on creating engaging, original content that meets these requirements.
    This is a first draft, so prioritize creativity and getting the core ideas down.
    """
    
    draft = llm.invoke(draft_prompt)
    state["draft_content"] = draft
    
    print(f"✍️ Initial draft generated ({len(draft.split())} words)")
    
    return state

def refine_content(state: CreativeState) -> CreativeState:
    """Refine the content for better flow, structure, and clarity"""
    llm = Ollama(model="llama3.2:latest")
    
    draft = state["draft_content"]
    content_type = state["content_type"]
    tone = state["tone"]
    
    refine_prompt = f"""
    Refine this {content_type} draft to improve:
    1. Flow and pacing
    2. Sentence structure and variety
    3. Clarity and readability
    4. Consistency of tone ({tone})
    5. Overall structure and organization
    
    Original Draft:
    {draft}
    
    Provide the refined version while maintaining the original creative vision and core content.
    """
    
    refined = llm.invoke(refine_prompt)
    state["refined_content"] = refined
    
    print("🔄 Content refined for flow and structure")
    
    return state

def enhance_creativity(state: CreativeState) -> CreativeState:
    """Enhance the creative elements like imagery, dialogue, and descriptions"""
    llm = Ollama(model="llama3.2:latest")
    
    refined = state["refined_content"]
    content_type = state["content_type"]
    target_audience = state["target_audience"]
    
    enhance_prompt = f"""
    Enhance this {content_type} by improving:
    1. Vivid imagery and sensory details
    2. Character development (if applicable)
    3. Dialogue quality (if applicable)
    4. Emotional impact
    5. Creative language and word choice
    6. Engagement for {target_audience} audience
    
    Current Version:
    {refined}
    
    Enhance the creative elements while keeping the same length and core story.
    """
    
    enhanced = llm.invoke(enhance_prompt)
    state["enhanced_content"] = enhanced
    
    print("✨ Creative elements enhanced")
    
    return state

def final_polish(state: CreativeState) -> CreativeState:
    """Apply final polish for grammar, style, and impact"""
    llm = Ollama(model="llama3.2:latest")
    
    enhanced = state["enhanced_content"]
    content_type = state["content_type"]
    
    polish_prompt = f"""
    Apply final polish to this {content_type}:
    1. Check grammar and punctuation
    2. Ensure consistent style
    3. Optimize word choice
    4. Strengthen opening and closing
    5. Final review for impact and engagement
    
    Current Version:
    {enhanced}
    
    Provide the final, polished version ready for publication.
    """
    
    final = llm.invoke(polish_prompt)
    state["final_content"] = final
    
    print("🎯 Final polish applied")
    
    return state

def generate_feedback(state: CreativeState) -> CreativeState:
    """Generate constructive feedback about the creative work"""
    llm = Ollama(model="llama3.2:latest")
    
    final_content = state["final_content"]
    content_type = state["content_type"]
    original_prompt = state["prompt"]
    
    feedback_prompt = f"""
    Provide constructive feedback on this {content_type}:
    
    Original Request: {original_prompt}
    
    Final Content:
    {final_content}
    
    Analyze:
    1. How well does it fulfill the original request?
    2. Strengths of the piece
    3. Areas that could be improved
    4. Overall quality assessment
    5. Suggestions for further development
    
    Provide encouraging but honest feedback.
    """
    
    feedback = llm.invoke(feedback_prompt)
    
    state["feedback"] = {
        "detailed": feedback,
        "summary": "Creative work completed with comprehensive refinement process"
    }
    
    print("📊 Feedback generated")
    
    return state

def create_creative_writing_workflow():
    """Create the creative writing workflow graph"""
    
    workflow = StateGraph(CreativeState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_creative_request)
    workflow.add_node("draft", generate_draft)
    workflow.add_node("refine", refine_content)
    workflow.add_node("enhance", enhance_creativity)
    workflow.add_node("polish", final_polish)
    workflow.add_node("feedback", generate_feedback)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Sequential flow: analyze -> draft -> refine -> enhance -> polish -> feedback
    workflow.add_edge("analyze", "draft")
    workflow.add_edge("draft", "refine")
    workflow.add_edge("refine", "enhance")
    workflow.add_edge("enhance", "polish")
    workflow.add_edge("polish", "feedback")
    workflow.add_edge("feedback", END)
    
    return workflow.compile()

def main():
    print("🎨 LangGraph Creative Writing Assistant with Ollama")
    print("Describe what you'd like to create and get a polished creative work")
    print("Type 'quit' to exit\n")
    
    # Create the creative writing workflow
    writer = create_creative_writing_workflow()
    
    # Example prompts for testing
    examples = [
        "Write a short mystery story about a missing painting in an art gallery",
        "Create a poem about the changing seasons and the passage of time",
        "Write a dialogue between a child and their grandmother about old family recipes",
        "Create a descriptive piece about a bustling farmers market in the morning",
        "Write a humorous story about a cat who thinks it's a dog"
    ]
    
    print("Example creative prompts you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print()
    
    while True:
        user_input = input("Creative Request: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.strip() == "":
            continue
        
        try:
            # Initialize state
            initial_state = {
                "prompt": user_input,
                "content_type": "",
                "target_length": "",
                "target_audience": "",
                "tone": "",
                "draft_content": "",
                "refined_content": "",
                "enhanced_content": "",
                "final_content": "",
                "feedback": {}
            }
            
            print(f"\n🔄 Starting creative writing process...")
            print("=" * 60)
            
            # Run the workflow
            result = writer.invoke(initial_state)
            
            print("\n" + "=" * 60)
            print("📋 CREATIVE WORK COMPLETE")
            print("=" * 60)
            
            print("🎯 Final Creative Work:")
            print("-" * 40)
            print(result['final_content'])
            print("-" * 40)
            
            print(f"\n📊 Work Details:")
            print(f"   Type: {result['content_type']}")
            print(f"   Length: {result['target_length']}")
            print(f"   Audience: {result['target_audience']}")
            print(f"   Tone: {result['tone']}")
            print(f"   Word Count: ~{len(result['final_content'].split())} words")
            
            print(f"\n💬 Feedback:")
            print(result['feedback']['detailed'])
            
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2:latest is available.")

if __name__ == "__main__":
    main()
