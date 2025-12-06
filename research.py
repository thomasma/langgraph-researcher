"""
Multi-Agent Research System using LangGraph
Three agents: Research → Formatter → Validator
Each using different models for independent reasoning
"""

from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
import re

# Import reusable modules
from agent_functions import research_agent, formatter_agent, validator_agent, finalizer_agent
from tools import get_tools

# Load environment variables
load_dotenv(override=True)

# Get tools
tools = get_tools()

# Define the state for our multi-agent system
class ResearchState(TypedDict):
    messages: Annotated[List, add_messages]
    topic: str
    raw_research: str
    formatted_content: Dict[str, str]  # {"summary": "...", "detailed": "..."}
    validation_results: Dict[str, Any]
    final_output: str
    sources: List[str]
    validation_issues: List[str]

# Initialize different models for each agent
research_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3
)

formatter_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

validator_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

def sanitize_topic(topic: str) -> str:
    """
    Sanitize user input to prevent prompt injection attacks

    Args:
        topic: User-provided research topic

    Returns:
        Sanitized topic string

    Raises:
        ValueError: If topic fails validation checks
    """
    # 1. Length validation
    if len(topic) > 200:
        raise ValueError("Topic must be less than 200 characters")

    # 2. Remove control characters and normalize whitespace
    # Replace newlines/carriage returns with spaces (breaks XML tag strategy)
    topic = topic.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove other control characters
    topic = ''.join(char for char in topic if char.isprintable() or char.isspace())

    # Normalize multiple spaces to single space
    topic = ' '.join(topic.split())

    # 3. Detect common prompt injection patterns (case-insensitive)
    injection_patterns = [
        r'\bignore\s+previous\s+instructions?\b',
        r'\bignore\s+all\s+previous\b',
        r'\bforget\s+about\b',
        r'\bforget\s+everything\b',
        r'\byou\s+are\s+now\b',
        r'\binstead\s+of\b',
        r'\bdisregard\b',
        r'\bnew\s+instructions?\b',
        r'\bsystem\s*:\s*\b',
        r'\boverride\b',
        r'\bpretend\s+to\s+be\b',
    ]

    topic_lower = topic.lower()
    for pattern in injection_patterns:
        if re.search(pattern, topic_lower):
            raise ValueError(f"Topic contains suspicious command pattern: '{pattern}'")

    # 4. Block markdown/XML structure injection
    # Check for excessive markdown headers
    if topic.count('#') > 2:
        raise ValueError("Topic contains too many '#' characters")

    # Check for code blocks
    if '```' in topic or '<script>' in topic.lower():
        raise ValueError("Topic contains code block or script tags")

    # Check for XML/HTML tag-like patterns that could break our XML strategy
    if re.search(r'</?\w+>', topic):
        raise ValueError("Topic contains XML/HTML-like tags")

    # 5. Character whitelist validation (relaxed for legitimate research topics)
    # Allow: letters, numbers, spaces, and common punctuation
    if not re.match(r'^[a-zA-Z0-9\s\.,\-&()\'":;!?]+$', topic):
        raise ValueError("Topic contains invalid characters. Use only letters, numbers, and basic punctuation.")

    # 6. Final cleanup
    topic = topic.strip()

    if len(topic) < 3:
        raise ValueError("Topic must be at least 3 characters long")

    return topic

# Create wrapper functions that pass the LLMs to the agent functions
def research_agent_wrapper(state: ResearchState) -> ResearchState:
    """Wrapper for research agent with LLM"""
    return research_agent(state, research_llm, tools)

def formatter_agent_wrapper(state: ResearchState) -> ResearchState:
    """Wrapper for formatter agent with LLM"""
    return formatter_agent(state, formatter_llm, tools)

def validator_agent_wrapper(state: ResearchState) -> ResearchState:
    """Wrapper for validator agent with LLM"""
    return validator_agent(state, validator_llm, tools)

def finalizer_wrapper(state: ResearchState) -> ResearchState:
    """Wrapper for finalizer agent"""
    return finalizer_agent(state)

# Create the LangGraph workflow
def create_research_graph():
    """Create and configure the research workflow graph"""
    
    # Initialize the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("research", research_agent_wrapper)
    workflow.add_node("formatter", formatter_agent_wrapper)
    workflow.add_node("validator", validator_agent_wrapper)
    workflow.add_node("finalizer", finalizer_wrapper)
    
    # Define the flow
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "formatter")
    workflow.add_edge("formatter", "validator")
    workflow.add_edge("validator", "finalizer")
    workflow.add_edge("finalizer", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def run_research(topic: str, thread_id: str = "research_session") -> Dict[str, Any]:
    """Run the complete research workflow for a given topic"""

    # SECURITY: Sanitize topic input before processing
    try:
        topic = sanitize_topic(topic)
    except ValueError as e:
        print(f"❌ Invalid topic: {str(e)}")
        return {"error": f"Invalid topic: {str(e)}"}

    print(f"🚀 Starting research on: {topic}")

    # Create the graph
    graph = create_research_graph()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=f"Research topic: {topic}")],
        "topic": topic,
        "raw_research": "",
        "formatted_content": {},
        "validation_results": {},
        "final_output": "",
        "sources": [],
        "validation_issues": []
    }
    
    # Configuration for the thread
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the workflow
    try:
        result = graph.invoke(initial_state, config=config)
        print("✅ Research completed successfully!")
        return result
    except Exception as e:
        print(f"❌ Research failed: {str(e)}")
        return {"error": str(e)}

def main():
    """Main function to demonstrate the research system"""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in your .env file")
        return
    
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment variables")
        print("Please set your Groq API key in your .env file")
        return
    
    if not os.getenv("SERPER_API_KEY"):
        print("Error: SERPER_API_KEY not found in environment variables")
        print("Please set your Serper API key in your .env file")
        return
    
    # Example usage
    topic = input("Enter research topic: ").strip()
    if not topic:
        topic = "Artificial Intelligence in retail sector"
        print(f"Using default topic: {topic}")

    # Validate topic before running research
    try:
        sanitized_topic = sanitize_topic(topic)
        print(f"✓ Topic validated: {sanitized_topic}")
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTips for valid topics:")
        print("  - Keep it under 200 characters")
        print("  - Use only letters, numbers, and basic punctuation")
        print("  - Avoid special characters like #, <, >, or code blocks")
        print("  - Don't include instructions or commands")
        return

    # Run research
    result = run_research(topic)
    
    if "error" not in result:
        print("\n" + "="*50)
        print("FINAL RESEARCH REPORT")
        print("="*50)
        print(result["final_output"])
        
        # Save to file
        filename = f"research_report_{topic.replace(' ', '_').lower()}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(result["final_output"])
        print(f"\n📄 Report saved to: {filename}")
    else:
        print(f"Research failed: {result['error']}")

if __name__ == "__main__":
    main()
