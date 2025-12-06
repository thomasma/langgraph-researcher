"""
Main script to run the multi-agent research system
"""

import sys
import os
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from langchain_core.messages import HumanMessage

from agents import research_agent, formatter_agent, validator_agent, finalizer_agent
from tools import get_tools
from graph import ResearchState, create_research_graph
from config import (
    get_research_llm,
    get_formatter_llm,
    get_validator_llm,
    check_required_env_vars,
    get_output_directory
)
from security import sanitize_topic


def create_agent_wrappers(research_llm, formatter_llm, validator_llm, tools):
    """Create wrapper functions that pass the LLMs to the agent functions"""

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

    return research_agent_wrapper, formatter_agent_wrapper, validator_agent_wrapper, finalizer_wrapper


def run_research(topic: str, thread_id: str = "research_session") -> Dict[str, Any]:
    """Run the complete research workflow for a given topic"""

    # SECURITY: Sanitize topic input before processing
    try:
        topic = sanitize_topic(topic)
    except ValueError as e:
        print(f"❌ Invalid topic: {str(e)}")
        return {"error": f"Invalid topic: {str(e)}"}

    print(f"🚀 Starting research on: {topic}")

    # Get LLMs and tools
    research_llm = get_research_llm()
    formatter_llm = get_formatter_llm()
    validator_llm = get_validator_llm()
    tools = get_tools()

    # Create agent wrappers
    research_wrapper, formatter_wrapper, validator_wrapper, finalizer_wrapper = create_agent_wrappers(
        research_llm, formatter_llm, validator_llm, tools
    )

    # Create the graph
    graph = create_research_graph(
        research_wrapper,
        formatter_wrapper,
        validator_wrapper,
        finalizer_wrapper
    )

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
    all_set, missing_vars = check_required_env_vars()
    if not all_set:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file")
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
        output_dir = get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"research_report_{topic.replace(' ', '_').lower()}.md")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(result["final_output"])
        print(f"\n📄 Report saved to: {filename}")
    else:
        print(f"Research failed: {result['error']}")


if __name__ == "__main__":
    main()
