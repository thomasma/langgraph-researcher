"""
LangGraph workflow creation and configuration
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import ResearchState


def create_research_graph(
    research_agent_wrapper,
    formatter_agent_wrapper,
    validator_agent_wrapper,
    finalizer_wrapper
):
    """Create and configure the research workflow graph with parallel execution

    Workflow:
    1. Research agent gathers information
    2. Formatter and Validator run in parallel (both process raw research)
    3. Finalizer compiles the complete report

    This parallel execution improves performance by running independent agents concurrently.
    """

    # Initialize the graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("research", research_agent_wrapper)
    workflow.add_node("formatter", formatter_agent_wrapper)
    workflow.add_node("validator", validator_agent_wrapper)
    workflow.add_node("finalizer", finalizer_wrapper)

    # Define the flow with parallel execution
    # After research completes, both formatter and validator run in parallel
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "formatter")
    workflow.add_edge("research", "validator")

    # Both formatter and validator must complete before finalizer starts
    workflow.add_edge("formatter", "finalizer")
    workflow.add_edge("validator", "finalizer")
    workflow.add_edge("finalizer", END)

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
