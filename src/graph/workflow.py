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
