"""
State definitions for the research workflow
"""

from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """Define the state for our multi-agent system"""
    messages: Annotated[List, add_messages]
    topic: str
    raw_research: str
    formatted_content: Dict[str, str]  # {"summary": "...", "detailed": "..."}
    validation_results: Dict[str, Any]
    final_output: str
    sources: List[str]
    validation_issues: List[str]
