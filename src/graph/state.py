"""
State definitions for the research workflow
"""

from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages


def merge_sources(existing: List[str], new: List[str]) -> List[str]:
    """Custom reducer to merge and deduplicate sources"""
    if not existing:
        return new
    if not new:
        return existing
    combined = existing + new
    # Remove duplicates while preserving order
    return list(dict.fromkeys(combined))


def merge_validation_issues(existing: List[str], new: List[str]) -> List[str]:
    """Custom reducer to merge validation issues"""
    if not existing:
        return new
    if not new:
        return existing
    return existing + new


class ResearchState(TypedDict):
    """Define the state for our multi-agent system with proper reducers for parallel execution"""
    messages: Annotated[List, add_messages]
    topic: str
    raw_research: str
    formatted_content: Dict[str, str]  # {"summary": "...", "detailed": "..."}
    validation_results: Dict[str, Any]
    final_output: str
    sources: Annotated[List[str], merge_sources]  # Custom merge logic for parallel execution
    validation_issues: Annotated[List[str], merge_validation_issues]  # Custom merge logic
