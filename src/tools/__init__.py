"""
Reusable tools for multi-agent systems
"""

from .web_search import web_search_tool
from .fact_check import fact_check_tool


def get_tools():
    """Get all available tools"""
    return [web_search_tool, fact_check_tool]


__all__ = [
    'web_search_tool',
    'fact_check_tool',
    'get_tools'
]
