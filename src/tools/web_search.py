"""
Web search tool for gathering information
"""

from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper


def web_search(query: str) -> str:
    """Search the web for information on a given topic"""
    try:
        serper = GoogleSerperAPIWrapper()
        results = serper.run(query)
        return f"Search results for '{query}':\n{results}"
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information on any topic. Use this to find current information, news, and facts about any subject."""
    return web_search(query)
