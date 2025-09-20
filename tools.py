"""
Reusable tools for multi-agent systems
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


def fact_check(claim: str) -> str:
    """Basic fact-checking by searching for verification"""
    try:
        serper = GoogleSerperAPIWrapper()
        verification_query = f"fact check verify: {claim}"
        results = serper.run(verification_query)
        return f"Fact-check results for '{claim}':\n{results}"
    except Exception as e:
        return f"Fact-check failed: {str(e)}"


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information on any topic. Use this to find current information, news, and facts about any subject."""
    return web_search(query)


@tool
def fact_check_tool(claim: str) -> str:
    """Verify facts and claims by searching for verification. Use this to check if a specific statement or fact is accurate."""
    return fact_check(claim)


def get_tools():
    """Get all available tools"""
    return [web_search_tool, fact_check_tool]
