"""
Fact checking tool for verifying claims
"""

from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper


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
def fact_check_tool(claim: str) -> str:
    """Verify facts and claims by searching for verification. Use this to check if a specific statement or fact is accurate."""
    return fact_check(claim)
