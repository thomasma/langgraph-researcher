"""
Reusable agent functions for multi-agent systems
"""

from typing import Dict, Any, List
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from datetime import datetime
import re


def research_agent(
    state: Dict[str, Any], 
    llm: BaseLanguageModel, 
    tools: List[BaseTool] = None
) -> Dict[str, Any]:
    """Agent 1: Conducts comprehensive research on the topic"""
    print("🔍 Research Agent: Starting research...")
    
    topic = state["topic"]
    
    research_prompt = f"""
    You are a research specialist. Conduct comprehensive research on: {topic}
    
    Your task:
    1. Use the web_search_tool to gather information from multiple sources
    2. Look for recent developments, key facts, statistics, and expert opinions
    3. Identify credible sources and citations
    4. Look for different perspectives on the topic
    5. Note any controversial or disputed claims
    6. Use fact_check_tool to verify important claims
    7. IMPORTANT: Research and identify the top 5 investment vehicles for retail investors related to this topic
    
    For the investment vehicles section, prioritize and search for:
    - ETFs (Exchange-Traded Funds) - Focus on sector-specific, thematic, and broad market ETFs
    - Individual stocks of major companies in the sector
    - Mutual funds focused on the sector
    - REITs (Real Estate Investment Trusts) if applicable
    - Include current ticker symbols, fund names, expense ratios, and brief descriptions
    - Note recent performance data, assets under management, and analyst ratings
    - Search specifically for "ETFs" and "exchange-traded funds" related to the topic
    
    Perform multiple targeted searches to get comprehensive coverage. Provide detailed research findings with proper source attribution.
    """
    
    # Use LLM with tools if available
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    
    messages = [SystemMessage(content=research_prompt)]
    response = llm_with_tools.invoke(messages)
    
    # Extract research content
    research_content = response.content if hasattr(response, 'content') else str(response)
    
    # Extract sources from tool calls
    sources = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'web_search_tool':
                query = tool_call['args'].get('query', 'N/A')
                sources.append(f"Web search: {query}")
    
    # Add the main topic as a source if no tool calls found
    if not sources:
        sources.append(f"Web search query: {topic}")
    
    # Try to extract URLs from the research content
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, research_content)
    for url in urls[:3]:  # Limit to first 3 URLs
        sources.append(f"Source: {url}")
    
    return {
        "raw_research": research_content,
        "sources": sources,
        "messages": state["messages"] + [response]
    }


def formatter_agent(
    state: Dict[str, Any], 
    llm: BaseLanguageModel, 
    tools: List[BaseTool] = None
) -> Dict[str, Any]:
    """Agent 2: Formats research into summary and detailed sections"""
    print("📝 Formatter Agent: Structuring content...")
    
    raw_research = state["raw_research"]
    topic = state["topic"]
    
    format_prompt = f"""
    You are a content formatter. Take the research data and structure it into three clear sections.
    
    Topic: {topic}
    
    Research Data:
    {raw_research}
    
    Create three sections:
    
    1. EXECUTIVE SUMMARY (2-3 paragraphs):
    - Key findings and main points
    - Most important statistics or facts
    - Overall conclusion or implications
    
    2. DETAILED RESEARCH (comprehensive):
    - Full research findings with sources
    - Supporting evidence and data
    - Different perspectives and viewpoints
    - Specific examples and case studies
    - Citations and references
    
    3. INVESTMENT OPPORTUNITIES (if applicable):
    - Top 5 investment vehicles for retail investors (prioritize ETFs)
    - Include ETF ticker symbols, stock symbols, fund names, and descriptions
    - Recent performance data, expense ratios, and analyst ratings
    - Risk considerations and investment thesis
    - Focus on ETFs as the primary investment vehicle for retail investors
    
    Format this as a professional research report with clear section headers.
    """
    
    response = llm.invoke([SystemMessage(content=format_prompt)])
    formatted_content = response.content if hasattr(response, 'content') else str(response)
    
    # Split into summary, detailed, and investment sections
    sections = formatted_content.split("DETAILED RESEARCH")
    summary = sections[0].replace("EXECUTIVE SUMMARY", "").strip() if len(sections) > 0 else ""
    
    if len(sections) > 1:
        detailed_section = sections[1]
        # Check if there's an investment opportunities section
        if "INVESTMENT OPPORTUNITIES" in detailed_section:
            investment_split = detailed_section.split("INVESTMENT OPPORTUNITIES")
            detailed = investment_split[0].strip()
            investment = investment_split[1].strip() if len(investment_split) > 1 else ""
        else:
            detailed = detailed_section.strip()
            investment = ""
    else:
        detailed = formatted_content
        investment = ""
    
    return {
        "formatted_content": {
            "summary": summary,
            "detailed": detailed,
            "investment": investment
        },
        "messages": state["messages"] + [response]
    }


def validator_agent(
    state: Dict[str, Any], 
    llm: BaseLanguageModel, 
    tools: List[BaseTool] = None
) -> Dict[str, Any]:
    """Agent 3: Validates research for accuracy and flags issues"""
    print("✅ Validator Agent: Checking accuracy...")
    
    formatted_content = state["formatted_content"]
    raw_research = state["raw_research"]
    topic = state["topic"]
    
    validation_prompt = f"""
    You are a fact-checker and validator. Review the research for accuracy and reliability.
    
    Topic: {topic}
    
    Original Research:
    {raw_research}
    
    Formatted Content:
    Summary: {formatted_content.get('summary', '')}
    Detailed: {formatted_content.get('detailed', '')}
    
    Your tasks:
    1. Identify any claims that seem questionable or unverified
    2. Check for potential fake quotes or misattributed statements
    3. Look for outdated information or statistics
    4. Flag any biased or one-sided perspectives
    5. Verify that sources are credible and properly cited
    6. Check for logical inconsistencies
    
    Provide a validation report with:
    - Overall confidence score (1-10)
    - List of flagged issues
    - Recommendations for improvement
    - Verification status of key claims
    """
    
    # Use LLM with tools if available
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    
    response = llm_with_tools.invoke([SystemMessage(content=validation_prompt)])
    validation_content = response.content if hasattr(response, 'content') else str(response)
    
    # Extract validation issues
    validation_issues = []
    if "flagged" in validation_content.lower() or "issue" in validation_content.lower():
        # Simple extraction of issues - in a real system, you'd parse this more carefully
        validation_issues.append("Some claims may need verification")
    
    return {
        "validation_results": {
            "report": validation_content,
            "confidence_score": 8,  # Placeholder - would extract from response
            "timestamp": datetime.now().isoformat()
        },
        "validation_issues": validation_issues,
        "messages": state["messages"] + [response]
    }


def finalizer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final step: Compile the complete research report"""
    print("📋 Finalizing research report...")
    
    topic = state["topic"]
    formatted_content = state["formatted_content"]
    validation_results = state["validation_results"]
    sources = state["sources"]
    
    final_report = f"""
# Research Report: {topic}

## Executive Summary
{formatted_content.get('summary', 'No summary available')}

## Detailed Research
{formatted_content.get('detailed', 'No detailed content available')}

## Investment Opportunities
{formatted_content.get('investment', 'No investment opportunities identified')}

## Validation Report
{validation_results.get('report', 'No validation available')}

## Sources
{chr(10).join(f"- {source}" for source in sources) if sources else "No sources listed"}

## Validation Status
- Confidence Score: {validation_results.get('confidence_score', 'N/A')}/10
- Issues Found: {len(state.get('validation_issues', []))}
- Validation Date: {validation_results.get('timestamp', 'N/A')}

---
*Report generated by Multi-Agent Research System*
"""
    
    from langchain_core.messages import AIMessage
    return {
        "final_output": final_report,
        "messages": state["messages"] + [AIMessage(content="Research report completed")]
    }
