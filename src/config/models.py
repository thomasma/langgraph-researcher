"""
LLM model configurations for different agents
"""

import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def get_research_llm():
    """Get the LLM for the research agent"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )


def get_formatter_llm():
    """Get the LLM for the formatter agent"""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )


def get_validator_llm():
    """Get the LLM for the validator agent"""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )
