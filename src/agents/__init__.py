"""
Agent modules for the multi-agent research system
"""

from .research import research_agent
from .formatter import formatter_agent
from .validator import validator_agent
from .finalizer import finalizer_agent

__all__ = [
    'research_agent',
    'formatter_agent',
    'validator_agent',
    'finalizer_agent'
]
