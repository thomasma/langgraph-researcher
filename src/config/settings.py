"""
Application settings and environment configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


def check_required_env_vars() -> tuple[bool, list[str]]:
    """
    Check if all required environment variables are set

    Returns:
        Tuple of (all_set: bool, missing_vars: list[str])
    """
    required_vars = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "SERPER_API_KEY"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    return len(missing_vars) == 0, missing_vars


def get_output_directory() -> str:
    """Get the directory for saving research reports"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
