"""
Input sanitization utilities for preventing prompt injection attacks
"""

import re


def sanitize_topic(topic: str) -> str:
    """
    Sanitize user input to prevent prompt injection attacks

    Args:
        topic: User-provided research topic

    Returns:
        Sanitized topic string

    Raises:
        ValueError: If topic fails validation checks
    """
    # 1. Length validation
    if len(topic) > 200:
        raise ValueError("Topic must be less than 200 characters")

    # 2. Remove control characters and normalize whitespace
    # Replace newlines/carriage returns with spaces (breaks XML tag strategy)
    topic = topic.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove other control characters
    topic = ''.join(char for char in topic if char.isprintable() or char.isspace())

    # Normalize multiple spaces to single space
    topic = ' '.join(topic.split())

    # 3. Detect common prompt injection patterns (case-insensitive)
    injection_patterns = [
        r'\bignore\s+previous\s+instructions?\b',
        r'\bignore\s+all\s+previous\b',
        r'\bforget\s+about\b',
        r'\bforget\s+everything\b',
        r'\byou\s+are\s+now\b',
        r'\binstead\s+of\b',
        r'\bdisregard\b',
        r'\bnew\s+instructions?\b',
        r'\bsystem\s*:\s*\b',
        r'\boverride\b',
        r'\bpretend\s+to\s+be\b',
    ]

    topic_lower = topic.lower()
    for pattern in injection_patterns:
        if re.search(pattern, topic_lower):
            raise ValueError(f"Topic contains suspicious command pattern: '{pattern}'")

    # 4. Block markdown/XML structure injection
    # Check for excessive markdown headers
    if topic.count('#') > 2:
        raise ValueError("Topic contains too many '#' characters")

    # Check for code blocks
    if '```' in topic or '<script>' in topic.lower():
        raise ValueError("Topic contains code block or script tags")

    # Check for XML/HTML tag-like patterns that could break our XML strategy
    if re.search(r'</?\w+>', topic):
        raise ValueError("Topic contains XML/HTML-like tags")

    # 5. Character whitelist validation (relaxed for legitimate research topics)
    # Allow: letters, numbers, spaces, and common punctuation
    if not re.match(r'^[a-zA-Z0-9\s\.,\-&()\'":;!?]+$', topic):
        raise ValueError("Topic contains invalid characters. Use only letters, numbers, and basic punctuation.")

    # 6. Final cleanup
    topic = topic.strip()

    if len(topic) < 3:
        raise ValueError("Topic must be at least 3 characters long")

    return topic
