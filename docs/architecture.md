# Architecture

## System Overview

The LangGraph Multi-Agent Research System is built on a modular architecture with clear separation of concerns. The system uses LangGraph to orchestrate multiple AI agents, each with specialized roles in the research pipeline.

## Directory Structure

```
langgraph-researcher/
├── src/                    # Main source code
│   ├── agents/            # Agent implementations
│   │   ├── research.py    # Research agent
│   │   ├── formatter.py   # Formatter agent
│   │   ├── validator.py   # Validator agent
│   │   └── finalizer.py   # Finalizer agent
│   ├── tools/             # Reusable tools
│   │   ├── web_search.py  # Web search functionality
│   │   └── fact_check.py  # Fact checking
│   ├── graph/             # LangGraph workflow
│   │   ├── state.py       # State definitions
│   │   └── workflow.py    # Graph configuration
│   ├── security/          # Security utilities
│   │   └── sanitization.py # Input sanitization
│   └── config/            # Configuration
│       ├── models.py      # LLM configurations
│       └── settings.py    # App settings
├── scripts/               # Executable scripts
│   ├── run_research.py    # Main entry point
│   └── demo.py            # Demo script
├── tests/                 # Test suite
└── outputs/               # Generated reports
```

## Agent Pipeline

The system uses a linear pipeline of four specialized agents:

1. **Research Agent** (GPT-4o-mini)
   - Conducts comprehensive research
   - Uses web search and fact-checking tools
   - Gathers information from multiple sources
   - Identifies investment opportunities

2. **Formatter Agent** (Llama-3.1-8b)
   - Structures raw research into sections
   - Creates executive summary
   - Organizes detailed findings
   - Formats investment opportunities

3. **Validator Agent** (GPT-OSS-20b)
   - Validates accuracy and credibility
   - Checks for biases and inconsistencies
   - Assigns confidence scores
   - Flags potential issues

4. **Finalizer Agent**
   - Compiles the complete report
   - Combines all sections
   - Adds metadata and timestamps
   - Generates final markdown output

## State Management

The system uses a TypedDict state that flows through all agents:

```python
class ResearchState(TypedDict):
    messages: List              # Conversation history
    topic: str                 # Research topic
    raw_research: str          # Raw research data
    formatted_content: Dict    # Structured sections
    validation_results: Dict   # Validation report
    final_output: str          # Complete report
    sources: List[str]         # Source citations
    validation_issues: List    # Flagged issues
```

## Security

Input sanitization is performed at the entry point to prevent prompt injection attacks:

- Length validation
- Control character removal
- Pattern detection for injection attempts
- XML/HTML tag filtering
- Character whitelist validation

## Configuration

The system uses environment variables for API keys and supports multiple LLM providers:

- OpenAI (Research agent)
- Groq (Formatter and Validator agents)
- Serper API (Web search)

## Extensibility

The modular design allows for easy extension:

- Add new agents by creating modules in `src/agents/`
- Add new tools in `src/tools/`
- Modify the workflow in `src/graph/workflow.py`
- Add custom security checks in `src/security/`
