# LangGraph Multi-Agent Research System

A modular, reusable multi-agent system built with LangGraph for conducting comprehensive research with AI-powered agents.

## Features

- **Multi-Agent Pipeline**: Four specialized agents (Research, Formatter, Validator, Finalizer) working in sequence
- **Multiple LLM Providers**: Uses OpenAI and Groq for diverse reasoning capabilities
- **Security First**: Built-in prompt injection protection with comprehensive input sanitization
- **Modular Architecture**: Clean separation of concerns for easy extension and maintenance
- **Tool Integration**: Web search and fact-checking capabilities
- **Investment Research**: Specialized in identifying investment opportunities (ETFs, stocks, funds)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd langgraph-researcher

# Install dependencies using uv
uv sync
```

### Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
SERPER_API_KEY=your-serper-key
```

### Run Research

```bash
uv run python scripts/run_research.py
```

Or use the demo:

```bash
uv run python scripts/demo.py
```

## Project Structure

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
│   ├── test_security.py   # Security tests
│   ├── test_injection_e2e.py # E2E injection tests
│   └── test_upgrade.py    # Compatibility tests
├── outputs/               # Generated reports
└── docs/                  # Documentation
    ├── architecture.md    # System architecture
    ├── api.md            # API documentation
    └── deployment.md     # Deployment guide
```

## How It Works

### Agent Pipeline

1. **Research Agent** (GPT-4o-mini)
   - Conducts web searches using Serper API
   - Gathers information from multiple sources
   - Fact-checks important claims
   - Identifies investment opportunities

2. **Formatter Agent** (Llama-3.1-8b)
   - Creates executive summary
   - Organizes detailed research findings
   - Structures investment opportunities
   - Formats into professional sections

3. **Validator Agent** (GPT-OSS-20b)
   - Validates accuracy and credibility
   - Checks for biases and inconsistencies
   - Assigns confidence scores
   - Flags potential issues

4. **Finalizer Agent**
   - Compiles complete markdown report
   - Adds metadata and timestamps
   - Combines all sections
   - Saves to outputs directory

### Security

The system implements multiple layers of protection against prompt injection:

- Length validation (max 200 characters)
- Control character removal
- Pattern detection for common injection attempts
- XML/HTML tag filtering
- Character whitelist validation

## Usage Examples

### Basic Usage

```python
from scripts.run_research import run_research

result = run_research("Electric Vehicles market trends")

if "error" not in result:
    print(result["final_output"])
    print(f"Sources: {len(result['sources'])}")
else:
    print(f"Error: {result['error']}")
```

### Custom Agent Pipeline

```python
from agents import research_agent, formatter_agent
from config import get_research_llm
from tools import get_tools

llm = get_research_llm()
tools = get_tools()

state = {"topic": "Quantum Computing", "messages": []}
result = research_agent(state, llm, tools)
print(result["raw_research"])
```

## Testing

Run the test suite:

```bash
# Security tests
uv run python tests/test_security.py

# End-to-end injection tests
uv run python tests/test_injection_e2e.py

# Upgrade compatibility tests
uv run python tests/test_upgrade.py
```

## Documentation

- [Architecture](docs/architecture.md) - System design and components
- [API Documentation](docs/api.md) - Function references
- [Deployment Guide](docs/deployment.md) - Production deployment

## Technology Stack

- **LangGraph 1.0.4** - Agent orchestration and workflow
- **LangChain 1.1.2** - LLM framework
- **OpenAI GPT-4o-mini** - Research agent
- **Groq (Llama-3.1-8b, GPT-OSS-20b)** - Formatter and validator agents
- **Serper API** - Web search capabilities

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See LICENSE file for details.

## Blog Post

For more details on building reusable multi-agent systems with LangGraph, see the blog post:
https://blogs.justenougharchitecture.com/building-multi-agent-systems-with-langgraph/
