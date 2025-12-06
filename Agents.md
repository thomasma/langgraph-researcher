# AGENTS.md

## Project Overview

This is a **LangGraph-based multi-agent research system** that demonstrates how to build reusable, modular agent architectures. The system conducts comprehensive research on any topic, formats findings professionally, validates accuracy, and generates polished research reports.

**Architecture**: 4 specialized agents working in a sequential pipeline:
1. **Research Agent** (GPT-4o-mini) - Conducts web research with fact-checking
2. **Formatter Agent** (Llama 3.1-8b) - Structures content into professional sections
3. **Validator Agent** (GPT-OSS-20B) - Validates accuracy and flags issues
4. **Finalizer Agent** (Python) - Assembles the final markdown report

**Key Files**:
- `research.py` - Main orchestrator, creates LangGraph workflow and runs the system
- `agent_functions.py` - Contains all 4 agent implementations (model-agnostic)
- `tools.py` - Web search and fact-checking tools (Google Serper API)
- `pyproject.toml` - Dependencies and project configuration

**State Management**: Uses `ResearchState` TypedDict shared across all agents, with LangGraph's MemorySaver for conversation history.

## Setup Commands

**Prerequisites**:
- Python >= 3.13
- LangGraph >= 1.0.4
- LangChain >= 1.1.2

**Install dependencies**:
```bash
uv sync
```

**Verify installation**:
```bash
uv pip list | grep -i "langgraph\|langchain"
```

**Configure API keys** (create `.env` file):
```bash
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here  # Optional: for tracing
LANGCHAIN_PROJECT=langgraph-researcher  # Optional
```

**Run the research system**:
```bash
python research.py
```

**Interactive mode**: The system will prompt for a research topic or use the default ("Artificial Intelligence in retail sector").

## Code Style Guidelines

### General Conventions
- **Python version**: 3.13+
- **Type hints**: Use TypedDict for state objects, type annotations for function parameters
- **Formatting**: Follow PEP 8 standards
- **Imports**: Group standard library, third-party, and local imports separately

### Agent Pattern
All agents follow this structure:

```python
def agent_name(state: ResearchState, llm: BaseLanguageModel, tools: List[BaseTool] = None) -> ResearchState:
    """
    Purpose: [Clear description]

    Args:
        state: Current research state
        llm: Language model to use
        tools: Optional list of tools

    Returns:
        Updated research state
    """
    # 1. Create system prompt
    system_prompt = "..."

    # 2. Bind tools if provided
    if tools:
        llm_with_tools = llm.bind_tools(tools)

    # 3. Invoke LLM
    response = llm_with_tools.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="...")
    ])

    # 4. Extract and parse response
    # Use regex or string parsing to extract structured data

    # 5. Return updated state
    return {
        "messages": [response],
        "new_field": extracted_data,
        **state
    }
```

### Temperature Settings
- **Research tasks**: 0.3 (allows creativity in finding sources)
- **Formatting tasks**: 0.2 (structured, consistent output)
- **Validation tasks**: 0.1 (highly deterministic analysis)

### Tool Creation
Use the `@tool` decorator from LangChain:

```python
from langchain.tools import tool

@tool
def your_tool(param: str) -> str:
    """Clear description of what the tool does."""
    # Implementation
    return result
```

## Agent Development Guidelines

### Creating New Agents

1. **Add to `agent_functions.py`**:
   - Make agent model-agnostic (accept `BaseLanguageModel`)
   - Accept tools as optional parameters
   - Use clear system prompts
   - Return updated state dictionary

2. **Create wrapper in `research.py`**:
   ```python
   def your_agent_wrapper(state: ResearchState) -> ResearchState:
       return your_agent(state, your_llm, tools)
   ```

3. **Add to workflow graph**:
   ```python
   graph.add_node("your_agent", your_agent_wrapper)
   graph.add_edge("previous_agent", "your_agent")
   ```

4. **Update `ResearchState` if needed**:
   Add new fields to the TypedDict for agent outputs

### Modifying Existing Agents

- **Research Agent**: Modify system prompt in `agent_functions.py:13-81`, adjust web search strategy
- **Formatter Agent**: Update section structure in `agent_functions.py:84-155`, change markdown format
- **Validator Agent**: Modify validation criteria in `agent_functions.py:158-220`, adjust confidence scoring
- **Finalizer Agent**: Change report assembly logic in `agent_functions.py:223-263`

### Changing LLMs

To swap models, update the LLM initialization in `research.py:27-44`:

```python
# Example: Switch to different OpenAI model
research_llm = ChatOpenAI(
    model="gpt-4",  # or "gpt-3.5-turbo"
    temperature=0.3
)

# Example: Switch to different Groq model
formatter_llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.2
)
```

## Testing Instructions

### Manual Testing

Run the system with a test topic:
```bash
python research.py
# Enter topic: "Quantum computing"
```

Verify:
1. Research Agent gathers relevant sources
2. Formatter Agent creates structured sections (Summary, Detailed, Investment)
3. Validator Agent provides confidence score (1-10)
4. Finalizer Agent generates complete markdown report
5. Output file created: `research_report_*.md`

### Testing Individual Agents

```python
from agent_functions import research_agent, formatter_agent
from research import research_llm, formatter_llm
from tools import get_tools

# Test research agent
test_state = {
    "topic": "Test topic",
    "messages": []
}
result = research_agent(test_state, research_llm, get_tools())
print(result["raw_research"])

# Test formatter agent
result2 = formatter_agent(result, formatter_llm)
print(result2["formatted_content"])
```

### Validation Checks

When testing changes, ensure:
- [ ] All agents execute without errors
- [ ] State is properly passed between agents
- [ ] Output file is generated with all sections
- [ ] Sources are properly cited
- [ ] Validation confidence score is present
- [ ] Investment opportunities section is populated (if applicable)

## Security Considerations

### API Key Management
- **NEVER commit `.env` file to git** (already in `.gitignore`)
- Store all API keys in environment variables
- Required keys: `OPENAI_API_KEY`, `GROQ_API_KEY`, `SERPER_API_KEY`
- Optional: `LANGCHAIN_API_KEY` for tracing

### Tool Safety
- **Web search tool**: Rate-limited by Serper API, no direct user input injection
- **Fact check tool**: Uses web search internally, same safety considerations
- Always validate and sanitize any user input before passing to tools

### LLM Output Handling
- Agents parse LLM outputs using regex - validate extracted data before using
- Be cautious with `eval()` or `exec()` on LLM outputs (currently not used)
- Investment recommendations are AI-generated - **not financial advice**

## Commit Message Guidelines

Use conventional commit format:

```
feat: add new agent for sentiment analysis
fix: correct validation confidence score calculation
docs: update AGENTS.md with testing instructions
refactor: make research agent model-agnostic
chore: update dependencies in pyproject.toml
```

**Examples**:
- `feat(research): add multi-source fact verification`
- `fix(validator): handle empty sources list gracefully`
- `docs(agents): document temperature tuning strategy`

## Deployment Notes

### Local Development
- Uses `.env` file for configuration
- No external database required
- LangGraph MemorySaver stores state in memory (resets on restart)

### Production Considerations
- **Persistence**: Replace MemorySaver with persistent checkpointer (e.g., PostgreSQL)
- **Rate Limiting**: Implement rate limiting for API calls to prevent quota exhaustion
- **Caching**: Consider caching web search results to reduce API costs
- **Monitoring**: Enable LangSmith tracing for production debugging
- **Error Handling**: Add retry logic and fallback strategies for failed API calls

### Environment Variables for Production
```bash
# Required
OPENAI_API_KEY=prod_key
GROQ_API_KEY=prod_key
SERPER_API_KEY=prod_key

# Recommended for monitoring
LANGCHAIN_API_KEY=prod_key
LANGCHAIN_PROJECT=langgraph-researcher-prod
LANGCHAIN_TRACING_V2=true
```

## Common Tasks

### Adding a New Tool

1. Define in `tools.py`:
```python
@tool
def your_new_tool(param: str) -> str:
    """Tool description."""
    # Implementation
    return result
```

2. Add to `get_tools()`:
```python
def get_tools() -> List[BaseTool]:
    return [web_search_tool, fact_check_tool, your_new_tool]
```

3. Use in agent:
```python
def research_agent(state, llm, tools):
    llm_with_tools = llm.bind_tools(tools)
    # Agent can now use your_new_tool
```

### Changing Output Format

Modify `finalizer_agent()` in `agent_functions.py:223-263`:
- Change markdown structure
- Add/remove sections
- Adjust formatting (headers, lists, emphasis)

### Debugging Agent Behavior

Enable LangSmith tracing:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=debug-session
```

Then run the system - all LLM calls will be traced in LangSmith dashboard.

### Adjusting Research Focus

Modify the system prompt in `research_agent()`:
- Current focus: Investment opportunities (ETFs, stocks, mutual funds, REITs)
- Change prompt to focus on different aspects (technical, policy, market trends, etc.)
- Adjust investment vehicle priorities or remove investment focus entirely

## Performance Optimization

### Cost Reduction
- **Use cheaper models**: Replace GPT-4o-mini with GPT-3.5-turbo for research
- **Reduce temperature**: Lower temperatures = more deterministic = fewer retries
- **Cache results**: Cache web search results to avoid duplicate API calls
- **Batch requests**: If processing multiple topics, batch API calls where possible

### Speed Improvements
- **Parallel agents**: Modify workflow to run independent agents in parallel
- **Smaller models**: Use faster models like Llama 3.1-8b for more tasks
- **Reduce context**: Limit message history passed between agents
- **Tool optimization**: Optimize web search queries to return fewer, more relevant results

## Known Limitations

1. **Sequential Processing**: Agents run sequentially, not in parallel (by design)
2. **No Retry Logic**: Failed API calls will cause the entire pipeline to fail
3. **Memory Only**: State is stored in memory, lost on restart
4. **No Human-in-Loop**: No checkpoints for human review/approval
5. **Investment Focus**: Hardcoded focus on retail investment vehicles
6. **Rate Limits**: No built-in rate limiting for API calls

## Troubleshooting

### "API key not found" Error
- Ensure `.env` file exists in project root
- Verify API key variable names match exactly
- Check that `python-dotenv` is installed: `uv add python-dotenv`

### "Module not found" Error
- Run `uv sync` to install all dependencies
- Ensure you're using Python >= 3.13

### Empty or Malformed Output
- Check LLM responses in LangSmith tracing
- Verify regex patterns in agent functions match LLM output format
- Increase temperature slightly if outputs are too rigid
- Check that system prompts are clear and specific

### Validation Confidence Score is Low
- Review sources being used (check `state.sources`)
- Adjust research agent prompt to prioritize credible sources
- Modify validator criteria in `validator_agent()` system prompt
- Check if fact-checking tool is being used effectively

## Version History

### v1.0.0 (December 2025)
- **Upgraded to LangGraph 1.0.4** (from 0.3.18)
- **Upgraded to LangChain 1.1.2** (from 0.3.x)
- **Upgraded to LangChain Core 1.1.1**
- All integrations updated (langchain-openai 1.1.0, langchain-groq 1.1.0, langchain-community 0.4.1)
- No breaking changes required - fully backward compatible
- All tests passing with new versions
- See MIGRATION_V1.md for upgrade details

### v0.1.0 (Initial Release)
- LangGraph 0.3.18
- LangChain 0.3.x ecosystem
- 4-agent research system

## Additional Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangGraph 1.0 Release**: https://blog.langchain.com/langchain-langgraph-1dot0/
- **LangChain 1.0 Release**: https://changelog.langchain.com/announcements/langchain-1-0-now-generally-available
- **LangChain Tools**: https://python.langchain.com/docs/modules/tools/
- **Groq API**: https://console.groq.com/
- **OpenAI API**: https://platform.openai.com/
- **Serper API**: https://serper.dev/
