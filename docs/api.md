# API Documentation

## Main Functions

### `run_research(topic: str, thread_id: str = "research_session") -> Dict[str, Any]`

Runs the complete research workflow for a given topic.

**Parameters:**
- `topic` (str): The research topic to investigate
- `thread_id` (str, optional): Identifier for the conversation thread. Default: "research_session"

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `final_output` (str): Complete research report in markdown
  - `raw_research` (str): Unformatted research data
  - `formatted_content` (Dict): Structured sections (summary, detailed, investment)
  - `validation_results` (Dict): Validation report and confidence score
  - `sources` (List[str]): List of sources cited
  - `validation_issues` (List): Any flagged concerns
  - `error` (str, optional): Error message if the request failed

**Example:**
```python
from scripts.run_research import run_research

result = run_research("Artificial Intelligence in healthcare")

if "error" not in result:
    print(result["final_output"])
else:
    print(f"Error: {result['error']}")
```

## Agent Functions

### Research Agent

```python
def research_agent(state: Dict[str, Any], llm: BaseLanguageModel, tools: List[BaseTool] = None) -> Dict[str, Any]
```

Conducts comprehensive research on the topic using web search and fact-checking tools.

**Updates state with:**
- `raw_research`: Research findings
- `sources`: List of sources

### Formatter Agent

```python
def formatter_agent(state: Dict[str, Any], llm: BaseLanguageModel, tools: List[BaseTool] = None) -> Dict[str, Any]
```

Formats raw research into structured sections.

**Updates state with:**
- `formatted_content`: Dict with keys: summary, detailed, investment

### Validator Agent

```python
def validator_agent(state: Dict[str, Any], llm: BaseLanguageModel, tools: List[BaseTool] = None) -> Dict[str, Any]
```

Validates research for accuracy and flags potential issues.

**Updates state with:**
- `validation_results`: Validation report and confidence score
- `validation_issues`: List of flagged concerns

### Finalizer Agent

```python
def finalizer_agent(state: Dict[str, Any]) -> Dict[str, Any]
```

Compiles the complete research report.

**Updates state with:**
- `final_output`: Complete markdown report

## Security Functions

### `sanitize_topic(topic: str) -> str`

Sanitizes user input to prevent prompt injection attacks.

**Parameters:**
- `topic` (str): User-provided research topic

**Returns:**
- `str`: Sanitized topic string

**Raises:**
- `ValueError`: If topic fails validation checks

**Example:**
```python
from security import sanitize_topic

try:
    clean_topic = sanitize_topic(user_input)
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Configuration Functions

### `get_research_llm()`
Returns configured LLM for research agent (GPT-4o-mini).

### `get_formatter_llm()`
Returns configured LLM for formatter agent (Llama-3.1-8b).

### `get_validator_llm()`
Returns configured LLM for validator agent (GPT-OSS-20b).

### `check_required_env_vars() -> tuple[bool, list[str]]`

Checks if all required environment variables are set.

**Returns:**
- Tuple of (all_set: bool, missing_vars: list[str])

**Example:**
```python
from config import check_required_env_vars

all_set, missing = check_required_env_vars()
if not all_set:
    print(f"Missing: {missing}")
```

## Tools

### `web_search_tool(query: str) -> str`

Searches the web for information using Serper API.

### `fact_check_tool(claim: str) -> str`

Verifies facts and claims by searching for verification.
