# Parallel Execution Implementation

## Overview

The LangGraph workflow has been enhanced to support parallel execution of independent agents, improving performance and efficiency.

## Changes Made

### 1. State Updates ([src/graph/state.py](../src/graph/state.py))

Added custom reducer functions to properly handle state updates from parallel agents:

```python
def merge_sources(existing: List[str], new: List[str]) -> List[str]:
    """Custom reducer to merge and deduplicate sources"""

def merge_validation_issues(existing: List[str], new: List[str]) -> List[str]:
    """Custom reducer to merge validation issues"""
```

These reducers ensure that when multiple agents update list fields concurrently, the values are properly merged rather than overwritten.

**Updated State:**
- `sources`: Now uses `merge_sources` reducer for proper deduplication
- `validation_issues`: Now uses `merge_validation_issues` reducer for proper merging

### 2. Validator Agent Updates ([src/agents/validator.py](../src/agents/validator.py))

Modified the validator agent to work independently of the formatter:

**Before:**
- Validated both `raw_research` and `formatted_content`
- Depended on formatter completing first

**After:**
- Validates only `raw_research`
- Independent of formatter agent
- Can run in parallel with formatter

### 3. Workflow Updates ([src/graph/workflow.py](../src/graph/workflow.py))

Reconfigured the workflow graph to enable parallel execution:

**Before (Sequential):**
```
START → research → formatter → validator → finalizer → END
```

**After (Parallel):**
```
START → research → ┬→ formatter ┬→ finalizer → END
                   └→ validator  ┘
```

## Workflow Diagram

```
    START
      │
      ▼
   research
      │
      ├─────────────┐
      ▼             ▼
  formatter    validator    (PARALLEL EXECUTION)
      │             │
      └─────┬───────┘
            ▼
       finalizer
            │
            ▼
          END
```

## Performance Benefits

### Expected Improvements

1. **Time Savings**: Formatter and validator now run concurrently
   - Sequential: `time(formatter) + time(validator)`
   - Parallel: `max(time(formatter), time(validator))`

2. **Resource Utilization**: Better use of available compute resources

3. **Scalability**: Pattern can be extended to more parallel agents in the future

### Measured Results

From test execution ([tests/test_parallel_workflow.py](../tests/test_parallel_workflow.py)):
- Both agents start at the same timestamp
- Total execution time reduced by running agents concurrently
- All state updates properly merged

## How It Works

### LangGraph Parallel Execution

When multiple edges point from the same source node to different target nodes:

```python
workflow.add_edge("research", "formatter")
workflow.add_edge("research", "validator")
```

LangGraph automatically executes both targets in parallel.

### State Merging

When both formatter and validator complete, their state updates are merged using the defined reducers:

1. **Formatter** updates: `formatted_content`, `messages`
2. **Validator** updates: `validation_results`, `validation_issues`, `messages`
3. **Reducers** handle: `sources`, `validation_issues`, `messages`

The `add_messages` reducer (built-in) and custom reducers ensure all updates are preserved.

### Synchronization

Both agents must complete before finalizer starts:

```python
workflow.add_edge("formatter", "finalizer")
workflow.add_edge("validator", "finalizer")
```

LangGraph waits for both edges to be satisfied before executing the finalizer.

## Testing

### Run Parallel Workflow Test

```bash
python tests/test_parallel_workflow.py
```

This test:
- Verifies parallel execution with timestamps
- Checks state merging correctness
- Validates all agent outputs
- Confirms message handling

### Visualize Graph Structure

```bash
python tests/visualize_graph.py
```

Displays the graph structure and execution flow.

## Backward Compatibility

✅ **Fully backward compatible**

- All existing code continues to work
- No changes to agent function signatures
- No changes to public API
- Existing scripts work without modification

## Usage

No changes needed to use the parallel workflow. Simply use the existing API:

```python
from scripts.run_research import run_research

result = run_research("Your research topic")
```

The parallel execution happens automatically within the workflow.

## Future Enhancements

Potential additional optimizations:

1. **More Parallel Agents**: Add additional agents that can run in parallel
2. **Conditional Parallelization**: Use conditional edges to decide when to parallelize
3. **Streaming Updates**: Add streaming support to show real-time progress
4. **Dynamic Fan-Out**: Create agents dynamically based on research complexity

## Technical Details

### State Type Safety

The `ResearchState` TypedDict with `Annotated` types ensures:
- Type checking at development time
- Proper runtime behavior with reducers
- Clear documentation of state structure

### Reducer Function Contract

Reducer functions must follow this contract:
```python
def reducer(existing: T, new: T) -> T:
    """
    Args:
        existing: Current value in state
        new: New value from agent update

    Returns:
        Merged value
    """
```

### Error Handling

If either formatter or validator fails:
- The error propagates immediately
- The workflow stops
- No partial state is committed
- The other parallel agent's work is discarded

This ensures consistency and prevents incomplete results.

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [StateGraph Parallel Execution](https://langchain-ai.github.io/langgraph/concepts/low_level/#parallel-execution)
- [State Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)

---

**Last Updated**: 2025-12-06
**Version**: 1.0.0
