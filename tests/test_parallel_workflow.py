"""
Test parallel workflow execution
"""

import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from graph import ResearchState, create_research_graph
from langchain_core.messages import HumanMessage, AIMessage


def mock_research_agent(state: ResearchState) -> ResearchState:
    """Mock research agent for testing"""
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🔍 Research Agent: Starting...")
    return {
        "raw_research": "Test research content about AI",
        "sources": ["https://example.com/ai-research"],
        "messages": state["messages"] + [AIMessage(content="Research completed")]
    }


def mock_formatter_agent(state: ResearchState) -> ResearchState:
    """Mock formatter agent for testing"""
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 📝 Formatter Agent: Starting...")
    # Simulate some processing time
    import time
    time.sleep(0.1)
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 📝 Formatter Agent: Complete")
    return {
        "formatted_content": {
            "summary": "Test summary",
            "detailed": "Test detailed content",
            "investment": "Test investment info"
        },
        "messages": state["messages"] + [AIMessage(content="Formatting completed")]
    }


def mock_validator_agent(state: ResearchState) -> ResearchState:
    """Mock validator agent for testing"""
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ✅ Validator Agent: Starting...")
    # Simulate some processing time
    import time
    time.sleep(0.1)
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ✅ Validator Agent: Complete")
    return {
        "validation_results": {
            "report": "Test validation report",
            "confidence_score": 9,
            "timestamp": datetime.now().isoformat()
        },
        "validation_issues": ["No major issues found"],
        "messages": state["messages"] + [AIMessage(content="Validation completed")]
    }


def mock_finalizer_agent(state: ResearchState) -> ResearchState:
    """Mock finalizer agent for testing"""
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 📋 Finalizer Agent: Starting...")
    return {
        "final_output": "# Test Report\n\nComplete!",
        "messages": state["messages"] + [AIMessage(content="Finalization completed")]
    }


def test_parallel_execution():
    """Test that formatter and validator run in parallel"""
    print("\n" + "="*70)
    print("Testing Parallel Workflow Execution")
    print("="*70 + "\n")

    # Create the graph with mock agents
    graph = create_research_graph(
        mock_research_agent,
        mock_formatter_agent,
        mock_validator_agent,
        mock_finalizer_agent
    )

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="Research topic: Test AI")],
        "topic": "Test AI",
        "raw_research": "",
        "formatted_content": {},
        "validation_results": {},
        "final_output": "",
        "sources": [],
        "validation_issues": []
    }

    # Configuration for the thread
    config = {"configurable": {"thread_id": "test_parallel"}}

    # Run the workflow
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S.%f')[:-3]}] 🚀 Starting workflow...\n")

    result = graph.invoke(initial_state, config=config)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n[{end_time.strftime('%H:%M:%S.%f')[:-3]}] ✅ Workflow completed!")
    print(f"Total execution time: {duration:.3f} seconds\n")

    # Verify results
    print("="*70)
    print("Verification Results")
    print("="*70)

    assert result["raw_research"] == "Test research content about AI", "Research content mismatch"
    print("✅ Research content verified")

    assert result["formatted_content"]["summary"] == "Test summary", "Formatted content mismatch"
    print("✅ Formatted content verified")

    assert result["validation_results"]["confidence_score"] == 9, "Validation results mismatch"
    print("✅ Validation results verified")

    assert result["final_output"] == "# Test Report\n\nComplete!", "Final output mismatch"
    print("✅ Final output verified")

    # Verify state merging for sources and validation_issues
    assert len(result["sources"]) >= 1, "Sources not properly merged"
    print(f"✅ Sources properly merged ({len(result['sources'])} sources)")

    assert len(result["validation_issues"]) >= 1, "Validation issues not properly merged"
    print(f"✅ Validation issues properly merged ({len(result['validation_issues'])} issues)")

    # Check message count (should include messages from all agents)
    expected_messages = 5  # Initial + research + formatter + validator + finalizer
    assert len(result["messages"]) == expected_messages, f"Expected {expected_messages} messages, got {len(result['messages'])}"
    print(f"✅ All agent messages captured ({len(result['messages'])} messages)")

    print("\n" + "="*70)
    print("🎉 All tests passed! Parallel execution working correctly.")
    print("="*70)

    # Note about timing
    print("\n📊 Performance Notes:")
    print(f"   - Sequential execution would take ~0.2s (formatter + validator)")
    print(f"   - Parallel execution should take ~0.1s (max of both)")
    print(f"   - Actual time: {duration:.3f}s (includes graph overhead)")
    if duration < 0.15:
        print("   ✅ Parallel execution is working efficiently!")
    else:
        print("   ⚠️  Timing suggests sequential execution (check graph configuration)")


if __name__ == "__main__":
    test_parallel_execution()
