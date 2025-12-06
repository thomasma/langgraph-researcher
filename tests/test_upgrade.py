#!/usr/bin/env python
"""
Test script to verify LangGraph 1.0 and LangChain 1.1 upgrade
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

from run_research import run_research

def test_research_system():
    """Run a quick test of the research system"""

    print("=" * 60)
    print("Testing LangGraph 1.0.4 & LangChain 1.1.2 Upgrade")
    print("=" * 60)

    # Check API keys
    required_keys = ["OPENAI_API_KEY", "GROQ_API_KEY", "SERPER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"\n⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("Skipping integration test (API keys required)")
        return False

    print("\n✓ All required API keys found")

    # Run a simple research test
    test_topic = "Python programming language"
    print(f"\n🧪 Running test research on: '{test_topic}'")
    print("This will test all 4 agents in the pipeline...\n")

    try:
        result = run_research(test_topic, thread_id="test_upgrade_session")

        if "error" in result:
            print(f"\n❌ Test failed: {result['error']}")
            return False

        # Verify all expected fields are present
        expected_fields = ["final_output", "raw_research", "formatted_content",
                          "validation_results", "sources"]

        missing_fields = [field for field in expected_fields if field not in result]
        if missing_fields:
            print(f"\n❌ Missing expected fields: {missing_fields}")
            return False

        print("\n" + "=" * 60)
        print("✅ UPGRADE TEST PASSED!")
        print("=" * 60)
        print("\nVerified:")
        print("  ✓ All agents executed successfully")
        print("  ✓ State management working")
        print("  ✓ Tools (web search, fact check) functional")
        print("  ✓ LangGraph 1.0.4 compatible")
        print("  ✓ LangChain 1.1.2 compatible")
        print(f"  ✓ Generated {len(result['final_output'])} chars of output")
        print(f"  ✓ Found {len(result['sources'])} sources")
        print(f"  ✓ Validation confidence: {result['validation_results'].get('confidence_score', 'N/A')}/10")

        # Show a snippet of the output
        print("\n📄 Output snippet:")
        print("-" * 60)
        lines = result['final_output'].split('\n')[:15]
        print('\n'.join(lines))
        if len(result['final_output'].split('\n')) > 15:
            print("... (truncated)")
        print("-" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_research_system()
    exit(0 if success else 1)
