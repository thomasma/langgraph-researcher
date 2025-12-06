"""
Demo script showing various uses of the research system
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from run_research import run_research


def main():
    """Run demonstration examples"""

    print("=" * 60)
    print("LangGraph Multi-Agent Research System - Demo")
    print("=" * 60)

    demo_topics = [
        "Electric Vehicles market trends 2024",
        "Quantum Computing applications",
        "Renewable Energy investments"
    ]

    print("\nDemo Topics:")
    for i, topic in enumerate(demo_topics, 1):
        print(f"  {i}. {topic}")

    choice = input("\nSelect a topic (1-3) or press Enter for custom: ").strip()

    if choice in ['1', '2', '3']:
        topic = demo_topics[int(choice) - 1]
    else:
        topic = input("Enter your custom research topic: ").strip()
        if not topic:
            print("No topic provided. Exiting.")
            return

    print(f"\n🚀 Starting research on: {topic}\n")

    result = run_research(topic)

    if "error" not in result:
        print("\n✅ Research completed successfully!")
        print(f"Generated report with {len(result['final_output'])} characters")
        print(f"Found {len(result['sources'])} sources")
    else:
        print(f"\n❌ Research failed: {result['error']}")


if __name__ == "__main__":
    main()
