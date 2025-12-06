"""
Visualize the research workflow graph structure
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from graph import create_research_graph


def mock_agent(state):
    """Mock agent for visualization"""
    return state


def visualize_graph():
    """Visualize the graph structure"""
    print("\n" + "="*70)
    print("Research Workflow Graph Structure")
    print("="*70 + "\n")

    # Create the graph with mock agents
    graph = create_research_graph(
        mock_agent,
        mock_agent,
        mock_agent,
        mock_agent
    )

    # Get the graph structure
    print("Nodes:")
    print(f"  • {', '.join(graph.nodes.keys())}")

    print("\n" + "="*70)
    print("Execution Flow")
    print("="*70)
    print("""
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
    """)
    print("="*70)
    print("\n✅ Formatter and Validator run in PARALLEL after research completes")
    print("✅ Both must complete before Finalizer can start")
    print("="*70)


if __name__ == "__main__":
    visualize_graph()
