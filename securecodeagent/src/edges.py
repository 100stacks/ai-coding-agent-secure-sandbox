"""Defines functions that transition the AI Coding agent from one state to another."""

from typing import Callable

from .common import GraphState

EXPECTED_NODES = [
    "generate",
    "check_node_imports",
    "check_code_execution",
    "finish",
]

def enrich(graph):
    """Adds transition edges to the graph."""

    for node_name in set(EXPECTED_NODES):
        assert node_name in graph.nodes, f"Node {node_name} not found in graph"

    graph.add_edge("generate", "check_code_imports")
    graph.add_conditional_edges(
        "check_code_imports",
        EDGE_MAP["decide_to_check_code_exec"],
        {
            "check_code_execution": "check_code_execution",
            "generate": "generate",
        },
    )
    graph.add_edge("check_code_execution", "evalute_execution")
    graph.add_conditional_edges(
        "evaluation_execution",
        EDGE_MAP["decide_to_finish"],
        {
            "finish": "finish",
            "generate": "generate",
        },
    )

    return graph

def decide_to_check_code_exec(state: GraphState) -> str:
    """
    Determines whether to test code execution, or re-try answer generation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]

    if error == "None":
        # All documents have been filtered check_relvance
        # If so, re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")

        return "check_code_execution"
    else:
        # Agent has relevant documents, so now generate answer
        print("---DECISION: RE-TRY SOLUTION---")

        return "generate"
