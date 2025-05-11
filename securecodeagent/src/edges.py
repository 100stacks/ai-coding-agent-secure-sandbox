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
