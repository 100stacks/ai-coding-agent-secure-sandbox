import modal

from .src import edges, nodes, retrieval
from .src.common import COLOR, PYTHON_VERSION, image

app = modal.App(
    "secure-code-exec-agent",
    image=image,
    secrets=[
        modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"]),
        modal.Secret.from_name("langsmith-secret", required_keys=["LANGCHAIN_API_KEY"]),
    ],
)

# create secure sandbox
def create_sandbox(app) -> modal.Sandbox:
    """
    Secure Sandbox

    Using Modal Sandbox construct, run arbitrary potentially unsecure code.  This
    proof of concept uses the HF `transformers` library to generate text with a
    pre-trained model.
    """
    agent_image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
        "torch==2.5.0",
        "transformers==4.46.0",
    )

    return modal.Sandbox.create(
        image=agent_image,
        timeout=60 * 10,    # 10 min default. Running @ .59/hour  30 days ... would  equal $424! 😳
        app=app,
        #gpu="T4",          # lowest baseline gpu
        gpu="A10G",         # runs @ $1.10/hr - seems slightly unstable
        # gpu="L4",           # runs @ $0.80/hr - slower upstart time
        # if needed, pass secrets for sandbox usage here
    )

# run Python code
def run(code: str, sb: modal.Sandbox) -> tuple[str, str]:
    """
    Run code in sandbox

    Use Python `exec` command to run dynamically generated code without
    spinning up a new container.
    """
    print(
        f"{COLOR['HEADER']}📦: Running in sandbox{COLOR['ENDC']}",
        f"{COLOR['GREEN']}\n{code}{COLOR['ENDC']}",
        sep="\n",
    )

    exc = sb.exec("python", "-c", code)
    exc.wait()

    stdout = exc.stdout.read()
    stderr = exc.stderr.read()

    if exc.returncode != 0:
        print(
            f"{COLOR['HEADER']}📦: Failed with exitcode {sb.returncode}{COLOR['ENDC']}"
        )

    return stdout, stderr

# Construct the AI agent's graph
def construct_graph(sandbox: modal.Sandbox, debug: bool = False):
    """
    Construct the AI Agent's Graph

    Using LangGraph, construct the agent's graph.  The graph is defined in `edges`
    and `nodes` modules.

    - `Nodes`: actions that change the agent's state
    - `Edges`: transitions between nodes
    """

    from langgraph.graph import StateGraph

    from .src.common import GraphState

    # Crawl the transformers documentation to inform code generation
    context = retrieval.retrieve_docs(debug=debug)
    print("==== agent.py - context ====")
    print(context)

    graph = StateGraph(GraphState)
    print("==== agent.py - graph ====")
    print(graph)
    # Attach nodes to the graph
    graph_nodes = nodes.Nodes(context, sandbox, run, debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # Construct the graph by adding edges
    graph = edges.enrich(graph)

    # Set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph

# Set up and compile graph
DEFAULT_QUESTION = """How do I generate Python code using a pre-trained model
from the transformers library?"""

@app.function()
def go(
    question: str = DEFAULT_QUESTION,
    debug: bool = False,
):
    """
    Compiles the Python code generation agent graph and runs it.

    Returns the result.
    """
    sb = create_sandbox(app)

    # LangChain LCEL syntax
    config = {"recursion_limit": 50}
    graph = construct_graph(sb, debug=debug, config=config)
    runnable = graph.compile()
    result = runnable.invoke(
        {"keys": {"question": question, "iterations": 0}},
        # config={"recursion_limit": 50},
    )

    sb.terminate()

    return result["keys"]["response"]

# Modal local entrypoint to run from command line
@app.local_entrypoint()
def main(
    question: str = DEFAULT_QUESTION,
    debug: bool = False,
):
    """
    Sends a question to the Python code generation agent.

    Debug mode: if `True`, switch to debug mode for shorter context with smaller model.
    """
    if debug:
        if question == DEFAULT_QUESTION:
            question = "hi there, how are you?"

    print(go.remote(question, debug=debug))
