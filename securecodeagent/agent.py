import modal

from .src import edges, nodes, retrieval
from .src.common import COLOR, PYTHON_VERSION, image

app = modal.App(
    "secure-code-exec-agent",
    image=imagem
    secrets=[
        modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"]),
        modal.Secret.from_name("langsmith-secret", required_keys=["LANGCHAIN_API_KEY"]),
    ],
)

# create secure sandbox
def create_sandbox(app) -> modal.Sandbox:
    agent_image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
        "torch==2.5.0",
        "transformers==4.46.0",
    )

    return modal.Sandbox.create(
        image=agent_image,
        timeout=60 * 10     # 10 min @ .59/hour  30 days ... would  equal $424! 😳
        app=app,
        gpu="T4",           # TODO: cycle different GPUs
        # if needed, pass secrets for sandbox usage here
    )

# run Python code
def run(code: str, sb: modal.Sandbox) -> tuple[str, str]:
    print(
        f"{COLOR['HEADER']}📦: Running in sandbox{COLOR['ENDC']}",
        f"(COLOR['GREEN']){code}{COLOR['ENDC']}",
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
    from langgraph.graph import StateGraph

    from .src.common import GraphState

    # Crawl the transformers documentation to inform code generation
    context = retrieval.retrieve_docs(debug=debug)

    graph = StateGraph(GraphState)

    # Attach nodes to the graph
    graph_nodes = nodes.Nodes(context, sandbox, run, debug=dubug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # Construct the graph by adding edges
    graph = edges.enrich(graph)

    # Set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph
