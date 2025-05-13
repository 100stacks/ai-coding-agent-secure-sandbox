import modal

from .agent import construct_graph, create_sandbox
from .src.common import image

app = modal.App("ai-coding-agent-secure-sandbox")

image = image.pip_install(
    "langserve[all]==0.3.1",
    "httpx>=0.23.0,<1.0",  # https://github.com/langchain-ai/langserve/blob/main/pyproject.toml#L14C1-L14C24
    "fastapi>=0.90.1,<1",  # https://github.com/langchain-ai/langserve/blob/main/pyproject.toml#L15C22-L15C35
)

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"]),
        modal.Secret.from_name("langsmith-secret", required_keys=["LANGCHAIN_API_KEY"]),
    ],
)
@modal.asgi_app()
def serve():
    """
    LangServe - FastAPI server wrapper for LangChain and LangGraph applications.
    """

    from fastapi import FastAPI, responses
    from fastapi.middleware.cors import CORSMiddleware
    from langchain_core.runnables import RunnableLambda
    from langserve import add_routes

    # Create a FastAPI app
    web_app = FastAPI(
        title="AI Coding Agent",
        version="1.0",
        description="AI Agent that generates code and checks if it runs in a secure sandbox.",
    )

    # CORS config
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    def inp(question: str) -> dict:
        """
        Question to ask the Agent.
        """
        return {"keys": {"question": question, "iterations": 0}}

    def out(state: dict) -> str:
        """
        Response from Agent to user question.
        """
        if "finish" in state:
            return state["finish"]["keys"]["response"]
        elif len(state) > 0 and "finish" in state[-1]:
            return state[-1]["finish"]["keys"]["response"]
        else:
            return str(state)

    graph = construct_graph(create_sandbox(app), debug=False).compile()

    chain = RunnableLambda(inp) | graph | RunnableLambda(out)

    add_routes(
        web_app,
        chain,
        path="/securecodeagent",
    )

    # redirect the root to the interactive playground
    @web_app.get("/")
    def redirect():
        return responses.RedirectResponse(url="/securecodeagent/playground")

    # return the FastAPI app and then Modal will deploy it
    return web_app
