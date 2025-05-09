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
