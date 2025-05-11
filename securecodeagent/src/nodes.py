import sys
from enum import Enum
from operator import itemgetter
from typing import Callable

import modal

from .common import GraphState, image

with image.imports():
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain.prompts import PromptTemplate
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

class Nodes:
    """
    Agent `brain`
    """
    def __init__(
        self,
        context: str,
        sb: modal.Sandbox,
        run: Callable[[str, modal.Sandbox], tuple[str, str]],
        debug: bool = False,
    ):
        self.context = context
        self.debug = debug
        self.model = "gpt-4o-2024-08-06" if not self.debug else "gpt-4o-mini-2024-07-18"
        self.node_map = {
            "generate": self.generate,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            "evaluate_execution": self.evaluate_execution,   # new node
            "finish": self.finish,
        }

        self.sb = sb
        self.run = run

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate a code solution based on docs and the input question
        with optional feedback from code execution tests

        Args:
            state (dict): The current graph state

        Retruns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        ## State
        state_dict = state["keys"]
        question = state_dict["question"]
        iter = state_dict["iterations"]

        ## Data model
        class Code(BaseModel):
            """Agent retrieval code output"""

            prefix: str = Field(description="Description of the problem and approach")
            imports: str = Field(description="Cide bloack import statements")
            code: str = Field(description="Code block not including import statements")

        ## LLM
        llm = ChatOpenAI(temperature=0.1, model=self.model, streaming=True)

        # Tool
        code_tool_oai = convert_to_openai_tool(Code)

        # LLM with tool and enforce invocation
        llm_with_tool = llm.bind(
            tools=[code_tool_oai],
            tool_choice={"type": "function", "function": {"name": "Code"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[Code])
