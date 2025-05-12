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

        ## Prompt
        template = """
You are a coding assistant with expertise in Python.
You are able to execute Python code in a secure sandbox environment.
You are tasked with responding to the following user question: {question}
Your response will be shown to the user.
Here is a full set of documentation:

---------
{context}
---------

Answer the user question based on the above provided documentation.
Ensure any code you provide can be executed with all required imports and variables defined.
Structure your answer as a description fo the code solution,
then a list of the imports, and then finally list the functioning code block.
Here is the user question again:

--- --- ---
{question}
"""

        ## Generation
        if "error" in state_dict:
            print("---RE-GENERATE SOLUTION with ERROR FEEDBACK---")

            error = state_dict["error"]
            code_solution = state_dict["generation"]

            # Update prompt
            addendum = """You previously tried to solve this problem.  Here is your solution:

{generation}

Here is the resulting error from code execution:

{error}

Please retry to answer this.  Structure your answer with a description of the code solution.
Then list the imports. And finally list the functioning code block.  Structure your answer
with a description of the code solution.  Then list the imports. And finally list the
functioning code block.

Here is the user question:

{question}
"""
            ## Prompt
            template = template + addendum

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question", "generation", "error"],
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                    "generation": itemgetter("generation"),
                    "error": itemgetter("error"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke(
                {
                    "question": question,
                    "generation": str(code_solution[0]),
                    "error": error,
                }
            )

        else:
            print("---GENERATE SOLUTION---")

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke({"question": question})

            iter = iter + 1

            return {
                "keys": {
                    "generation": code_solution,
                    "question": question,
                    "iterations": iter,
                }
            }

    def check_code_imports(self, state: GraphState) -> GraphState:
        """
        Check imports

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE IMPORTS---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        iter = state_dict["iterations"]

        # Attempt to execute the imports
        output, error = self.run(imports, self.sb)

        if error:
            print("---CODE IMPORT CHECK: FAILED ⚠️⛔️ ---")

            # Catch any error during execution (e.g., ImportError, SyntaxError)
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)

            if "error" in state_dict:
                error_prev_runs = state_dict["error"]

                # Error Output prompt to send to Agent
                error = f"""
{error_prev_runs}

--- Most recent run output and error ---
------ output ------
{output}
------ error ------
{error}
"""
        else:
            print("---CODE IMPORT CHECK: SUCCESS ✅---")

            # No error occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "iterations": iter,
            }
        }
