from .common import COLOR

hf_url = "https://huggingface.co/docs/transformers/index"
# docs_url = "https://langchain-ai.github.io/langgraph/concepts/why-langgraph/"
anthropic_url = "https://www.anthropic.com/engineering/building-effective-agents"
langchain_lcel_url = "https://python.langchain.com/docs/concepts/lcel/"
agents_url = "https://huyenchip.com/2025/01/07/agents.html"
docs_url = hf_url

def retrieve_docs(url: str=docs_url, debug=False):
    """
    RAG - Retrieve docs for agent
    """
    from bs4 import BeautifulSoup as Soup
    from langchain_community.document_loaders.recursive_url_loader import (
        RecursiveUrlLoader,
    )

    print(f"{COLOR['HEADER']}📜: Retrieving documents from {url}{COLOR['ENDC']}")
    loader = RecursiveUrlLoader(
        url=docs_url,
        max_depth=2 // (int(debug) + 1),  # retrieve fewer docs in debug mode
        extractor=lambda x: Soup(x, "html.parser").text,
    )
    docs = loader.load()

    # sort the list based on the URLs
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"], reverse=True)

    # combine them all together
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [
            "## " + doc.metadata["source"] + "\n\n" + doc.page_content.strip()
            for doc in d_sorted
        ]
    )

    print(
        f"{COLOR['HEADER']}📜: Retrived {len(docs)} documents{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{concatenated_content[:100].strip()}{COLOR['ENDC']}",
        sep="\n",
    )

    if debug:
        print(
            f"{COLOR['HEADER']}📜: DEBUG MODE - Restricting to at most 5,000 characters{COLOR['ENDC']}"
        )
        concatenated_content = concatenated_content[:5000]

    return concatenated_content
