from typing import Any, Callable, Optional
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

_qa_chain = None

# === Retriever handler functions ===

def try_as_retriever(vs: Any) -> Optional[Any]:
    if hasattr(vs, "as_retriever"):
        return vs.as_retriever()
    return None

def try_retriever_method(vs: Any) -> Optional[Any]:
    if hasattr(vs, "retriever"):
        return vs.retriever()
    return None

def try_vectordb_as_retriever(vs: Any) -> Optional[Any]:
    if hasattr(vs, "vectordb") and hasattr(vs.vectordb, "as_retriever"):
        return vs.vectordb.as_retriever()
    return None

def get_retriever_dispatch(vs: Any) -> Any:
    strategies: list[Callable[[Any], Optional[Any]]] = [
        try_as_retriever,
        try_retriever_method,
        try_vectordb_as_retriever,
    ]

    for strategy in strategies:
        retriever = strategy(vs)
        if retriever is not None:
            return retriever

    raise AttributeError("Vectorstore has no compatible retriever method.")

# === QA Chain Builder ===

def build_qa_chain(api_key: str, vectorstore: Any):
    global _qa_chain
    if not vectorstore:
        raise ValueError("Vectorstore must be provided to build QA chain")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    retriever = get_retriever_dispatch(vectorstore)

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def get_qa_chain():
    return _qa_chain
