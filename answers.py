from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Callable, Union
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from services.vectorstore import get_vectorstore
from services.rag_service import build_qa_chain, get_qa_chain
from langchain.schema import Document

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "db")

# === FastAPI app ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore = get_vectorstore()
    build_qa_chain(api_key=API_KEY, vectorstore=vectorstore)
    yield

app = FastAPI(lifespan=lifespan)

# === Pydantic Schemas ===

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# === Dispatcher Handlers ===

def handle_dict_result(result: dict) -> QueryResponse:
    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])
    sources = [{"source": doc.metadata.get("source", "unknown")} for doc in source_docs if isinstance(doc, Document)]
    return QueryResponse(answer=answer, sources=sources)

def handle_str_result(result: str) -> QueryResponse:
    return QueryResponse(answer=result, sources=[])

# === Result Dispatcher ===

ResultHandler = Callable[[Union[dict, str]], QueryResponse]

result_dispatch: Dict[type, ResultHandler] = {
    dict: handle_dict_result,
    str: handle_str_result,
}

# === Route ===

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    qa_chain = get_qa_chain()
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA chain not initialized")

    result = qa_chain.run(request.query)
    result_type = type(result)

    handler = result_dispatch.get(result_type)
    if not handler:
        raise HTTPException(status_code=500, detail=f"Unsupported result type: {result_type}")

    return handler(result)
