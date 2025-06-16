from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Optional, Callable, Dict, Type
import os

class VectorStore:
    def __init__(self, persist_dir: str = "db", api_key: Optional[str] = None):
        self.persist_dir = persist_dir
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.vectordb = None

    def _handle_chroma_failure(self, error: Exception):
        # Fallback to empty store
        print(f"Chroma load failed: {error}. Falling back to empty store.")
        self.vectordb = Chroma.from_documents([], embedding=self.embedding, persist_directory=self.persist_dir)

    def load(self):
        # Dispatcher map for error handling strategies
        error_dispatch: Dict[Type[BaseException], Callable[[Exception], None]] = {
            Exception: self._handle_chroma_failure
        }

        try:
            self.vectordb = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding)
        except Exception as e:
            handler = error_dispatch.get(type(e), self._handle_chroma_failure)
            handler(e)

    def add_documents(self, documents: List[Document]):
        if not self.vectordb:
            self.load()
        self.vectordb.add_documents(documents)
        self.vectordb.persist()

    def retriever(self, k: int = 5):
        if not self.vectordb:
            self.load()
        return self.vectordb.as_retriever(search_kwargs={"k": k})

# Singleton instance
_vectorstore_instance = None

def get_vectorstore():
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStore()
        _vectorstore_instance.load()
    return _vectorstore_instance
