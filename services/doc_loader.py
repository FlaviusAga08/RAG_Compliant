import os
from pathlib import Path
from typing import Generator, List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_unstructured import UnstructuredLoader
import textract

def add_metadata(docs: List[Document], filename: str) -> List[Document]:
    for doc in docs:
        doc.metadata["source"] = filename
    return docs

def load_pdf(file_path: str) -> List[Document]:
    return PyPDFLoader(file_path).load()

def load_docx(file_path: str) -> List[Document]:
    return Docx2txtLoader(file_path).load()

def load_doc(file_path: str, filename: str) -> Generator[Document, None, None]:
    text = textract.process(file_path).decode("utf-8")
    if text:
        yield Document(page_content=text, metadata={"source": filename})

def load_text(file_path: str) -> List[Document]:
    return UnstructuredLoader(file_path).load()

def process_file(file_path: str, filename: str) -> Generator[Document, None, None]:
    ext = Path(file_path).suffix.lower()

    match ext:
        case ".pdf":
            docs = load_pdf(file_path)
        case ".docx":
            docs = load_docx(file_path)
        case ".doc":
            yield from load_doc(file_path, filename)
            return
        case ".txt" | ".md" | ".rtf":
            docs = load_text(file_path)
        case _:
            return

    for doc in add_metadata(docs, filename):
        yield doc

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def process_and_upsert(file_path: str, vectorstore) -> None:
    filename = Path(file_path).name
    documents = list(process_file(file_path, filename))
    if documents:
        chunks = split_documents(documents)
        vectorstore.add_documents(chunks)

def process_directory(doc_location: str, vectorstore) -> None:
    for path in Path(doc_location).rglob("*"):
        if path.is_file():
            process_and_upsert(str(path), vectorstore)

def run_indexing_from_env(vectorstore):
    load_dotenv()
    doc_location = os.getenv("DOC_LOCATION")
    if not doc_location:
        raise ValueError("DOC_LOCATION not set in .env")
    process_directory(doc_location, vectorstore)