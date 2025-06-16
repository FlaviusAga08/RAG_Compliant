from pathlib import Path
from typing import Generator, List, Callable
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_unstructured import UnstructuredLoader
import textract

try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    excel_supported = True
except ImportError:
    excel_supported = False


def handle_pdf(file_path: str, filename: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return add_metadata(docs, filename)

def handle_docx(file_path: str, filename: str) -> List[Document]:
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    return add_metadata(docs, filename)

def handle_doc(file_path: str, filename: str) -> List[Document]:
    text = textract.process(file_path).decode("utf-8")
    if text:
        return [Document(page_content=text, metadata={"source": filename})]
    return []

def handle_excel(file_path: str, filename: str) -> List[Document]:
    if not excel_supported:
        return []
    loader = UnstructuredExcelLoader(file_path)
    docs = loader.load()
    return add_metadata(docs, filename)

def handle_text_like(file_path: str, filename: str) -> List[Document]:
    loader = UnstructuredLoader(file_path)
    docs = loader.load()
    return add_metadata(docs, filename)

def add_metadata(docs: List[Document], filename: str) -> List[Document]:
    for doc in docs:
        doc.metadata["source"] = filename
    return docs


# Dispatcher mapping file extensions to handlers
EXTENSION_DISPATCHER: dict[str, Callable[[str, str], List[Document]]] = {
    ".pdf": handle_pdf,
    ".docx": handle_docx,
    ".doc": handle_doc,
    ".txt": handle_text_like,
    ".md": handle_text_like,
    ".rtf": handle_text_like,
}
if excel_supported:
    EXTENSION_DISPATCHER[".xlsx"] = handle_excel


def load_documents(doc_location: str) -> Generator[Document, None, None]:
    for path in Path(doc_location).rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        filename = path.name
        file_path = str(path)

        handler = EXTENSION_DISPATCHER.get(ext)
        if handler:
            try:
                docs = handler(file_path, filename)
                for doc in docs:
                    yield doc
            except Exception as e:
                print(f"Error loading {filename}: {e}")


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)
