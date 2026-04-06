
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import get_settings

def load_pdf(pdf_path: str) -> list[Document]:
    """Load PDF → list of page Documents"""
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents → smaller chunks"""
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def load_and_chunk(pdf_path: str) -> list[Document]:
    """Full pipeline: PDF path → chunks"""
    return chunk_documents(load_pdf(pdf_path))