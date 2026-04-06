import re
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from app.config import get_settings

settings = get_settings()

#singleton embedding model
_embedding_model = None

def get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEndpointEmbeddings(
            model=settings.embedding_model,
            huggingfacehub_api_token=settings.huggingfacehub_api_token
        )
    return _embedding_model


#cleaing the text for BM25 chunks
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# build
def build_vectorstore(chunks: list[Document]) -> FAISS:
    vectorstore = FAISS.from_documents(chunks, get_embedding_model())
    print(f"Vector store built — {len(chunks)} chunks")
    return vectorstore


#clean chunks before BM25
def build_bm25(chunks: list[Document]) -> BM25Retriever:
    cleaned_chunks = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata
        )
        for doc in chunks
    ]

    bm25 = BM25Retriever.from_documents(cleaned_chunks)
    bm25.k = settings.bm25_k
    print(f"BM25 retriever built")
    return bm25


# BM25 checking signal
def has_bm25_signal(query: str, bm25: BM25Retriever) -> bool:
    tokenized = clean_text(query).split()
    scores = bm25.vectorizer.get_scores(tokenized)
    return any(score > 0.0 for score in scores)


# hybrid retrieval 
def hybrid_retrieve(
    query: str,
    vectorstore: FAISS,
    bm25: BM25Retriever,
) -> tuple[list[Document], str]:

    v_docs = vectorstore.similarity_search(query, k=settings.vector_k)
    b_docs = bm25.invoke(clean_text(query))   # ✅ cleaned query here
    signal = has_bm25_signal(query, bm25)

    rrf_scores: dict[str, dict] = {}

    for rank, doc in enumerate(v_docs):
        key = doc.page_content
        rrf_scores.setdefault(key, {"doc": doc, "score": 0.0})
        rrf_scores[key]["score"] += 1 / (rank + 1 + settings.rrf_k_vector)

    mode = "vector_only"
    if signal:
        mode = "hybrid"
        for rank, doc in enumerate(b_docs):
            key = doc.page_content
            rrf_scores.setdefault(key, {"doc": doc, "score": 0.0})
            rrf_scores[key]["score"] += 1 / (rank + 1 + settings.rrf_k_bm25)

    sorted_docs = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [item["doc"] for item in sorted_docs[:settings.final_top_k]], mode