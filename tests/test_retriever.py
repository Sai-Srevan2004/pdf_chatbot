import os
from app.loader import load_and_chunk
from app.retriever import build_vectorstore, build_bm25, hybrid_retrieve, has_bm25_signal

def test_hybrid_retrieval():

    #Load & validate chunks 
    chunks = load_and_chunk("SaiSrevan_Resume.pdf")

    if len(chunks) == 0:
        print("No chunks created from PDF")
        return

    print(f"Loaded {len(chunks)} chunks")

    #Build retrievers 
    vectorstore = build_vectorstore(chunks)
    bm25 = build_bm25(chunks)

    #Hybrid retrieval test 
    docs, mode = hybrid_retrieve("where is sai from?", vectorstore, bm25)

    if len(docs) == 0:
        print("No documents retrieved")
        return

    if mode not in ["hybrid", "vector_only"]:
        print(f"Invalid mode: {mode}")
        return

    print(f"Retrieved {len(docs)} docs in mode: {mode}")

    for doc in docs:
        print(f"Page {doc.metadata.get('page','?')}: {doc.page_content}")

    has_signal = has_bm25_signal("Narayanpet", bm25)
    print(f"\nBM25 signal for 'Narayanpet': {has_signal}")

    no_signal = has_bm25_signal("where is sai from", bm25)
    print(f"BM25 signal for 'where is sai from': {no_signal}")

    print("\nTest execution completed")


if __name__ == "__main__":
    test_hybrid_retrieval()