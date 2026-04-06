
import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.loader import load_and_chunk
from app.retriever import build_vectorstore, build_bm25, hybrid_retrieve
from app.chain import build_rag_chain, build_streaming_chain
from app.memory import ConversationMemory
from app.logger import log_rag_call, log_upload, log_error, read_logs, get_stats
from app.schemas import (
    RAGResponse, UploadResponse,
    QuestionRequest, ChunkSource
)

app = FastAPI(
    title="Hybrid PDF Chatbot",
    description="Production RAG system with hybrid search, memory, and streaming",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

#Session Store
sessions: dict = {}


#health check

@app.get("/")
def health():
    return {
        "status":  "running",
        "version": "1.0.0",
        "sessions": len(sessions)
    }


#upload pdf

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and build RAG pipeline"""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks      = load_and_chunk(pdf_path)
        vectorstore = build_vectorstore(chunks)
        bm25        = build_bm25(chunks)
        chain       = build_rag_chain(vectorstore, bm25)

        session_id = file.filename.replace(".pdf","").replace(" ","_")

        sessions[session_id] = {
            "chain":       chain,
            "vectorstore": vectorstore,
            "bm25":        bm25,
            "filename":    file.filename,
            "chunk_count": len(chunks),
            "memory":      ConversationMemory(max_turns=10)
        }

        log_upload(session_id, file.filename, len(chunks))

    finally:
        os.remove(pdf_path)  # always cleanup even if error

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        chunks_indexed=len(chunks),
        message=f"Ready. Use session_id '{session_id}' to ask questions."
    )


#ask llm questions about pdf

@app.post("/ask", response_model=RAGResponse)
def ask(request: QuestionRequest):
    """Ask a question about uploaded PDF"""

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Session '{request.session_id}' not found")

    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    start = time.time()

    try:
        memory = session["memory"]

        answer = session["chain"].invoke({
            "question":     request.question,
            "chat_history": memory.get_history()
        })

        docs, mode = hybrid_retrieve(
            request.question,
            session["vectorstore"],
            session["bm25"]
        )

        latency_ms = (time.time() - start) * 1000

        # Log the call
        log_rag_call(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            chunks=[
                {"page": d.metadata.get("page","?"), "preview": d.page_content[:100]}
                for d in docs
            ],
            latency_ms=latency_ms,
            retrieval_mode=mode
        )

        # Save to memory
        memory.add_turn(request.question, answer)

        return RAGResponse(
            question=request.question,
            answer=answer,
            sources=[
                ChunkSource(
                    page=d.metadata.get("page","?"),
                    preview=d.page_content[:100]
                )
                for d in docs
            ],
            chunks_retrieved=len(docs),
            retrieval_mode=mode,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        log_error(request.session_id, request.question, str(e))
        raise HTTPException(500, f"Pipeline error: {str(e)}")


#Streaming

@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    """Stream answer token by token"""

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    streaming_chain = build_streaming_chain(
        session["vectorstore"],
        session["bm25"]
    )

    async def token_generator():
        async for token in streaming_chain.astream(request.question):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


#managing the memory

@app.delete("/sessions/{session_id}/memory")
def clear_memory(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session["memory"].clear()
    return {"message": f"Memory cleared for '{session_id}'"}


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"history": session["memory"].summary()}


#Sessions

@app.get("/sessions")
def list_sessions():
    return {
        "sessions": [
            {
                "session_id": sid,
                "filename":   s["filename"],
                "chunks":     s["chunk_count"]
            }
            for sid, s in sessions.items()
        ]
    }


#log and monitoring of the application

@app.get("/logs")
def get_logs(n: int = 20, log_type: str = None):
    """View recent logs"""
    return {
        "logs": read_logs(n=n, log_type=log_type)
    }

@app.get("/stats")
def get_statistics():
    """Get system performance stats"""
    return get_stats()