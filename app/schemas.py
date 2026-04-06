# app/schemas.py
from pydantic import BaseModel
from typing import Optional

class ChunkSource(BaseModel):
    page: int | str
    preview: str
    score: Optional[float] = None

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChunkSource]
    chunks_retrieved: int
    retrieval_mode: str      
    latency_ms: float

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_indexed: int
    message: str

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class EvalScore(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    overall: float