# app/logger.py
import json
import os
from datetime import datetime
from pathlib import Path
from app.config import get_settings

settings = get_settings()

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

def log_rag_call(
    session_id: str,
    question: str,
    answer: str,
    chunks: list[dict],
    latency_ms: float,
    retrieval_mode: str
):
    entry = {
        "type":           "rag_call",
        "timestamp":      datetime.utcnow().isoformat(),
        "session_id":     session_id,
        "question":       question,
        "answer":         answer,
        "latency_ms":     round(latency_ms, 2),
        "retrieval_mode": retrieval_mode,
        "chunks_count":   len(chunks),
        "chunks":         [
            {"page": c.get("page","?"), "preview": c.get("preview","")}
            for c in chunks
        ]
    }
    _write(entry)

def log_upload(session_id: str, filename: str, chunk_count: int):
    entry = {
        "type":        "upload",
        "timestamp":   datetime.utcnow().isoformat(),
        "session_id":  session_id,
        "filename":    filename,
        "chunk_count": chunk_count
    }
    _write(entry)

def log_error(session_id: str, question: str, error: str):
    entry = {
        "type":       "error",
        "timestamp":  datetime.utcnow().isoformat(),
        "session_id": session_id,
        "question":   question,
        "error":      error
    }
    _write(entry)

def _write(entry: dict):
    with open(settings.log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def read_logs(n: int = 50, log_type: str = None) -> list[dict]:
    if not Path(settings.log_file).exists():
        return []
    with open(settings.log_file, "r") as f:
        lines = f.readlines()
    logs = [json.loads(line) for line in lines]
    if log_type:
        logs = [l for l in logs if l.get("type") == log_type]
    return logs[-n:]

def get_stats() -> dict:
    logs = read_logs(n=10000)
    rag_calls = [l for l in logs if l.get("type") == "rag_call"]
    errors    = [l for l in logs if l.get("type") == "error"]

    if not rag_calls:
        return {"total_calls": 0}

    latencies = [l["latency_ms"] for l in rag_calls]
    return {
        "total_calls":      len(rag_calls),
        "total_errors":     len(errors),
        "avg_latency_ms":   round(sum(latencies) / len(latencies), 2),
        "max_latency_ms":   round(max(latencies), 2),
        "min_latency_ms":   round(min(latencies), 2),
        "hybrid_calls":     sum(1 for l in rag_calls if l.get("retrieval_mode") == "hybrid"),
        "vector_only_calls": sum(1 for l in rag_calls if l.get("retrieval_mode") == "vector_only")
    }