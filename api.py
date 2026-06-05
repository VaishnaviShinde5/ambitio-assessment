"""
api.py
-------
Optional FastAPI layer exposing the pipeline as REST endpoints.

Endpoints:
  POST /ingest          — extract + index a document
  POST /draft           — generate draft for an indexed document
  POST /feedback        — submit operator edit, trigger learning
  GET  /preferences     — view learned style preferences
  GET  /evidence        — retrieve evidence for a query
  GET  /health          — health check
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ingestion.extractor import extract
from retrieval.rag_pipeline import VectorStore, retrieve, format_evidence_block
from generation.drafter import Drafter, gather_evidence_for_draft
from feedback.edit_learner import capture_edit, load_style_instructions, list_preferences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Ambitio Legal Document Pipeline",
    description="Ingest messy legal docs → extract → retrieve → draft → learn from edits",
    version="1.0.0",
)

# Shared state (in-memory for simplicity; production would use a DB)
_store = VectorStore()
_drafter = Drafter()
_docs: dict[str, object] = {}   # filename → ExtractedDocument


# ── Request / Response models ─────────────────────────────────────────────────
class DraftRequest(BaseModel):
    document_name: str
    top_k: int = 5

class FeedbackRequest(BaseModel):
    document_name: str
    original_draft: str
    edited_draft: str

class EvidenceRequest(BaseModel):
    query: str
    top_k: int = 5


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "indexed_docs": len(_docs), "chunks": len(_store.chunks)}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload a .txt or .pdf document for extraction and indexing."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".pdf"):
        raise HTTPException(400, "Only .txt and .pdf files are supported.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = extract(tmp_path)
        doc.source_file = file.filename   # restore original name
        _store.add_document(doc)
        _docs[file.filename] = doc
        return {
            "status": "indexed",
            "document": file.filename,
            "extraction_method": doc.extraction_method,
            "confidence": doc.confidence,
            "char_count": len(doc.raw_text),
            "structured_fields": doc.structured_fields,
            "warnings": doc.warnings,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/draft")
def generate_draft(req: DraftRequest):
    """Generate a Case Fact Summary for an already-indexed document."""
    if req.document_name not in _docs:
        raise HTTPException(404, f"Document '{req.document_name}' not indexed yet. POST to /ingest first.")

    doc = _docs[req.document_name]
    doc_type = doc.structured_fields.get("document_type", "Unknown")
    style_instructions = load_style_instructions(doc_type)
    evidence = gather_evidence_for_draft(_store, top_k_per_query=req.top_k)

    draft = _drafter.generate(doc, evidence, style_instructions=style_instructions)
    return draft.to_dict()


@app.post("/evidence")
def get_evidence(req: EvidenceRequest):
    """Retrieve relevant evidence chunks for a query."""
    if len(_store.chunks) == 0:
        raise HTTPException(400, "No documents indexed yet.")
    results = retrieve(req.query, _store, top_k=req.top_k)
    return {
        "query": req.query,
        "evidence": [
            {
                "rank": e.rank,
                "chunk_id": e.chunk_id,
                "source": Path(e.source_file).name,
                "page": e.page_num + 1,
                "score": round(e.score, 4),
                "text": e.text,
            }
            for e in results
        ],
    }


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit an operator-edited draft. Triggers preference learning."""
    if req.document_name not in _docs:
        raise HTTPException(404, f"Document '{req.document_name}' not found.")

    doc = _docs[req.document_name]
    doc_type = doc.structured_fields.get("document_type", "Unknown")

    record = capture_edit(
        original_draft=req.original_draft,
        edited_draft=req.edited_draft,
        document_name=req.document_name,
        document_type=doc_type,
    )
    return {
        "edit_id": record.edit_id,
        "extracted_preferences": record.extracted_preferences,
        "diff_line_count": len(record.diff_lines),
    }


@app.get("/preferences")
def get_preferences():
    """Return all learned operator style preferences."""
    prefs = list_preferences()
    return {"count": len(prefs), "preferences": prefs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
