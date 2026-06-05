"""
retrieval/rag_pipeline.py
--------------------------
Retrieval-Augmented Generation pipeline.

Steps:
  1. Chunker    — splits extracted text into overlapping passages
  2. Embedder   — encodes chunks via sentence-transformers
  3. VectorStore — FAISS index for fast similarity search
  4. Retriever  — top-k search + evidence packaging for generation

Each retrieved chunk carries metadata (source, page, char offsets) so
every generated claim can cite back to its exact source passage.
"""

import os
import json
import pickle
import hashlib
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Uses TF-IDF + LSA (local, no internet required).
# In production: swap Embedder for sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM    = 128                  # LSA dimensions
CHUNK_SIZE   = 300                  # characters per chunk
CHUNK_OVERLAP = 80                  # overlap to preserve context across boundaries
INDEX_DIR    = Path(__file__).parent.parent / "data" / "faiss_index"


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str          # deterministic hash
    source_file: str
    page_num: int
    char_start: int
    char_end: int
    text: str

    def to_dict(self):
        return asdict(self)


@dataclass
class RetrievedEvidence:
    chunk_id: str
    source_file: str
    page_num: int
    text: str
    score: float           # cosine similarity (0-1, higher = more relevant)
    rank: int


# ── Chunker ───────────────────────────────────────────────────────────────────
def chunk_document(doc) -> list[Chunk]:
    """
    Splits ExtractedDocument into overlapping fixed-size chunks.
    Respects page boundaries when possible.
    """
    chunks = []
    for page_num, page_text in enumerate(doc.pages):
        if not page_text.strip():
            continue
        start = 0
        while start < len(page_text):
            end = min(start + CHUNK_SIZE, len(page_text))
            # Try to end at a sentence boundary
            if end < len(page_text):
                for boundary in ['. ', '.\n', '\n\n', '\n']:
                    pos = page_text.rfind(boundary, start, end)
                    if pos != -1 and pos > start + CHUNK_SIZE // 2:
                        end = pos + len(boundary)
                        break
            text = page_text[start:end].strip()
            if text:
                chunk_id = hashlib.md5(
                    f"{doc.source_file}:{page_num}:{start}".encode()
                ).hexdigest()[:12]
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    source_file=doc.source_file,
                    page_num=page_num,
                    char_start=start,
                    char_end=end,
                    text=text,
                ))
            next_start = end - CHUNK_OVERLAP
            if next_start <= start:
                next_start = start + max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            start = next_start
            if start >= len(page_text):
                break
    return chunks


# ── Embedder ──────────────────────────────────────────────────────────────────
class Embedder:
    """
    Local TF-IDF + Latent Semantic Analysis embedder.
    No internet required. In production, swap with sentence-transformers
    (all-MiniLM-L6-v2) for better semantic understanding.
    """
    _instance = None

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.dim = EMBED_DIM
        self.tfidf = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.svd = TruncatedSVD(n_components=EMBED_DIM, random_state=42)
        self._fitted = False
        self._corpus: list[str] = []

    def fit(self, texts: list[str]):
        self._corpus = list(texts)
        tfidf_matrix = self.tfidf.fit_transform(texts)
        if tfidf_matrix.shape[0] < EMBED_DIM:
            self.svd = TruncatedSVD(
                n_components=max(1, tfidf_matrix.shape[0] - 1), random_state=42
            )
        self.svd.fit(tfidf_matrix)
        self._fitted = True
        self.dim = self.svd.n_components

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        tfidf_matrix = self.tfidf.transform(texts)
        vecs = self.svd.transform(tfidf_matrix).astype(np.float32)
        return normalize(vecs, norm="l2")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ── Vector Store ──────────────────────────────────────────────────────────────
class VectorStore:
    """
    FAISS flat inner-product index (equivalent to cosine sim on normalised vecs).
    Persists both the FAISS index and chunk metadata to disk.
    """

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path  = self.index_dir / "chunks.pkl"
        self.embedder   = Embedder.get()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[Chunk] = []

        if self.index_path.exists() and self.meta_path.exists():
            self._load()

    def _load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Loaded index: {len(self.chunks)} chunks")

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def add_document(self, doc):
        """Chunk a document, embed, and add to the index."""
        new_chunks = chunk_document(doc)
        if not new_chunks:
            logger.warning(f"No chunks from {doc.source_file}")
            return

        # Remove existing chunks for this file (re-indexing)
        self.chunks = [c for c in self.chunks if c.source_file != doc.source_file]
        all_chunks = self.chunks + new_chunks
        all_texts  = [c.text for c in all_chunks]

        # Refit embedder on full corpus (TF-IDF needs global vocabulary)
        self.embedder.fit(all_texts)
        all_vecs = self.embedder.embed(all_texts)

        self.index = faiss.IndexFlatIP(self.embedder.dim)
        self.index.add(all_vecs)
        self.chunks = all_chunks
        self._save()
        logger.info(f"Indexed {len(new_chunks)} chunks from {Path(doc.source_file).name}")

    def search(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        """Return top-k most relevant chunks for a query."""
        if self.index is None or len(self.chunks) == 0:
            return []
        q_vec = self.embedder.embed_one(query).reshape(1, -1)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_num=chunk.page_num,
                text=chunk.text,
                score=float(score),
                rank=rank + 1,
            ))
        return results

    def clear(self):
        self.index = None
        self.chunks = []
        for p in [self.index_path, self.meta_path]:
            if p.exists():
                p.unlink()


# ── Public API ────────────────────────────────────────────────────────────────
def build_index(docs: list) -> VectorStore:
    """Index a list of ExtractedDocuments."""
    store = VectorStore()
    store.clear()
    for doc in docs:
        store.add_document(doc)
    return store


def retrieve(query: str, store: VectorStore, top_k: int = 5) -> list[RetrievedEvidence]:
    """Retrieve relevant evidence chunks for a drafting query."""
    return store.search(query, top_k=top_k)


def format_evidence_block(evidence: list[RetrievedEvidence]) -> str:
    """Format retrieved evidence for injection into LLM prompt."""
    lines = []
    for e in evidence:
        src = Path(e.source_file).name
        lines.append(
            f"[EVIDENCE {e.rank} | source: {src} | page: {e.page_num+1} | "
            f"score: {e.score:.2f} | id: {e.chunk_id}]\n{e.text}"
        )
    return "\n\n---\n\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    from ingestion.extractor import extract

    sample_dir = Path(__file__).parent.parent / "samples"
    docs = []
    for f in sample_dir.glob("*.txt"):
        print(f"Extracting: {f.name}")
        docs.append(extract(str(f)))

    store = build_index(docs)
    print(f"\nIndex ready: {len(store.chunks)} chunks\n")

    query = "What is the monthly rent and security deposit?"
    results = retrieve(query, store, top_k=3)
    print(f"Query: {query}\n")
    print(format_evidence_block(results))
