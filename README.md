# Ambitio Legal Document Pipeline
### AI Intern Assessment — Document Understanding, Grounded Drafting & Improvement from Edits

---

## What This System Does

Ingests messy legal-style documents (scanned PDFs, noisy text, handwritten annotations), extracts structured information, builds a retrieval index over it, generates grounded Case Fact Summary drafts anchored to source evidence, and improves future drafts by learning from operator edits.

---

## Setup

### Prerequisites
- Python 3.10+
- An Anthropic API key (for draft generation and preference extraction)
- Tesseract OCR (optional, for scanned PDFs): `sudo apt install tesseract-ocr`

### Install

```bash
git clone <repo-url>
cd ambitio-assessment
pip install -r requirements.txt
```

### Configure

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Generate sample documents

```bash
python samples/create_samples.py
```

This creates 3 synthetic messy legal documents: a lease agreement, a court notice, and an internal memo — all with intentional OCR noise, inconsistent formatting, and scan artifacts.

---

## Usage

### Process a single document
```bash
python main.py --file samples/lease_agreement.txt
```

### Process all sample documents
```bash
python main.py --all-samples
```

### Process + simulate operator edit (triggers learning loop)
```bash
python main.py --file samples/lease_agreement.txt --simulate-edit
```

### View learned operator preferences
```bash
python main.py --show-preferences
```

### Start REST API
```bash
python api.py
# → http://localhost:8000/docs for Swagger UI
```

### Run evaluation
```bash
python tests/evaluate.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status + index stats |
| POST | `/ingest` | Upload + extract + index a document |
| POST | `/draft` | Generate a Case Fact Summary |
| POST | `/evidence` | Retrieve relevant passages for a query |
| POST | `/feedback` | Submit operator edit → trigger learning |
| GET | `/preferences` | View all learned style preferences |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT DOCUMENTS                       │
│          (.txt, .pdf — messy, scanned, noisy)           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              1. INGESTION  (ingestion/extractor.py)      │
│                                                          │
│  pdfplumber text layer → clean_text() → field regex     │
│       ↓ (if text layer empty)                           │
│  pytesseract OCR → clean_text() → field regex           │
│                                                          │
│  Output: ExtractedDocument                               │
│    .raw_text        (cleaned full text)                  │
│    .pages[]         (per-page text)                      │
│    .structured_fields  {dates, amounts, parties, ...}   │
│    .extraction_method  text | ocr | hybrid               │
│    .confidence         high | medium | low               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│            2. RETRIEVAL  (retrieval/rag_pipeline.py)     │
│                                                          │
│  Chunker: 300-char overlapping windows (80-char overlap) │
│  Embedder: sentence-transformers/all-MiniLM-L6-v2        │
│  VectorStore: FAISS IndexFlatIP (cosine sim on L2-norm)  │
│                                                          │
│  Multi-query retrieval: 7 targeted section queries       │
│  → top-k chunks per query, deduped, sorted by score      │
│                                                          │
│  Output: RetrievedEvidence[]                             │
│    .chunk_id, .source_file, .page_num, .score, .text    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│           3. GENERATION  (generation/drafter.py)         │
│                                                          │
│  Prompt = system_prompt + evidence_block + style_rules   │
│  Model: claude-sonnet-4-20250514                        │
│                                                          │
│  Output: Draft (JSON-structured)                         │
│    sections[]:                                           │
│      .heading, .content (with [chunk_id] citations)      │
│      .supporting_chunk_ids[], .confidence                │
│                                                          │
│  Sections: Overview | Parties | Dates | Financial |      │
│            Obligations | Flags | Recommended Actions     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          4. FEEDBACK LOOP  (feedback/edit_learner.py)    │
│                                                          │
│  capture_edit(original, edited)                          │
│    → difflib unified diff                                │
│    → summarise_diff() → human-readable change summary    │
│    → Claude: "What preference does this edit reveal?"    │
│    → list[str] reusable rules                            │
│    → merged into learned_preferences.json                │
│      (deduped by word overlap ≥ 60%, occurrence-counted) │
│                                                          │
│  load_style_instructions(doc_type)                       │
│    → top-N rules by occurrence → injected into prompt    │
└─────────────────────────────────────────────────────────┘
```

---

## Assumptions & Tradeoffs

### Document Processing
- **Assumption**: Most inputs are text-layer PDFs or plain text; OCR is a fallback. In production with heavily scanned documents, Tesseract would need fine-tuning or replacement with a commercial OCR API (e.g. Google Document AI, AWS Textract).
- **Tradeoff**: `clean_text()` uses heuristic regex to fix OCR noise. This is fast but brittle for non-English scripts or unusual formatting. A learned denoiser would be more robust.
- **Tradeoff**: Structured field extraction uses regex patterns. SpaCy NER would be more robust but adds startup cost. Given the scope, regex gives adequate coverage for the document types shown.

### Retrieval
- **Choice**: FAISS flat inner-product on normalized vectors = exact cosine similarity. Fast enough for corpora up to ~100k chunks on CPU. For larger scale, FAISS IVF or HNSW would be needed.
- **Choice**: `all-MiniLM-L6-v2` — 384-dim, fast, good quality for semantic search. A domain-adapted legal embedding model (e.g. `legal-bert`) would improve relevance on real legal text.
- **Tradeoff**: Multi-query retrieval (7 targeted queries) gives broader coverage than single-query, at the cost of more embedding calls. These are batched and fast in practice.

### Draft Generation
- **Choice**: Case Fact Summary — clear structure, easy to ground evidence to sections, matches what a junior paralegal would produce as a first pass.
- **Tradeoff**: Forcing JSON output from the LLM is reliable with Claude Sonnet but can occasionally fail. The code has a fallback that stores raw output instead of crashing.
- **Design**: Each draft section carries `supporting_chunk_ids` — enabling an operator to click any claim and see the exact source passage. This is the core grounding guarantee.

### Improvement Loop
- **Design**: Instead of fine-tuning, we extract natural language preference rules from diffs and inject them into future system prompts. This is prompt-based few-shot learning — zero infrastructure cost, immediately effective, fully inspectable.
- **Tradeoff**: Word-overlap deduplication for merging preferences is simple. A production system would embed preferences and deduplicate by semantic similarity.
- **Assumption**: Operator edits are the ground truth signal. The system trusts that if an operator consistently changes something, it should be learned.

---

## Sample Outputs

### Extraction Output (lease_agreement.txt)
```json
{
  "source_file": "samples/lease_agreement.txt",
  "extraction_method": "text",
  "confidence": "high",
  "structured_fields": {
    "document_type": "Lease Agreement",
    "dates": ["March 15, 2024", "April 1, 2024", "March 31, 2025"],
    "amounts": ["Rs. 22,000/-", "Rs. 66,000/-", "Rs. 500/-"],
    "parties": ["Rajesh Mehta", "Ananya Sharma"]
  }
}
```

### Draft Section (with evidence citations)
```
## Financial Terms
Monthly rent is Rs. 22,000/- payable by the 5th of each month [abc123].
Security deposit of Rs. 66,000/- (3 months rent) is held by landlord [def456].
Late payment attracts a penalty of Rs. 500/- per day [abc123].
  [Evidence: abc123, def456]
```

### Learned Preference (after operator edit)
```
[2x] [Lease Agreement] Always express security deposit as a multiple of monthly rent.
[1x] [Court Notice] Always capture handwritten annotations as separate Adjournment Notes section.
```

---

## Evaluation Approach

Run `python tests/evaluate.py` to score all components automatically.

| Component | Max Points | Evaluated By |
|-----------|-----------|--------------|
| Document Processing | 25 | Extraction success, field count, entity detection |
| Retrieval & Grounding | 25 | Relevance of top-k, chunk ID traceability, source attribution |
| Draft Quality | 10 | Section count, citation presence, parseable JSON |
| Improvement from Edits | 25 | Diff capture, preference extraction, persistence |
| Code Quality | 10 | Module structure, docstrings, error handling, logging |
| Documentation | 5 | README completeness |

---

## File Structure

```
ambitio-assessment/
├── ingestion/
│   └── extractor.py          # OCR + text extraction, field parsing
├── retrieval/
│   └── rag_pipeline.py       # Chunking, embedding, FAISS, retrieval
├── generation/
│   └── drafter.py            # Grounded draft generation via Claude
├── feedback/
│   └── edit_learner.py       # Edit capture, preference extraction, style store
├── samples/
│   ├── create_samples.py     # Generates synthetic test documents
│   ├── lease_agreement.txt
│   ├── court_notice.txt
│   └── internal_memo.txt
├── tests/
│   └── evaluate.py           # Full evaluation harness
├── data/                     # Created at runtime
│   ├── faiss_index/          # Persisted vector store
│   ├── feedback/             # Edit log + learned preferences
│   └── outputs/              # Generated drafts
├── main.py                   # CLI orchestrator
├── api.py                    # FastAPI REST layer
├── requirements.txt
└── README.md
```
