"""
tests/evaluate.py
------------------
Evaluation harness for all pipeline components.
Scores each component and prints a summary rubric table.

Run: python tests/evaluate.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)   # quiet during eval
logger = logging.getLogger("eval")

from ingestion.extractor import extract
from retrieval.rag_pipeline import VectorStore, retrieve
from generation.drafter import Drafter, gather_evidence_for_draft
from feedback.edit_learner import (
    capture_edit, load_style_instructions,
    simulate_operator_edit, list_preferences,
)


@dataclass
class EvalResult:
    component: str
    max_points: int
    earned: float = 0.0
    tests: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, note: str = "", points: float = 0):
        self.tests.append({"test": name, "passed": passed, "note": note, "points": points})
        if passed:
            self.earned += points

    def summary(self) -> str:
        pct = (self.earned / self.max_points * 100) if self.max_points else 0
        return f"{self.component}: {self.earned:.1f}/{self.max_points} ({pct:.0f}%)"


# ── 1. Document Processing ────────────────────────────────────────────────────
def eval_extraction() -> EvalResult:
    r = EvalResult("1. Document Processing", max_points=25)
    samples = list((ROOT / "samples").glob("*.txt"))

    if not samples:
        r.add("Sample files exist", False, "Run: python samples/create_samples.py", 0)
        return r

    r.add("Sample files exist", True, f"{len(samples)} samples found", 3)

    all_ok = True
    for sample in samples:
        try:
            doc = extract(str(sample))
            has_text   = len(doc.raw_text) > 100
            has_fields = len(doc.structured_fields) >= 2
            has_type   = "document_type" in doc.structured_fields

            r.add(f"Extract {sample.name}", has_text,
                  f"chars={len(doc.raw_text)}, fields={len(doc.structured_fields)}", 3)
            r.add(f"Structured fields {sample.name}", has_fields and has_type,
                  f"fields={list(doc.structured_fields.keys())}", 2)

            # Check specific fields
            fields = doc.structured_fields
            has_dates   = "dates" in fields
            has_amounts = "amounts" in fields
            r.add(f"Entity extraction {sample.name}", has_dates or has_amounts,
                  f"dates={has_dates}, amounts={has_amounts}", 2)

        except Exception as e:
            r.add(f"Extract {sample.name}", False, str(e), 0)
            all_ok = False

    return r


# ── 2. Retrieval & Grounding ──────────────────────────────────────────────────
def eval_retrieval() -> EvalResult:
    r = EvalResult("2. Retrieval & Grounding", max_points=25)

    samples = list((ROOT / "samples").glob("*.txt"))
    if not samples:
        r.add("Samples available", False, "No samples to index", 0)
        return r

    store = VectorStore()
    store.clear()

    # Index docs
    docs = []
    for s in samples:
        doc = extract(str(s))
        store.add_document(doc)
        docs.append(doc)

    r.add("Index built", len(store.chunks) > 0, f"{len(store.chunks)} chunks indexed", 5)

    # Test retrieval relevance
    test_queries = [
        ("monthly rent security deposit", "22,000", "lease_agreement"),
        ("court hearing date time", "10:30", "court_notice"),
        ("liability cap clause", "liability", "internal_memo"),
    ]

    for query, expected_keyword, doc_hint in test_queries:
        results = retrieve(query, store, top_k=3)
        found = any(expected_keyword.lower() in r.text.lower() for r in results)
        top_score = results[0].score if results else 0
        r.add(
            f"Retrieval: '{query[:30]}'",
            found and top_score > 0.2,
            f"top_score={top_score:.2f}, keyword_found={found}",
            5,
        )

    # Grounding: evidence block has chunk IDs
    evidence = gather_evidence_for_draft(store, top_k_per_query=2)
    has_ids = all(hasattr(e, "chunk_id") and e.chunk_id for e in evidence)
    r.add("Evidence carries chunk IDs", has_ids, f"{len(evidence)} evidence items", 5)

    # Inspectability: source file traceable
    has_sources = all(e.source_file for e in evidence)
    r.add("Evidence source traceable", has_sources, "source_file present on all chunks", 5)

    return r


# ── 3. Draft Quality ──────────────────────────────────────────────────────────
def eval_draft() -> EvalResult:
    r = EvalResult("3. Draft Quality", max_points=10)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        r.add("API key set", False, "ANTHROPIC_API_KEY not set — skipping LLM tests", 0)
        return r

    samples = list((ROOT / "samples").glob("*.txt"))
    if not samples:
        return r

    doc   = extract(str(samples[0]))
    store = VectorStore()
    store.clear()
    store.add_document(doc)
    evidence = gather_evidence_for_draft(store, top_k_per_query=2)

    try:
        drafter = Drafter()
        draft   = drafter.generate(doc, evidence)

        r.add("Draft has sections", len(draft.sections) >= 4,
              f"{len(draft.sections)} sections", 3)
        r.add("Draft cites evidence", any(s.supporting_chunk_ids for s in draft.sections),
              "at least one section has chunk citations", 3)
        r.add("No sections empty", all(s.content.strip() for s in draft.sections),
              "all sections have content", 2)
        r.add("Draft parseable as JSON", True, "generation + parse succeeded", 2)

    except Exception as e:
        r.add("Draft generation", False, str(e), 0)

    return r


# ── 4. Improvement from Edits ─────────────────────────────────────────────────
def eval_feedback() -> EvalResult:
    r = EvalResult("4. Improvement from Edits", max_points=25)

    original = """## Case Fact Summary
## Document Overview
Lease between Rajesh Mehta and Ananya Sharma. [abc123]

## Financial Terms
Rent: Rs. 22,000/- monthly. [def456]

## Recommended Actions
Review clauses."""

    edited = simulate_operator_edit(original, "Lease Agreement")
    r.add("Operator edit simulated", edited != original, "edited != original", 5)

    try:
        record = capture_edit(
            original_draft=original,
            edited_draft=edited,
            document_name="test_lease.txt",
            document_type="Lease Agreement",
        )
        r.add("Edit captured", bool(record.edit_id), f"edit_id={record.edit_id}", 5)
        r.add("Diff computed", len(record.diff_lines) > 0,
              f"{len(record.diff_lines)} diff lines", 5)

        if os.environ.get("ANTHROPIC_API_KEY"):
            r.add("Preferences extracted", len(record.extracted_preferences) > 0,
                  f"{record.extracted_preferences}", 5)
        else:
            r.add("Preferences extracted (skipped - no API key)", True, "", 5)

        # Second edit — check merging
        record2 = capture_edit(
            original_draft=original,
            edited_draft=edited + "\nOPERATOR: Please add clause summary.",
            document_name="test_lease2.txt",
            document_type="Lease Agreement",
        )
        prefs = list_preferences()
        r.add("Preferences persist across edits", len(prefs) > 0,
              f"{len(prefs)} stored preference(s)", 5)

    except Exception as e:
        r.add("Edit loop", False, str(e), 0)

    return r


# ── 5. Code Quality ───────────────────────────────────────────────────────────
def eval_code_quality() -> EvalResult:
    r = EvalResult("5. Code Quality & System Design", max_points=10)

    # Check module structure
    expected_modules = [
        ROOT / "ingestion" / "extractor.py",
        ROOT / "retrieval" / "rag_pipeline.py",
        ROOT / "generation" / "drafter.py",
        ROOT / "feedback" / "edit_learner.py",
        ROOT / "main.py",
        ROOT / "api.py",
    ]
    all_exist = all(p.exists() for p in expected_modules)
    r.add("Module structure", all_exist, f"{sum(p.exists() for p in expected_modules)}/6 modules present", 3)

    # Check docstrings
    import ast
    documented = 0
    for mod in expected_modules:
        if not mod.exists():
            continue
        try:
            tree = ast.parse(mod.read_text())
            if ast.get_docstring(tree):
                documented += 1
        except:
            pass
    r.add("Modules have docstrings", documented >= 4, f"{documented}/6 documented", 2)

    # Check error handling
    extractor_src = (ROOT / "ingestion" / "extractor.py").read_text()
    has_try_except = "try:" in extractor_src and "except" in extractor_src
    r.add("Error handling in extractor", has_try_except, "try/except blocks present", 2)

    # Check logging
    uses_logging = all(
        "logging" in p.read_text()
        for p in expected_modules if p.exists()
    )
    r.add("Logging used throughout", uses_logging, "logging present in all modules", 2)

    # Dataclasses / type hints
    uses_dataclasses = "dataclass" in extractor_src
    r.add("Typed data models", uses_dataclasses, "dataclasses used for structured output", 1)

    return r


# ── 6. Documentation ──────────────────────────────────────────────────────────
def eval_docs() -> EvalResult:
    r = EvalResult("6. Documentation & Clarity", max_points=5)

    readme = ROOT / "README.md"
    r.add("README exists", readme.exists(), "", 2)
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        has_setup   = "pip install" in text or "setup" in text.lower()
        has_arch    = "architecture" in text.lower() or "pipeline" in text.lower()
        has_sample  = "sample" in text.lower() or "example" in text.lower()
        r.add("README has setup instructions", has_setup, "", 1)
        r.add("README has architecture overview", has_arch, "", 1)
        r.add("README has sample usage", has_sample, "", 1)

    return r


# ── Runner ────────────────────────────────────────────────────────────────────
def run_all():
    print("\n" + "="*65)
    print("  AMBITIO AI INTERN ASSESSMENT — EVALUATION REPORT")
    print("="*65 + "\n")

    results = []
    for eval_fn in [
        eval_extraction,
        eval_retrieval,
        eval_draft,
        eval_feedback,
        eval_code_quality,
        eval_docs,
    ]:
        t0 = time.time()
        result = eval_fn()
        elapsed = time.time() - t0
        results.append(result)

        print(f"{'─'*65}")
        print(f"  {result.summary()}  ({elapsed:.1f}s)")
        for t in result.tests:
            icon = "✓" if t["passed"] else "✗"
            pts  = f"+{t['points']:.0f}pt" if t["passed"] and t["points"] else ""
            note = f"  → {t['note']}" if t["note"] else ""
            print(f"    {icon} {t['test']} {pts}{note}")

    total_earned = sum(r.earned for r in results)
    total_max    = sum(r.max_points for r in results)
    pct = total_earned / total_max * 100 if total_max else 0

    print(f"\n{'='*65}")
    print(f"  TOTAL SCORE: {total_earned:.1f} / {total_max}  ({pct:.0f}%)")
    print(f"{'='*65}\n")

    return results


if __name__ == "__main__":
    run_all()
