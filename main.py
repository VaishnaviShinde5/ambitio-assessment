"""
main.py
--------
End-to-end pipeline orchestrator for Ambitio legal document processing.

Usage:
  python main.py --file samples/lease_agreement.txt
  python main.py --file samples/court_notice.txt --simulate-edit
  python main.py --all-samples
  python main.py --show-preferences
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pipeline")

# Ensure project root is in path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ingestion.extractor import extract, ExtractedDocument
from retrieval.rag_pipeline import VectorStore, retrieve, format_evidence_block
from generation.drafter import Drafter, gather_evidence_for_draft, Draft
from feedback.edit_learner import (
    capture_edit, load_style_instructions,
    simulate_operator_edit, list_preferences,
)

OUTPUT_DIR = ROOT / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Pipeline stages ───────────────────────────────────────────────────────────
def stage_extract(file_path: str) -> ExtractedDocument:
    logger.info(f"[1/4] EXTRACTION: {Path(file_path).name}")
    doc = extract(file_path)
    logger.info(
        f"  → method={doc.extraction_method}, confidence={doc.confidence}, "
        f"chars={len(doc.raw_text)}, fields={list(doc.structured_fields.keys())}"
    )
    if doc.warnings:
        for w in doc.warnings:
            logger.warning(f"  ⚠ {w}")
    return doc


def stage_index(doc: ExtractedDocument, store: VectorStore) -> VectorStore:
    logger.info(f"[2/4] INDEXING into vector store")
    store.add_document(doc)
    logger.info(f"  → store now has {len(store.chunks)} chunks")
    return store


def stage_retrieve(store: VectorStore, doc_type: str) -> list:
    logger.info(f"[3/4] RETRIEVAL: gathering evidence for {doc_type}")
    evidence = gather_evidence_for_draft(store, top_k_per_query=3)
    logger.info(f"  → retrieved {len(evidence)} unique evidence chunks")
    for e in evidence[:3]:
        logger.info(f"  chunk {e.chunk_id} | score={e.score:.2f} | {e.text[:60]}...")
    return evidence


def stage_generate(doc: ExtractedDocument, evidence: list) -> Draft:
    logger.info(f"[4/4] DRAFT GENERATION")
    doc_type = doc.structured_fields.get("document_type", "Unknown")
    style_instructions = load_style_instructions(doc_type)
    if style_instructions:
        logger.info(f"  → applying {style_instructions.count(chr(10))+1} learned style rule(s)")
    drafter = Drafter()
    draft = drafter.generate(doc, evidence, style_instructions=style_instructions)
    logger.info(f"  → generated {len(draft.sections)} section(s)")
    return draft


def stage_feedback(draft: Draft, doc: ExtractedDocument, simulate: bool = False) -> None:
    if not simulate:
        return
    logger.info("[FEEDBACK] Simulating operator edit + learning")
    doc_type = doc.structured_fields.get("document_type", "Unknown")
    original_text = draft.to_readable()
    edited_text   = simulate_operator_edit(original_text, doc_type)
    record = capture_edit(
        original_draft=original_text,
        edited_draft=edited_text,
        document_name=Path(doc.source_file).name,
        document_type=doc_type,
    )
    logger.info(f"  → edit captured: {record.edit_id}")
    logger.info(f"  → learned preferences: {record.extracted_preferences}")


def save_outputs(doc: ExtractedDocument, draft: Draft) -> dict:
    """Save extraction JSON and draft to output directory."""
    stem = Path(doc.source_file).stem
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    extraction_path = OUTPUT_DIR / f"{stem}_extracted_{ts}.json"
    draft_json_path = OUTPUT_DIR / f"{stem}_draft_{ts}.json"
    draft_text_path = OUTPUT_DIR / f"{stem}_draft_{ts}.txt"

    with open(extraction_path, "w") as f:
        f.write(doc.to_json())

    with open(draft_json_path, "w") as f:
        f.write(draft.to_json())

    with open(draft_text_path, "w") as f:
        f.write(draft.to_readable())

    return {
        "extraction": str(extraction_path),
        "draft_json": str(draft_json_path),
        "draft_text": str(draft_text_path),
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(file_path: str, simulate_edit: bool = False) -> dict:
    store = VectorStore()

    doc      = stage_extract(file_path)
    store    = stage_index(doc, store)
    evidence = stage_retrieve(store, doc.structured_fields.get("document_type", ""))
    draft    = stage_generate(doc, evidence)

    print("\n" + "="*60)
    print(draft.to_readable())
    print("="*60 + "\n")

    paths = save_outputs(doc, draft)
    logger.info(f"Outputs saved: {paths}")

    stage_feedback(draft, doc, simulate=simulate_edit)

    return {"doc": doc, "draft": draft, "paths": paths}


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ambitio Legal Document Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single document (.txt or .pdf)")
    group.add_argument("--all-samples", action="store_true", help="Process all sample docs")
    group.add_argument("--show-preferences", action="store_true", help="Show learned style preferences")
    parser.add_argument("--simulate-edit", action="store_true",
                        help="Simulate an operator edit after drafting (triggers learning)")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set.")
        sys.exit(1)

    if args.show_preferences:
        prefs = list_preferences()
        if not prefs:
            print("No learned preferences yet. Run with --simulate-edit to generate some.")
        else:
            print(f"\n=== LEARNED OPERATOR PREFERENCES ({len(prefs)} rules) ===\n")
            for p in sorted(prefs, key=lambda x: x["occurrences"], reverse=True):
                print(f"[{p['occurrences']}x] [{p['document_type']}] {p['rule']}")
        return

    if args.all_samples:
        sample_dir = ROOT / "samples"
        files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.pdf"))
        if not files:
            print("No sample files found. Run: python samples/create_samples.py")
            sys.exit(1)
        for f in files:
            print(f"\n{'#'*60}")
            print(f"# Processing: {f.name}")
            print(f"{'#'*60}")
            try:
                run_pipeline(str(f), simulate_edit=args.simulate_edit)
            except Exception as e:
                logger.error(f"Failed on {f.name}: {e}", exc_info=True)
        return

    if args.file:
        run_pipeline(args.file, simulate_edit=args.simulate_edit)


if __name__ == "__main__":
    main()
