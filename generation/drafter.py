"""
generation/drafter.py
----------------------
Generates grounded legal-style draft outputs using Groq API (free).
Model: llama-3.3-70b-versatile (free on Groq)

Draft type: Case Fact Summary (structured, evidence-traceable).

Every section of the draft cites the exact evidence chunk IDs that
support it. Unsupported claims are explicitly flagged as [UNSUPPORTED].
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Draft output model ────────────────────────────────────────────────────────
@dataclass
class DraftSection:
    heading: str
    content: str
    supporting_chunk_ids: list[str] = field(default_factory=list)
    confidence: str = "supported"

@dataclass
class Draft:
    document_name: str
    draft_type: str
    sections: list[DraftSection] = field(default_factory=list)
    raw_llm_output: str = ""
    evidence_used: list[dict] = field(default_factory=list)
    model: str = GROQ_MODEL

    def to_dict(self):
        return asdict(self)

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_readable(self) -> str:
        lines = [
            f"{'='*60}",
            f"DRAFT: {self.draft_type.upper()}",
            f"Document: {self.document_name}",
            f"Model: {self.model}",
            f"{'='*60}\n",
        ]
        for sec in self.sections:
            lines.append(f"## {sec.heading}")
            lines.append(sec.content)
            if sec.supporting_chunk_ids:
                lines.append(f"  [Evidence: {', '.join(sec.supporting_chunk_ids)}]")
            lines.append("")
        lines.append(f"{'='*60}")
        lines.append(f"Evidence blocks used: {len(self.evidence_used)}")
        return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a legal document analyst generating structured case fact summaries.

RULES (strictly enforced):
1. Every factual claim MUST be grounded in the provided EVIDENCE blocks.
2. Cite evidence using its chunk ID in square brackets: [chunk_id].
3. If something cannot be found in the evidence, write [UNSUPPORTED] — do not invent.
4. Do not hallucinate dates, amounts, names, or legal terms.
5. Be concise and structured — this is a first-pass internal review document.

OUTPUT FORMAT (return valid JSON only, no markdown, no extra text):
{
  "sections": [
    {
      "heading": "Section Title",
      "content": "Content with [chunk_id] citations inline.",
      "supporting_chunk_ids": ["id1", "id2"],
      "confidence": "supported"
    }
  ]
}"""

def build_user_prompt(doc_name, doc_type, structured_fields, evidence_block, style_instructions=""):
    fields_summary = json.dumps(structured_fields, indent=2)
    style_section = f"\nSTYLE PREFERENCES FROM PRIOR EDITS:\n{style_instructions}\n" if style_instructions else ""

    return f"""Analyse the following legal document and produce a Case Fact Summary.

DOCUMENT NAME: {doc_name}
DOCUMENT TYPE: {doc_type}

EXTRACTED STRUCTURED FIELDS:
{fields_summary}

EVIDENCE PASSAGES (cite these by their chunk IDs):
{evidence_block}
{style_section}
Generate a Case Fact Summary with these sections:
1. Document Overview
2. Parties Involved
3. Key Dates & Deadlines
4. Financial Terms
5. Key Obligations & Clauses
6. Flags & Risk Items
7. Recommended Actions

Return ONLY valid JSON matching the output format. No markdown, no explanation."""


# ── Generator ─────────────────────────────────────────────────────────────────
class Drafter:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)

    def generate(self, extracted_doc, evidence: list, style_instructions: str = "") -> Draft:
        from retrieval.rag_pipeline import format_evidence_block

        doc_name = Path(extracted_doc.source_file).name
        doc_type = extracted_doc.structured_fields.get("document_type", "Unknown")
        evidence_block = format_evidence_block(evidence)

        prompt = build_user_prompt(
            doc_name=doc_name,
            doc_type=doc_type,
            structured_fields=extracted_doc.structured_fields,
            evidence_block=evidence_block,
            style_instructions=style_instructions,
        )

        logger.info(f"Generating draft for {doc_name} with {len(evidence)} evidence chunks")

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        raw_output = response.choices[0].message.content

        # Parse JSON response
        sections = []
        try:
            clean = raw_output.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])
            parsed = json.loads(clean)
            for s in parsed.get("sections", []):
                sections.append(DraftSection(
                    heading=s.get("heading", ""),
                    content=s.get("content", ""),
                    supporting_chunk_ids=s.get("supporting_chunk_ids", []),
                    confidence=s.get("confidence", "supported"),
                ))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON output: {e}")
            sections = [DraftSection(
                heading="Raw Output (parse error)",
                content=raw_output,
                confidence="unsupported",
            )]

        return Draft(
            document_name=doc_name,
            draft_type="Case Fact Summary",
            sections=sections,
            raw_llm_output=raw_output,
            evidence_used=[{"chunk_id": e.chunk_id, "score": e.score, "text": e.text[:100]} for e in evidence],
            model=GROQ_MODEL,
        )


# ── Retrieval queries for each section ───────────────────────────────────────
SECTION_QUERIES = [
    "document type parties involved overview",
    "landlord tenant plaintiff defendant names addresses",
    "dates deadlines commencement termination notice period",
    "rent amount payment security deposit financial terms",
    "obligations conditions clauses terms",
    "flags risks penalties termination liability",
    "recommended actions next steps",
]

def gather_evidence_for_draft(store, top_k_per_query: int = 3) -> list:
    from retrieval.rag_pipeline import retrieve
    seen_ids = set()
    all_evidence = []
    for query in SECTION_QUERIES:
        results = retrieve(query, store, top_k=top_k_per_query)
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                all_evidence.append(r)
    all_evidence.sort(key=lambda x: x.score, reverse=True)
    return all_evidence[:20]
