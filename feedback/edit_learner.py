"""
feedback/edit_learner.py
-------------------------
Improvement loop from operator edits.
Uses Groq API (free) for preference extraction.
"""

import os
import json
import difflib
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

from groq import Groq

logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path(__file__).parent.parent / "data" / "feedback"
PREFS_FILE   = FEEDBACK_DIR / "learned_preferences.json"
EDITS_FILE   = FEEDBACK_DIR / "edit_log.jsonl"
GROQ_MODEL   = "llama-3.3-70b-versatile"

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EditRecord:
    edit_id: str
    document_name: str
    draft_type: str
    timestamp: str
    original_text: str
    edited_text: str
    diff_lines: list[str] = field(default_factory=list)
    extracted_preferences: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class LearnedPreference:
    rule: str
    source_edit_id: str
    document_type: str
    occurrences: int = 1
    last_seen: str = ""

    def to_dict(self):
        return asdict(self)


def compute_diff(original: str, edited: str) -> list[str]:
    orig_lines = original.splitlines(keepends=True)
    edit_lines = edited.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, edit_lines,
        fromfile="original_draft",
        tofile="operator_edit",
        lineterm="",
    ))
    return [
        line for line in diff
        if not (line.startswith(('+', '-')) and line[1:].strip() == "")
    ]


def summarise_diff(diff_lines: list[str]) -> str:
    added   = [l[1:].strip() for l in diff_lines if l.startswith('+') and not l.startswith('+++')]
    removed = [l[1:].strip() for l in diff_lines if l.startswith('-') and not l.startswith('---')]
    parts = []
    if removed:
        parts.append("REMOVED:\n" + "\n".join(f"  - {r}" for r in removed[:10]))
    if added:
        parts.append("ADDED:\n" + "\n".join(f"  + {a}" for a in added[:10]))
    return "\n".join(parts) if parts else "No meaningful changes detected."


EXTRACT_SYSTEM = """You are an AI assistant that analyses document editing patterns.
Given an original draft and an operator's edited version, extract reusable writing preferences.

Return ONLY a JSON array of preference strings. Each string should be:
- A concrete, actionable instruction (starts with a verb)
- Generalizable (not specific to this one document's values)
- Something that would improve ALL future drafts of this type

Example output:
["Always lead the Financial Terms section with total contract value before itemizing.",
 "Flag liability caps with explicit comparison to industry standard.",
 "Use Rs. X/- format for all Indian currency amounts."]

Return ONLY the JSON array, no explanation, no markdown."""

def extract_preferences_from_edit(original, edited, document_type, diff_summary) -> list[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return []
    client = Groq(api_key=api_key)
    prompt = f"""Document type: {document_type}

DIFF SUMMARY (what the operator changed):
{diff_summary}

ORIGINAL DRAFT:
{original[:2000]}

OPERATOR-EDITED VERSION:
{edited[:2000]}

Extract reusable preferences from this edit."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        prefs = json.loads(raw)
        return [str(p) for p in prefs if p] if isinstance(prefs, list) else []
    except Exception as e:
        logger.error(f"Preference extraction failed: {e}")
        return []


def _load_preferences() -> list[LearnedPreference]:
    if not PREFS_FILE.exists():
        return []
    with open(PREFS_FILE) as f:
        data = json.load(f)
    return [LearnedPreference(**p) for p in data]


def _save_preferences(prefs: list[LearnedPreference]):
    with open(PREFS_FILE, "w") as f:
        json.dump([p.to_dict() for p in prefs], f, indent=2)


def _append_edit_log(record: EditRecord):
    with open(EDITS_FILE, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def _merge_preferences(existing, new_rules, edit_id, doc_type):
    now = datetime.utcnow().isoformat()
    result = list(existing)
    for rule in new_rules:
        rule_lower = rule.lower()
        matched = False
        for pref in result:
            existing_words = set(pref.rule.lower().split())
            new_words = set(rule_lower.split())
            if len(existing_words) == 0:
                continue
            overlap = len(existing_words & new_words) / len(existing_words)
            if overlap > 0.6:
                pref.occurrences += 1
                pref.last_seen = now
                matched = True
                break
        if not matched:
            result.append(LearnedPreference(
                rule=rule,
                source_edit_id=edit_id,
                document_type=doc_type,
                occurrences=1,
                last_seen=now,
            ))
    return result


def capture_edit(original_draft, edited_draft, document_name, document_type, draft_type="Case Fact Summary"):
    edit_id = hashlib.md5(
        f"{document_name}:{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:10]

    diff_lines   = compute_diff(original_draft, edited_draft)
    diff_summary = summarise_diff(diff_lines)

    logger.info(f"Captured edit {edit_id} for {document_name}")

    new_prefs = []
    if diff_lines:
        new_prefs = extract_preferences_from_edit(
            original=original_draft,
            edited=edited_draft,
            document_type=document_type,
            diff_summary=diff_summary,
        )
        logger.info(f"Extracted {len(new_prefs)} preference(s)")

    record = EditRecord(
        edit_id=edit_id,
        document_name=document_name,
        draft_type=draft_type,
        timestamp=datetime.utcnow().isoformat(),
        original_text=original_draft,
        edited_text=edited_draft,
        diff_lines=diff_lines,
        extracted_preferences=new_prefs,
    )
    _append_edit_log(record)

    existing = _load_preferences()
    merged   = _merge_preferences(existing, new_prefs, edit_id, document_type)
    _save_preferences(merged)
    logger.info(f"Preference store now has {len(merged)} rule(s)")

    return record


def load_style_instructions(document_type: str = "", top_n: int = 5) -> str:
    prefs = _load_preferences()
    if not prefs:
        return ""
    filtered = [
        p for p in prefs
        if not document_type or not p.document_type
        or p.document_type.lower() == document_type.lower()
    ]
    filtered.sort(key=lambda p: p.occurrences, reverse=True)
    top = filtered[:top_n]
    if not top:
        return ""
    rules = "\n".join(f"- {p.rule} (reinforced {p.occurrences}x)" for p in top)
    return f"Apply these operator-preferred style rules:\n{rules}"


def list_preferences() -> list[dict]:
    return [p.to_dict() for p in _load_preferences()]


SIMULATED_EDITS = {
    "Lease Agreement": {
        "additions": [
            "Note: Security deposit is equivalent to 3 months rent.",
            "Late payment penalty: Rs. 500/- per day after 5th of month.",
        ],
    },
    "Court Notice": {
        "additions": [
            "Adjournment noted: Hearing rescheduled per handwritten annotation.",
        ],
    },
    "Internal Memo": {
        "additions": [
            "PRIORITY FLAGS: 3 clauses require immediate negotiation before April 10.",
        ],
    },
}

def simulate_operator_edit(draft_text: str, document_type: str) -> str:
    edits = SIMULATED_EDITS.get(document_type, {})
    edited = f"[OPERATOR REVIEWED — {datetime.utcnow().strftime('%Y-%m-%d')}]\n\n" + draft_text
    for addition in edits.get("additions", []):
        edited += f"\n\nOPERATOR NOTE: {addition}"
    edited = edited.replace("Recommended Actions", "Recommended Actions (OPERATOR: Prioritized)")
    return edited
