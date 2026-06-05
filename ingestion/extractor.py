"""
ingestion/extractor.py
----------------------
Handles text extraction from messy legal documents.
Supports: .txt, .pdf (text-layer), .pdf (scanned/image via OCR fallback).
Produces structured JSON output ready for downstream retrieval + drafting.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ── Optional heavy deps (graceful degradation) ──────────────────────────────
try:
    import pdfplumber
    PDF_PLUMBER_OK = True
except ImportError:
    PDF_PLUMBER_OK = False
    logger.warning("pdfplumber not available — PDF extraction disabled")

try:
    from PIL import Image
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False
    logger.warning("pytesseract/PIL not available — OCR fallback disabled")


# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class ExtractedDocument:
    """Structured output from document extraction stage."""
    source_file: str
    raw_text: str                          # full cleaned text
    pages: list[str] = field(default_factory=list)  # per-page text
    structured_fields: dict = field(default_factory=dict)  # extracted entities
    extraction_method: str = "unknown"     # text | ocr | hybrid
    confidence: str = "high"              # high | medium | low
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalise messy OCR/scan artifacts:
    - collapse excessive whitespace
    - fix common OCR substitutions (O→0, l→1 in numeric contexts)
    - strip scan noise markers
    - normalise line endings
    """
    if not text:
        return ""

    # Remove scan-noise annotations
    text = re.sub(r'\[scan noise[^\]]*\]', '', text, flags=re.IGNORECASE)

    # Normalise unicode whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix OCR digit confusion in obvious numeric contexts (Rs. 22,OOO → 22,000)
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)   # digit-O-digit → 0
    text = re.sub(r'(?<=Rs\. \d{2}),OOO', ',000', text)

    # Fix spaced-out words that are clearly one token (e.g. "AGREEM ENT")
    text = re.sub(r'([A-Z]{2,})\s{1,2}([A-Z]{2,})', r'\1\2', text)

    return text.strip()


# ── Field extraction ──────────────────────────────────────────────────────────
_PATTERNS = {
    "dates": re.compile(
        r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|'
        r'\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4}|'
        r'\w+\s+\d{1,2},\s+\d{4})\b',
        re.IGNORECASE
    ),
    "amounts": re.compile(
        r'Rs\.?\s*[\d,]+(?:/-)?(?:\s*\(.*?\))?|'
        r'Rupees\s+[A-Za-z\s]+(?:Only)?',
        re.IGNORECASE
    ),
    "case_numbers": re.compile(
        r'(?:Case\s+No\.?|CIV|CRIM|WP|CWP)[/\-\s]\d+[/\-\s]\d+',
        re.IGNORECASE
    ),
    "parties": re.compile(
        r'(?:Landlord|Tenant|Plaintiff|Defendant|FROM|TO)\s*:\s*(.+)',
        re.IGNORECASE
    ),
    "contract_ids": re.compile(
        r'(?:Contract|Agreement|Vendor)\s*(?:No\.?|#|ID)?\s*[A-Z0-9\-]+',
        re.IGNORECASE
    ),
    "flags": re.compile(
        r'\[FLAG[^\]]*\]',
        re.IGNORECASE
    ),
}

def extract_structured_fields(text: str) -> dict:
    """Pull key entities from legal document text."""
    fields = {}
    for field_name, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Deduplicate, clean up
            seen = []
            for m in matches:
                val = (m if isinstance(m, str) else m[0]).strip()
                if val and val not in seen:
                    seen.append(val)
            fields[field_name] = seen

    # Infer document type
    text_lower = text.lower()
    if any(w in text_lower for w in ["lease", "tenancy", "rent", "landlord", "tenant"]):
        fields["document_type"] = "Lease Agreement"
    elif any(w in text_lower for w in ["court", "plaintiff", "defendant", "hearing", "notice"]):
        fields["document_type"] = "Court Notice"
    elif any(w in text_lower for w in ["memorandum", "memo", "internal"]):
        fields["document_type"] = "Internal Memo"
    elif any(w in text_lower for w in ["contract", "vendor", "agreement"]):
        fields["document_type"] = "Contract / Agreement"
    else:
        fields["document_type"] = "Unknown"

    return fields


# ── Extraction strategies ─────────────────────────────────────────────────────
def extract_from_txt(path: str) -> ExtractedDocument:
    """Plain text file — direct read + clean."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    cleaned = clean_text(raw)
    return ExtractedDocument(
        source_file=path,
        raw_text=cleaned,
        pages=[cleaned],
        structured_fields=extract_structured_fields(cleaned),
        extraction_method="text",
        confidence="high",
    )


def extract_from_pdf_text_layer(path: str) -> Optional[ExtractedDocument]:
    """Try pdfplumber text extraction first (fast, high quality)."""
    if not PDF_PLUMBER_OK:
        return None
    try:
        pages_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                pages_text.append(clean_text(t))
        full_text = "\n\n".join(p for p in pages_text if p)
        if len(full_text.strip()) < 50:
            return None   # Likely scanned — fall through to OCR
        return ExtractedDocument(
            source_file=path,
            raw_text=full_text,
            pages=pages_text,
            structured_fields=extract_structured_fields(full_text),
            extraction_method="text",
            confidence="high",
        )
    except Exception as e:
        logger.warning(f"pdfplumber failed on {path}: {e}")
        return None


def extract_from_pdf_ocr(path: str) -> Optional[ExtractedDocument]:
    """OCR fallback for scanned PDFs using pytesseract."""
    if not OCR_OK or not PDF_PLUMBER_OK:
        return None
    try:
        import pdfplumber
        pages_text = []
        warnings = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                img = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                cleaned = clean_text(ocr_text)
                pages_text.append(cleaned)
                conf_data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT
                )
                confs = [c for c in conf_data["conf"] if c != -1]
                avg_conf = sum(confs) / len(confs) if confs else 0
                if avg_conf < 60:
                    warnings.append(f"Page {i+1}: low OCR confidence ({avg_conf:.0f}%)")

        full_text = "\n\n".join(p for p in pages_text if p)
        confidence = "low" if warnings else "medium"
        return ExtractedDocument(
            source_file=path,
            raw_text=full_text,
            pages=pages_text,
            structured_fields=extract_structured_fields(full_text),
            extraction_method="ocr",
            confidence=confidence,
            warnings=warnings,
        )
    except Exception as e:
        logger.error(f"OCR failed on {path}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────
def extract(file_path: str) -> ExtractedDocument:
    """
    Main entry point. Routes to the right strategy based on file type.
    Falls back gracefully: text-layer → OCR → error with partial result.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return extract_from_txt(str(path))

    elif suffix == ".pdf":
        doc = extract_from_pdf_text_layer(str(path))
        if doc:
            return doc
        logger.info(f"Text layer empty/thin for {path.name} — trying OCR")
        doc = extract_from_pdf_ocr(str(path))
        if doc:
            doc.extraction_method = "ocr"
            return doc
        # Last resort: empty doc with warning
        return ExtractedDocument(
            source_file=str(path),
            raw_text="",
            warnings=["Extraction failed: no text layer and OCR unavailable"],
            extraction_method="failed",
            confidence="low",
        )

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .pdf")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else "../samples/lease_agreement.txt"
    doc = extract(target)
    print(doc.to_json())
